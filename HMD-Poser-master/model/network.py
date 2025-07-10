import torch
import torch.nn as nn
from torch.nn.functional import relu
import os
import torch.nn.utils.weight_norm as weightNorm
from functools import partial
from torch.nn import functional as F

class TemporalSpatialBackbone(torch.nn.Module):
    def __init__(self, input_dim, number_layer=3, hidden_size=256, dropout=0.05, nhead=8, block_num=2):
        super().__init__()
        assert input_dim == 18, "Invalid input dim. Expected 18 for AgilePoser."
        # --- 新设计 1: 物理感知的嵌入层 ---
        # 头部运动学(旋转+速度)的统一嵌入层
        self.head_kinematics_embedding = nn.Sequential(
            nn.Linear(12, hidden_size),  # 6D rot + 6D ang_vel = 12 dims
            nn.LeakyReLU()
        )
        # 耳部加速度的独立嵌入层
        self.ear_accel_embedding = nn.Sequential(
            nn.Linear(6, hidden_size),  # 3D left_acc + 3D right_acc = 6 dims
            nn.LeakyReLU()
        )

        # --- 新设计 2: 统一的归一化层 ---
        # 共享的LayerNorm，在特征映射到高维空间后使用
        self.shared_norm = nn.LayerNorm(hidden_size, eps=1e-6)

        self.num_block = block_num
        num_rnn_layer = 1

        # --- 新设计 3: 匹配新输入结构的时序编码器 ---
        # 现在我们只有两个并行的特征流
        self.time_encoder = nn.ModuleList(
            [nn.ModuleList([torch.nn.LSTM(hidden_size, hidden_size, num_rnn_layer,
                                          bidirectional=False, batch_first=True) for _ in range(2)]) for _ in
             range(self.num_block)]
        )

        encoder_layer = nn.TransformerEncoderLayer(hidden_size, nhead=nhead, batch_first=True)
        self.spatial_encoder = nn.ModuleList(
            [nn.TransformerEncoder(encoder_layer, num_layers=number_layer) for _ in range(self.num_block)]
        )

        # weight norm and initialization
        for encoder_i in self.time_encoder:
            for model_i in encoder_i:
                for layeridx in range(num_rnn_layer):
                    model_i = weightNorm( model_i, f"weight_ih_l{layeridx}"  )
                    model_i = weightNorm( model_i, f"weight_hh_l{layeridx}"  )
                for name, param in model_i.named_parameters():
                    if name.startswith("weight"):
                        torch.nn.init.orthogonal_(param)
    
    def forward(self, x_in, rnn_state=None):
        batch_size, time_seq = x_in.shape[0], x_in.shape[1]

        # 1. 按物理意义分解输入
        head_kinematics_raw = x_in[..., 0:12]  # [B, T, 12] (旋转+角速度)
        ear_accel_raw = x_in[..., 12:18]  # [B, T, 6]  (双耳加速度)

        # 2. 独立的非线性嵌入
        head_kinematics_emb = self.head_kinematics_embedding(head_kinematics_raw)
        ear_accel_emb = self.ear_accel_embedding(ear_accel_raw)

        # 3. 在高维空间进行统一归一化
        head_kinematics_norm = self.shared_norm(head_kinematics_emb)
        ear_accel_norm = self.shared_norm(ear_accel_emb)

        # 4. 堆叠成待处理的特征序列
        collect_feats = torch.stack([head_kinematics_norm, ear_accel_norm], dim=-2)  # [B, T, 2, hidden_size]

        # 5. 时序和空间编码
        for idx in range(self.num_block):
            collect_feats_temporal = []
            for idx_num in range(2):  # 循环范围是2
                temporal_out, _ = self.time_encoder[idx][idx_num](collect_feats[:, :, idx_num, :], None)
                collect_feats_temporal.append(temporal_out)

            collect_feats_temporal = torch.stack(collect_feats_temporal, dim=-2)

            # Reshape for Transformer
            reshaped_for_spatial = collect_feats_temporal.reshape(batch_size * time_seq, 2, -1)  # Token数量是2
            spatial_out = self.spatial_encoder[idx](reshaped_for_spatial)

            collect_feats = spatial_out.reshape(batch_size, time_seq, 2, -1)

        return collect_feats

class HMD_imu_HME_Universe(torch.nn.Module):
    def __init__(self, input_dim, number_layer=3, hidden_size=256, dropout=0.05, nhead=8, block_num=2):
        super().__init__()
        assert input_dim == 18, "invalid input dim"
        # 实例化我们修改好的Backbone
        self.backbone = TemporalSpatialBackbone(input_dim, number_layer, hidden_size, dropout, nhead, block_num)
        self.feature_dim = hidden_size * 2  # 或者根据你的backbone输出决定 (您的最新代码是2)
        # 解码器的输入维度现在由backbone的输出决定
        # 我们的backbone输出是 [B, T, 3, hidden_size]，所以展平后是 hidden_size*3
        self.pose_est_head = nn.Sequential(
                                nn.Linear(hidden_size * 2, 256),
                                nn.LeakyReLU(),
                                nn.Linear(256, 22 * 6)
            )

        self.shape_est_head = nn.Sequential(
                            nn.Linear(hidden_size * 2, 256),
                            nn.LeakyReLU(),
                            nn.Linear(256, 16)
            )

    def forward(self, x_in, rnn_state=None):
        # Backbone輸出原始特徵和姿態草稿
        aggregated_features = self.backbone(x_in, rnn_state)
        batch_size, time_seq = x_in.shape[0], x_in.shape[1]

        features_flat = aggregated_features.reshape(batch_size, time_seq, -1)

        pose_draft = self.pose_est_head(features_flat)
        pred_shapes = self.shape_est_head(features_flat)

        # 返回特徵和草稿，供HMDIMUModel進行後續修正
        return pose_draft, pred_shapes, aggregated_features


# --- 新增：一個更強大的、帶有上下文感知的修正網路 ---
class ContextualRefiner(nn.Module):
    """
    一個基於MLP的修正網路，它接收部位的姿態草稿和對應的原始特徵，來預測殘差。
    """

    def __init__(self, pose_part_dim, feature_dim, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(pose_part_dim + feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, pose_part_dim)
        )

    def forward(self, pose_part_flat, context_flat):
        """
        Args:
            pose_part_flat (torch.Tensor): 該部位的姿態草稿。
            context_flat (torch.Tensor): 所有上下文特征 (已拼接好)。
        """
        combined_input = torch.cat([pose_part_flat, context_flat], dim=-1)
        return self.network(combined_input)
    # --- 结束替换forward方法 ---
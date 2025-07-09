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
        # --- 修改 2: 彻底简化线性嵌入层 ---
        # 移除所有冗余的、针对不存在传感器的Embedding层（如脚、骨盆等）
        # 我们只为真正存在的信息源定义清晰的嵌入模块
        self.head_rot_embedding = nn.Sequential(nn.Linear(6, hidden_size), nn.LeakyReLU())
        self.head_vel_embedding = nn.Sequential(nn.Linear(6, hidden_size), nn.LeakyReLU())
        self.ear_acc_embedding = nn.Sequential(nn.Linear(6, hidden_size), nn.LeakyReLU())

        norm_layer=partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(hidden_size)
        
        self.num_block = block_num
        num_rnn_layer = 1
        # --- 修改 3: 调整时序编码器以匹配新的输入结构 ---
        # 我们的输入被分为3个部分（头旋转、头速度、耳加速度），所以LSTM列表长度为3
        self.time_encoder = nn.ModuleList(
            [nn.ModuleList([torch.nn.LSTM(hidden_size, hidden_size, num_rnn_layer,
                                          bidirectional=False, batch_first=True) for _ in range(3)]) for _ in
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
        # separate the input features
        batch_size, time_seq = x_in.shape[0], x_in.shape[1]
        # --- 修改 4: 简化特征分解，直接、清晰地处理18维输入 ---
        # 不再有复杂的、基于54维度的切片
        head_rot_feat = x_in[..., 0:6]   # [B, T, 6]
        head_vel_feat = x_in[..., 6:12]  # [B, T, 6]
        ear_acc_feat  = x_in[..., 12:18] # [B, T, 6]
        # 分别进行嵌入
        head_rot_emb = self.norm(self.head_rot_embedding(head_rot_feat))
        head_vel_emb = self.norm(self.head_vel_embedding(head_vel_feat))
        ear_acc_emb  = self.norm(self.ear_acc_embedding(ear_acc_feat))
        
        # 将嵌入后的特征堆叠，形成一个 [B, T, 3, hidden_size] 的张量
        collect_feats = torch.stack([head_rot_emb, head_vel_emb, ear_acc_emb], dim=-2)

        # --- 修改 5: 调整时空编码器的循环范围以匹配新的结构 ---
        # temporal and spatial backbone
        for idx in range(self.num_block):
            collect_feats_temporal = []
            # 循环范围从6改为3
            for idx_num in range(3):
                # 对每个部位的特征独立进行时序编码
                part_features = collect_feats[:, :, idx_num, :]
                temporal_out, _ = self.time_encoder[idx][idx_num](part_features, None)
                collect_feats_temporal.append(temporal_out)

            collect_feats_temporal = torch.stack(collect_feats_temporal, dim=-2)

            # 空间编码器现在融合3个部位的信息
            # Reshape for Transformer: [B*T, 3, hidden_size]
            reshaped_for_spatial = collect_feats_temporal.reshape(batch_size * time_seq, 3, -1)
            spatial_out = self.spatial_encoder[idx](reshaped_for_spatial)

            # Reshape back to [B, T, 3, hidden_size]
            collect_feats = spatial_out.reshape(batch_size, time_seq, 3, -1)

        return collect_feats

class HMD_imu_HME_Universe(torch.nn.Module):
    def __init__(self, input_dim, number_layer=3, hidden_size=256, dropout=0.05, nhead=8, block_num=2):
        super().__init__()
        assert input_dim == 18, "invalid input dim"
        # 实例化我们修改好的Backbone
        self.backbone = TemporalSpatialBackbone(input_dim, number_layer, hidden_size, dropout, nhead, block_num)

        # 解码器的输入维度现在由backbone的输出决定
        # 我们的backbone输出是 [B, T, 3, hidden_size]，所以展平后是 hidden_size*3
        self.pose_est_head = nn.Sequential(
                                nn.Linear(hidden_size * 3, 256),
                                nn.LeakyReLU(),
                                nn.Linear(256, 22 * 6)
            )

        self.shape_est_head = nn.Sequential(
                            nn.Linear(hidden_size * 3, 256),
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

    def forward(self, pose_part_flat, feature_part_flat):
        """
        Args:
            pose_part_flat (torch.Tensor): 該部位的姿態草稿。
            feature_part_flat (torch.Tensor): 主幹網絡輸出的、對應的原始特徵。

        Returns:
            torch.Tensor: 預測出的殘差。
        """
        combined_input = torch.cat([pose_part_flat, feature_part_flat], dim=-1)
        return self.network(combined_input)

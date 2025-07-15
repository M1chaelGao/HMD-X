import torch
import torch.nn as nn
from torch.nn.functional import relu
import os
import torch.nn.utils.weight_norm as weightNorm
from functools import partial
from torch.nn import functional as F


class UnifiedRefiner(nn.Module):
    def __init__(self, feature_dim, pose_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim + pose_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, pose_dim)
        )

    def forward(self, pose_draft, fused_features):
        # pose_draft: (B, T, pose_dim)
        # fused_features: (B, T, feature_dim)
        x = torch.cat([pose_draft, fused_features], dim=-1)
        pose_residual = self.mlp(x)
        return pose_residual


class TemporalSpatialBackbone(torch.nn.Module):
    def __init__(self, input_dim, number_layer=3, hidden_size=256, dropout=0.05, nhead=8, block_num=2):
        super().__init__()
        assert input_dim == 54, "invalid input dim"
        self.head1_linear_embeddings = nn.ModuleList([nn.Sequential(nn.Linear(6, 128), nn.LeakyReLU()),
                                                      nn.Sequential(nn.Linear(6, 128), nn.LeakyReLU())])
        self.lfoot_linear_embeddings = nn.ModuleList([nn.Sequential(nn.Linear(6, 128), nn.LeakyReLU()),
                                                      nn.Sequential(nn.Linear(6, 128), nn.LeakyReLU())])
        self.rfoot_linear_embeddings = nn.ModuleList([nn.Sequential(nn.Linear(6, 128), nn.LeakyReLU()),
                                                      nn.Sequential(nn.Linear(6, 128), nn.LeakyReLU())])
        self.pelvis_linear_embeddings = nn.ModuleList([nn.Sequential(nn.Linear(6, 128), nn.LeakyReLU()),
                                                       nn.Sequential(nn.Linear(6, 128), nn.LeakyReLU())])
        self.ear1_linear_embeddings = nn.ModuleList([nn.Sequential(nn.Linear(3, 256), nn.LeakyReLU())])
        self.ear2_linear_embeddings = nn.ModuleList([nn.Sequential(nn.Linear(3, 256), nn.LeakyReLU())])

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(hidden_size)

        self.num_block = block_num
        num_rnn_layer = 1
        self.time_encoder = nn.ModuleList(
            [nn.ModuleList([torch.nn.LSTM(hidden_size, hidden_size, num_rnn_layer,
                                          bidirectional=False, batch_first=True) for _ in range(6)]) for _ in
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
                    model_i = weightNorm(model_i, f"weight_ih_l{layeridx}")
                    model_i = weightNorm(model_i, f"weight_hh_l{layeridx}")
                for name, param in model_i.named_parameters():
                    if name.startswith("weight"):
                        torch.nn.init.orthogonal_(param)

    def forward(self, x_in, rnn_state=None):
        # separate the input features
        batch_size, time_seq = x_in.shape[0], x_in.shape[1]
        head_feats1 = [x_in[..., 0:6], x_in[..., 24:30]]
        lfoot_feats = [x_in[..., 6:12], x_in[..., 30:36]]
        rfoot_feats = [x_in[..., 12:18], x_in[..., 36:42]]
        pelvis_feats = [x_in[..., 18:24], x_in[..., 42:48]]
        ear1_feats = [x_in[..., 48:51]]
        ear2_feats = [x_in[..., 51:54]]

        # MLP embedding
        head1_emb = []
        for idx in range(len(head_feats1)):
            head1_emb.append(self.head1_linear_embeddings[idx](head_feats1[idx]))
        head1_emb = self.norm(torch.cat(head1_emb, dim=-1))
        lfoot_emb = []
        for idx in range(len(lfoot_feats)):
            lfoot_emb.append(self.lfoot_linear_embeddings[idx](lfoot_feats[idx]))
        lfoot_emb = self.norm(torch.cat(lfoot_emb, dim=-1))
        rfoot_emb = []
        for idx in range(len(rfoot_feats)):
            rfoot_emb.append(self.rfoot_linear_embeddings[idx](rfoot_feats[idx]))
        rfoot_emb = self.norm(torch.cat(rfoot_emb, dim=-1))
        pelvis_emb = []
        for idx in range(len(pelvis_feats)):
            pelvis_emb.append(self.pelvis_linear_embeddings[idx](pelvis_feats[idx]))
        pelvis_emb = self.norm(torch.cat(pelvis_emb, dim=-1))
        ear1_emb = []
        for idx in range(len(ear1_feats)):
            ear1_emb.append(self.ear1_linear_embeddings[idx](ear1_feats[idx]))
        ear1_emb = self.norm(torch.cat(ear1_emb, dim=-1))
        ear2_emb = []
        for idx in range(len(ear2_feats)):
            ear2_emb.append(self.ear2_linear_embeddings[idx](ear2_feats[idx]))
        ear2_emb = self.norm(torch.cat(ear2_emb, dim=-1))

        collect_feats = torch.stack([head1_emb, lfoot_emb, rfoot_emb, pelvis_emb, ear1_emb, ear2_emb], dim=-2).reshape(
            batch_size, time_seq, 6, -1)
        for idx in range(self.num_block):
            collect_feats_temporal = []
            for idx_num in range(6):
                collect_feats_temporal.append(self.time_encoder[idx][idx_num](collect_feats[:, :, idx_num, :], None)[0])
            collect_feats_temporal = torch.stack(collect_feats_temporal, dim=-2)
            collect_feats = self.spatial_encoder[idx](
                collect_feats_temporal.reshape(batch_size * time_seq, 6, -1)).reshape(batch_size, time_seq, 6, -1)
        return collect_feats


class HMD_imu_HME_Universe(torch.nn.Module):
    def __init__(self, input_dim, number_layer=3, hidden_size=256, dropout=0.05, nhead=8, block_num=2):
        super().__init__()
        assert input_dim == 54, "invalid input dim"
        self.backbone = TemporalSpatialBackbone(input_dim, number_layer, hidden_size, dropout, nhead, block_num)

        self.pose_est = nn.Sequential(
            nn.Linear(hidden_size * 6, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 22 * 6)
        )

        self.shape_est = nn.Sequential(
            nn.Linear(hidden_size * 6, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 16)
        )

    def forward(self, x_in, rnn_state=None, return_features=False):
        collect_feats = self.backbone(x_in, rnn_state)
        batch_size, time_seq = x_in.shape[0], x_in.shape[1]
        aggregated_features = collect_feats.reshape(batch_size, time_seq, -1)

        pred_pose = self.pose_est(aggregated_features)
        pred_shapes = self.shape_est(aggregated_features)

        if return_features:
            return pred_pose, pred_shapes, aggregated_features
        else:
            return pred_pose, pred_shapes


class ExpertRefiner(nn.Module):
    def __init__(self, feature_dim, pose_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim + pose_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, pose_dim)
        self.act = nn.LeakyReLU()

    def forward(self, x_pose, x_features):
        x = torch.cat([x_pose, x_features], dim=-1)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        residual = self.fc3(x)
        return residual


class MoEHead(nn.Module):
    def __init__(self, feature_dim, pose_dim, hidden_dim):
        super().__init__()
        self.static_expert = ExpertRefiner(feature_dim, pose_dim, hidden_dim)
        self.dynamic_expert = ExpertRefiner(feature_dim, pose_dim, hidden_dim)

    def forward(self, pose_draft_part, context_features, gate_signal):
        static_residual = self.static_expert(pose_draft_part, context_features)
        dynamic_residual = self.dynamic_expert(pose_draft_part, context_features)
        mixed_residual = gate_signal * dynamic_residual + (1 - gate_signal) * static_residual
        return mixed_residual


class PhysicsEncoder(nn.Module):
    def __init__(self, feature_dim=1536, hidden_dim=128, out_dim=2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=False
        )
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, out_dim),
            nn.Sigmoid()
        )

    def forward(self, features):
        # features: [B, T, feature_dim]
        gru_out, _ = self.gru(features)  # [B, T, hidden_dim]
        contact_probs = self.predictor(gru_out)  # [B, T, out_dim]
        return contact_probs

class GruMSGN(nn.Module):
    def __init__(self, feature_dim, hidden_dim, out_dim, num_layers=1):
        super().__init__()
        # 修改input_size为 feature_dim + 2
        self.gru = nn.GRU(
            input_size=feature_dim + 2,  # 拼接接触概率
            hidden_size=hidden_dim,
            batch_first=True,
            num_layers=num_layers,
            bidirectional=False
        )
        self.out_layer = nn.Linear(hidden_dim, out_dim)

    def forward(self, slow_features, contact_context):
        # slow_features: [B, T, feature_dim]
        # contact_context: [B, T, 2]
        x = torch.cat([slow_features, contact_context], dim=-1)  # [B, T, feature_dim+2]
        gru_out, _ = self.gru(x)
        out = self.out_layer(gru_out)
        return out
class TimeFPN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TimeFPN, self).__init__()
        # Bottom-up (主干网络)
        self.c1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.c2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, stride=2)
        self.c3 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, stride=2)

        # Top-down (横向连接)
        self.p3_conv = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.p2_conv = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.p1_conv = nn.Conv1d(out_channels, out_channels, kernel_size=1)

        # Smoothing layers
        self.p2_smooth = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.p1_smooth = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, features):
        # features: [B, T, C_in]
        # Permute to [B, C_in, T]
        x = features.permute(0, 2, 1)

        # Bottom-up pathway
        t1 = F.relu(self.c1(x))      # [B, C_out, T]
        t2 = F.relu(self.c2(t1))     # [B, C_out, T//2]
        t3 = F.relu(self.c3(t2))     # [B, C_out, T//4]

        # Top-down pathway + lateral connections
        p3 = self.p3_conv(t3)        # [B, C_out, T//4]
        p2 = self.p2_conv(t2) + F.interpolate(p3, size=t2.shape[2], mode='nearest')  # [B, C_out, T//2]
        p2 = F.relu(self.p2_smooth(p2))
        p1 = self.p1_conv(t1) + F.interpolate(p2, size=t1.shape[2], mode='nearest')  # [B, C_out, T]
        p1 = F.relu(self.p1_smooth(p1))

        # Permute back to [B, T, C_out]
        p1 = p1.permute(0, 2, 1)
        p2 = p2.permute(0, 2, 1)
        p3 = p3.permute(0, 2, 1)
        return [p1, p2, p3]
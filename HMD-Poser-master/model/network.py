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
        assert input_dim == 54, "invalid input dim"
        self.head1_linear_embeddings = nn.ModuleList([nn.Sequential(nn.Linear(6, 128), nn.LeakyReLU()),
                                                      nn.Sequential(nn.Linear(6, 128), nn.LeakyReLU())])
        # self.head2_linear_embeddings = nn.ModuleList([nn.Sequential(nn.Linear(6, hidden_size//4), nn.LeakyReLU()),
        #                             nn.Sequential(nn.Linear(6, hidden_size//4), nn.LeakyReLU()),
        #                             nn.Sequential(nn.Linear(3, hidden_size//4), nn.LeakyReLU()),
        #                             nn.Sequential(nn.Linear(3, hidden_size//4), nn.LeakyReLU())])
        # self.rhand_linear_embeddings = nn.ModuleList([nn.Sequential(nn.Linear(6, hidden_size//4), nn.LeakyReLU()),
        #                             nn.Sequential(nn.Linear(6, hidden_size//4), nn.LeakyReLU()),
        #                             nn.Sequential(nn.Linear(3, hidden_size//4), nn.LeakyReLU()),
        #                             nn.Sequential(nn.Linear(3, hidden_size//4), nn.LeakyReLU())])
        self.lfoot_linear_embeddings = nn.ModuleList([nn.Sequential(nn.Linear(6, 128), nn.LeakyReLU()),
                                                      nn.Sequential(nn.Linear(6, 128), nn.LeakyReLU())])
        self.rfoot_linear_embeddings = nn.ModuleList([nn.Sequential(nn.Linear(6, 128), nn.LeakyReLU()),
                                                      nn.Sequential(nn.Linear(6, 128), nn.LeakyReLU())])
        self.pelvis_linear_embeddings = nn.ModuleList([nn.Sequential(nn.Linear(6, 128), nn.LeakyReLU()),
                                                       nn.Sequential(nn.Linear(6, 128), nn.LeakyReLU())])
        self.ear1_linear_embeddings = nn.ModuleList([nn.Sequential(nn.Linear(3, 256), nn.LeakyReLU())])
        self.ear2_linear_embeddings = nn.ModuleList([nn.Sequential(nn.Linear(3, 256), nn.LeakyReLU())])
        # self.lhand_inhead_linear_embeddings = nn.ModuleList([nn.Sequential(nn.Linear(6, hidden_size//4), nn.LeakyReLU()),
        #                             nn.Sequential(nn.Linear(6, hidden_size//4), nn.LeakyReLU()),
        #                             nn.Sequential(nn.Linear(3, hidden_size//4), nn.LeakyReLU()),
        #                             nn.Sequential(nn.Linear(3, hidden_size//4), nn.LeakyReLU())])
        # self.rhand_inhead_linear_embeddings = nn.ModuleList([nn.Sequential(nn.Linear(6, hidden_size//4), nn.LeakyReLU()),
        #                             nn.Sequential(nn.Linear(6, hidden_size//4), nn.LeakyReLU()),
        #                             nn.Sequential(nn.Linear(3, hidden_size//4), nn.LeakyReLU()),
        #                             nn.Sequential(nn.Linear(3, hidden_size//4), nn.LeakyReLU())])

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
        head_feats1 = [x_in[..., 0:6], x_in[..., 24:30]]  # 提取特征？？feature embedding 18
        # head_feats2 = [x_in[..., 6:12], x_in[..., 36:42], x_in[..., 63:66], x_in[..., 69:72]]  # 提取特征？？feature embedding 18
        # lhand_feats = [x_in[..., 6:12], x_in[..., 42:48], x_in[..., 75:78], x_in[..., 84:87]]
        # rhand_feats = [x_in[..., 12:18], x_in[..., 48:54], x_in[..., 78:81], x_in[..., 87:90]]
        lfoot_feats = [x_in[..., 6:12], x_in[..., 30:36]]
        rfoot_feats = [x_in[..., 12:18], x_in[..., 36:42]]
        pelvis_feats = [x_in[..., 18:24], x_in[..., 42:48]]
        ear1_feats = [x_in[..., 48:51]]
        ear2_feats = [x_in[..., 51:54]]
        # lhand_inhead_feats = [x_in[..., 90:96], x_in[..., 102:108], x_in[..., 114:117], x_in[..., 120:123]]
        # rhand_inhead_feats = [x_in[..., 96:102], x_in[..., 108:114], x_in[..., 117:120], x_in[..., 123:126]]

        # MLP embedding
        head1_emb = []
        for idx in range(len(head_feats1)):
            head1_emb.append(self.head1_linear_embeddings[idx](head_feats1[idx]))
        head1_emb = self.norm(torch.cat(head1_emb, dim=-1))

        # head2_emb = []
        # for idx in range(len(head_feats2)):
        #     head2_emb.append(self.head2_linear_embeddings[idx](head_feats2[idx]))
        # head2_emb = self.norm(torch.cat(head2_emb, dim=-1))
        #
        # rhand_emb = []
        # for idx in range(len(rhand_feats)):
        #     rhand_emb.append(self.rhand_linear_embeddings[idx](rhand_feats[idx]))
        # rhand_emb = self.norm(torch.cat(rhand_emb, dim=-1))

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

        # lhand_inhead_emb = []
        # for idx in range(len(lhand_inhead_feats)):
        #     lhand_inhead_emb.append(self.lhand_inhead_linear_embeddings[idx](lhand_inhead_feats[idx]))
        # lhand_inhead_emb = self.norm(torch.cat(lhand_inhead_emb, dim=-1))
        #
        # rhand_inhead_emb = []
        # for idx in range(len(rhand_inhead_feats)):
        #     rhand_inhead_emb.append(self.rhand_inhead_linear_embeddings[idx](rhand_inhead_feats[idx]))
        # rhand_inhead_emb = self.norm(torch.cat(rhand_inhead_emb, dim=-1))

        collect_feats = torch.stack([head1_emb, lfoot_emb, rfoot_emb, pelvis_emb, ear1_emb, ear2_emb], dim=-2).reshape(
            batch_size, time_seq, 6, -1)  # （768,40,8,256）！！！！ 8个部分

        # temporal and spatial backbone
        for idx in range(self.num_block):
            collect_feats_temporal = []
            for idx_num in range(6):
                collect_feats_temporal.append(self.time_encoder[idx][idx_num](collect_feats[:, :, idx_num, :], None)[
                                                  0])  # 第0,1块idx的模块中的，第0~7个种类数据从里面选择执行
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

    def forward(self, x_in, rnn_state=None):
        collect_feats = self.backbone(x_in, rnn_state)  # 时空特征学习
        batch_size, time_seq = x_in.shape[0], x_in.shape[1]
        collect_feats = collect_feats.reshape(batch_size, time_seq, -1)  # 时空特征聚合

        pred_pose = self.pose_est(collect_feats)
        pred_shapes = self.shape_est(collect_feats)

        return pred_pose, pred_shapes

import numpy as np
import torch
import torch.nn as nn
from model.network import HMD_imu_HME_Universe, ExpertRefiner, MoEHead, GruMSGN, PhysicsEncoder
from utils import utils_transform
from human_body_prior.body_model.body_model import BodyModel
import os
from torch.nn.parallel import DataParallel, DistributedDataParallel

model_wraps = {
    "HMD_imu_HME_Universe": HMD_imu_HME_Universe,
}

def forward_kinematics_R(R_local: torch.Tensor, parent):
    R_local = R_local.view(R_local.shape[0], -1, 3, 3)
    R_global = _forward_tree(R_local, parent, torch.bmm)
    return R_global

def _forward_tree(x_local: torch.Tensor, parent, reduction_fn):
    x_global = [x_local[:, 0]]
    for i in range(1, len(parent)):
        x_global.append(reduction_fn(x_global[parent[i]], x_local[:, i]))
    x_global = torch.stack(x_global, dim=1)
    return x_global

class HMDIMUModel(nn.Module):
    def __init__(self, configs, device):
        super().__init__()
        self.netG = model_wraps[configs.model_name](
            configs.sparse_dim,
            configs.model_params.number_layer,
            configs.model_params.hidden_size,
            configs.model_params.dropout,
            configs.model_params.nhead,
            configs.model_params.block_num
            ).to(device)
        hidden_size = configs.model_params.hidden_size
        aggregated_feature_dim = hidden_size * 6

        # SMPL Joint Indexing
        self.trunk_joints = [0, 3, 6, 9]
        self.lower_joints = [1, 2, 4, 5, 7, 8, 10, 11]
        self.upper_joints = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21]

        self.trunk_dims = []
        for j in self.trunk_joints:
            self.trunk_dims.extend(list(range(j*6, (j+1)*6)))
        self.lower_dims = []
        for j in self.lower_joints:
            self.lower_dims.extend(list(range(j*6, (j+1)*6)))
        self.upper_dims = []
        for j in self.upper_joints:
            self.upper_dims.extend(list(range(j*6, (j+1)*6)))

        self.device = device
        self.trunk_dims = torch.tensor(self.trunk_dims, dtype=torch.long, device=self.device)
        self.lower_dims = torch.tensor(self.lower_dims, dtype=torch.long, device=self.device)
        self.upper_dims = torch.tensor(self.upper_dims, dtype=torch.long, device=self.device)

        # PhysicsEncoder for contact prediction
        self.physics_encoder = PhysicsEncoder(
            feature_dim=aggregated_feature_dim,
            hidden_dim=128,
            out_dim=2
        ).to(device)

        # Gating network and MoE modules
        self.gating_network = GruMSGN(
            feature_dim=aggregated_feature_dim,
            hidden_dim=256,
            out_dim=3,  # 这里用 out_dim 代表门的数量
            num_layers=1
        ).to(device)
        self.moe_trunk = MoEHead(
            feature_dim=aggregated_feature_dim,
            pose_dim=len(self.trunk_dims),
            hidden_dim=512
        ).to(device)
        self.moe_lower = MoEHead(
            feature_dim=aggregated_feature_dim,
            pose_dim=len(self.lower_dims),
            hidden_dim=512
        ).to(device)
        self.moe_upper = MoEHead(
            feature_dim=aggregated_feature_dim,
            pose_dim=len(self.upper_dims),
            hidden_dim=512
        ).to(device)

        support_dir = configs.support_dir
        subject_gender = "neutral"
        bm_fname = os.path.join(support_dir, 'smplh/{}/model.npz'.format(subject_gender))
        dmpl_fname = os.path.join(support_dir, 'dmpls/{}/model.npz'.format(subject_gender))
        num_betas = 16
        num_dmpls = 8
        self.bm = BodyModel(bm_fname=bm_fname, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname).to(device)

    def fk_module(self, global_orientation, joint_rotation, body_shape):
        global_orientation = utils_transform.sixd2aa(global_orientation.reshape(-1,6)).reshape(global_orientation.shape[0],-1).float()
        joint_rotation = utils_transform.sixd2aa(joint_rotation.reshape(-1,6)).reshape(joint_rotation.shape[0],-1).float()
        body_pose = self.bm(**{
            'root_orient':global_orientation,
            'pose_body':joint_rotation,
            'betas': body_shape.float()
        })
        joint_position = body_pose.Jtr[:, :22]
        return joint_position

    def get_bare_model(self, network):
        if isinstance(network, (DataParallel, DistributedDataParallel)):
            network = network.module
        return network

    def _controlled_fk(self, global_orientation, joint_rotation, body_shape, chunk_size=512):
        full_size = global_orientation.shape[0]
        if full_size <= chunk_size:
            return self.fk_module(global_orientation, joint_rotation, body_shape)
        results = []
        for i in range(0, full_size, chunk_size):
            end_idx = min(i + chunk_size, full_size)
            chunk_pos = self.fk_module(
                global_orientation[i:end_idx],
                joint_rotation[i:end_idx],
                body_shape[i:end_idx]
            )
            results.append(chunk_pos)
        return torch.cat(results, dim=0)

    def load_network(self, load_path, network, strict=True, param_key='state_dict'):
        network = self.get_bare_model(network)
        state_dict = torch.load(load_path)
        if param_key in state_dict.keys():
            state_dict = state_dict[param_key]
        network.load_state_dict(state_dict, strict=strict)

    def load(self, model_path, strict=True):
        self.load_network(model_path, self.netG, strict)

    def save(self, epoch, filename):
        network = self.get_bare_model(self.netG)
        torch.save({'state_dict': network.state_dict(),
            'epoch': epoch}, filename)

    def forward(self, sparse_input, do_fk=True):
        batch_size, time_length = sparse_input.shape[0], sparse_input.shape[1]
        pose_draft, pred_shapes, aggregated_features = self.netG(sparse_input, return_features=True)

        fast_features = aggregated_features
        slow_features = aggregated_features[:, ::2, :]

        # Predict contact probabilities for the whole sequence
        pred_contact_probs = self.physics_encoder(aggregated_features)  # [B, T, 2]

        # Downsample contact context to match slow_features' temporal stride
        contact_context_slow = pred_contact_probs[:, ::2, :]  # [B, T_slow, 2]

        # Upsample slow features to match time_length
        slow_features_upsampled = slow_features.repeat_interleave(2, dim=1)
        if slow_features_upsampled.shape[1] > time_length:
            slow_features_upsampled = slow_features_upsampled[:, :time_length, :]
        elif slow_features_upsampled.shape[1] < time_length:
            pad_shape = (slow_features_upsampled.shape[0], time_length - slow_features_upsampled.shape[1], slow_features_upsampled.shape[2])
            pad = torch.zeros(pad_shape, device=slow_features_upsampled.device, dtype=slow_features_upsampled.dtype)
            slow_features_upsampled = torch.cat([slow_features_upsampled, pad], dim=1)

        # Gating network: pass both slow_features and contact_context_slow
        gate_signals = self.gating_network(slow_features, contact_context_slow)
        gate_signals_upsampled = gate_signals.repeat_interleave(2, dim=1)
        if gate_signals_upsampled.shape[1] > time_length:
            gate_signals_upsampled = gate_signals_upsampled[:, :time_length, :]
        elif gate_signals_upsampled.shape[1] < time_length:
            pad_shape = (gate_signals_upsampled.shape[0], time_length - gate_signals_upsampled.shape[1], gate_signals_upsampled.shape[2])
            pad = torch.zeros(pad_shape, device=gate_signals_upsampled.device, dtype=gate_signals_upsampled.dtype)
            gate_signals_upsampled = torch.cat([gate_signals_upsampled, pad], dim=1)

        # Hierarchical MoE correction
        refined_pose = pose_draft.clone()
        residual_trunk = self.moe_trunk(
            refined_pose[:, :, self.trunk_dims],
            fast_features,
            gate_signals_upsampled[:, :, 0:1]
        )
        refined_pose[:, :, self.trunk_dims] += residual_trunk

        residual_lower = self.moe_lower(
            refined_pose[:, :, self.lower_dims],
            fast_features,
            gate_signals_upsampled[:, :, 1:2]
        )
        refined_pose[:, :, self.lower_dims] += residual_lower

        residual_upper = self.moe_upper(
            refined_pose[:, :, self.upper_dims],
            fast_features,
            gate_signals_upsampled[:, :, 2:3]
        )
        refined_pose[:, :, self.upper_dims] += residual_upper

        pred_pose = refined_pose

        rotation_local_matrot = utils_transform.sixd2matrot(pred_pose.reshape(-1, 6)).reshape(batch_size*time_length, 22, 3, 3)
        rotation_global_matrot = forward_kinematics_R(rotation_local_matrot, self.bm.kintree_table[0][:22].long()).view(batch_size, time_length, 22, 3, 3)
        rotation_global_r6d = utils_transform.matrot2sixd(rotation_global_matrot.reshape(-1, 3, 3)).reshape(batch_size, time_length, 22*6)
        if do_fk:
            pred_joint_position = self._controlled_fk(
                pred_pose[:, :, :6].reshape(-1, 6),
                pred_pose[:, :, 6:].reshape(-1, 21 * 6),
                pred_shapes.reshape(-1, 16)
            )
            pred_joint_position = pred_joint_position.reshape(batch_size, time_length, 22, 3)
            return pred_pose, pred_shapes, rotation_global_r6d, pred_joint_position, pred_contact_probs, gate_signals_upsampled
        else:
            return pred_pose, pred_shapes, rotation_global_r6d, pred_contact_probs, gate_signals_upsampled
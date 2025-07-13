import numpy as np
import torch
import torch.nn as nn
from model.network import HMD_imu_HME_Universe, UnifiedRefiner
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
        self.refiner = UnifiedRefiner(
            feature_dim=aggregated_feature_dim,
            pose_dim=22*6,
            hidden_dim=512
        ).to(device)

        support_dir = configs.support_dir
        subject_gender = "neutral"
        bm_fname = os.path.join(support_dir, 'smplh/{}/model.npz'.format(subject_gender))
        dmpl_fname = os.path.join(support_dir, 'dmpls/{}/model.npz'.format(subject_gender))
        num_betas = 16 # number of body parameters
        num_dmpls = 8 # number of DMPL parameters
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
        # Downsample by stride 2 for slow features
        slow_features = aggregated_features[:, ::2, :]

        # Upsample slow features to time_length
        # repeat_interleave: each slow frame covers 2 fast frames, so repeat 2 times
        slow_features_upsampled = slow_features.repeat_interleave(2, dim=1)
        # If time_length is odd, slow_features_upsampled may be longer, so truncate or pad
        if slow_features_upsampled.shape[1] > time_length:
            slow_features_upsampled = slow_features_upsampled[:, :time_length, :]
        elif slow_features_upsampled.shape[1] < time_length:
            pad_shape = (slow_features_upsampled.shape[0], time_length - slow_features_upsampled.shape[1], slow_features_upsampled.shape[2])
            pad = torch.zeros(pad_shape, device=slow_features_upsampled.device, dtype=slow_features_upsampled.dtype)
            slow_features_upsampled = torch.cat([slow_features_upsampled, pad], dim=1)

        fused_features = fast_features + slow_features_upsampled
        pose_residual = self.refiner(pose_draft, fused_features)
        pred_pose = pose_draft + pose_residual

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
            return pred_pose, pred_shapes, rotation_global_r6d, pred_joint_position
        else:
            return pred_pose, pred_shapes, rotation_global_r6d
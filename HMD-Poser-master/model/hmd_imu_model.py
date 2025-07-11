import numpy as np
import torch
import torch.nn as nn
from model.network import HMD_imu_HME_Universe, CnnMSGN
from utils import utils_transform
from human_body_prior.body_model.body_model import BodyModel
import os
from torch.nn.parallel import DataParallel, DistributedDataParallel
from .network import ContextualRefiner, MoEHead

model_wraps = {
    "HMD_imu_HME_Universe": HMD_imu_HME_Universe,
}


def _forward_tree(x_local: torch.Tensor, parent, reduction_fn):
    r"""
    Multiply/Add matrices along the tree branches. x_local [N, J, *]. parent [J].
    """
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

        support_dir = configs.support_dir
        subject_gender = "neutral"
        bm_fname = os.path.join(support_dir, 'smplh/{}/model.npz'.format(subject_gender))
        dmpl_fname = os.path.join(support_dir, 'dmpls/{}/model.npz'.format(subject_gender))
        num_betas = 16  # number of body parameters
        num_dmpls = 8  # number of DMPL parameters
        self.bm = BodyModel(bm_fname=bm_fname, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname).to(
            device)
        # --- 开始添加 ---
        torso_joints = [0, 1, 2, 3, 6, 9, 12, 13, 14, 15]  # 骨盆, 3個脊柱, 3個脖子/頭部
        upper_body_joints = [16, 17, 18, 19, 20, 21]  # 左右肩、肘、腕
        lower_body_joints = [4, 5, 7, 8, 10, 11]  # 左右臀、膝、踝

        self.part_indices = {}
        self.part_dims = {}
        self.agg_feature_dim = self.netG.feature_dim

        for part, joints in zip(['torso', 'upper', 'lower'], [torso_joints, upper_body_joints, lower_body_joints]):
            indices = torch.cat([torch.arange(j * 6, (j + 1) * 6) for j in joints])
            self.part_indices[part] = indices.long().to(device)
            self.part_dims[part] = len(indices)

        # --- 核心修正 2: 实例化具有正确输入维度的修正网络 ---
        self.torso_refiner = MoEHead(
            pose_part_dim=self.part_dims['torso'],  # 躯干姿态本身的维度
            feature_dim=self.agg_feature_dim,  # 全局特征的维度
            hidden_dim=256
        ).to(device)

        self.lower_refiner = MoEHead(
            pose_part_dim=self.part_dims['lower'],  # 下肢姿态本身的维度
            feature_dim=self.agg_feature_dim + self.part_dims['torso'],  # 全局特征 + 已修正的躯干姿态
            hidden_dim=256
        ).to(device)

        self.upper_refiner = MoEHead(
            pose_part_dim=self.part_dims['upper'],  # 上肢姿态本身的维度
            feature_dim=self.agg_feature_dim + self.part_dims['torso'] + self.part_dims['lower'],
            # 全局特征 + 修正后的躯干 + 修正后的下肢
            hidden_dim=256
        ).to(device)

        # --- 新增MSGN实例化 ---
        self.msgn = CnnMSGN(input_dim=18, output_dim=3).to(device)

    def forward_kinematics_R(self, R_local: torch.Tensor, parent):
        R_local = R_local.view(R_local.shape[0], -1, 3, 3)
        R_global = _forward_tree(R_local, parent, torch.bmm)
        return R_global

    # --- 核心修改 3: 新增/修改輔助函數 ---
    def _get_pose_part(self, full_pose_flat, part_name):
        return full_pose_flat[:, self.part_indices[part_name]]

    def _update_pose_part(self, full_pose_flat, part_pose, part_name):
        # 確保不會在計算圖中產生問題
        updated_pose = full_pose_flat.clone()
        updated_pose[:, self.part_indices[part_name]] = part_pose
        return updated_pose

    def fk_module(self, global_orientation, joint_rotation, body_shape):
        global_orientation = utils_transform.sixd2aa(global_orientation.reshape(-1, 6)).reshape(
            global_orientation.shape[0], -1).float()
        joint_rotation = utils_transform.sixd2aa(joint_rotation.reshape(-1, 6)).reshape(joint_rotation.shape[0],
                                                                                        -1).float()
        body_pose = self.bm(**{
            'root_orient': global_orientation,
            'pose_body': joint_rotation,
            'betas': body_shape.float()
        })
        joint_position = body_pose.Jtr[:, :22]
        return joint_position

    def get_bare_model(self, network):
        """Get bare model, especially under wrapping with
        DistributedDataParallel or DataParallel.
        """
        if isinstance(network, (DataParallel, DistributedDataParallel)):
            network = network.module
        return network

    def load_network(self, load_path, network, strict=True, param_key='state_dict'):
        network = self.get_bare_model(network)
        if strict:
            state_dict = torch.load(load_path)
            if param_key in state_dict.keys():
                state_dict = state_dict[param_key]
            network.load_state_dict(state_dict, strict=strict)
        else:
            pretrained_state_dict = torch.load(load_path)
            if param_key in pretrained_state_dict.keys():
                pretrained_state_dict = pretrained_state_dict[param_key]
            model_state_dict = network.state_dict()
            new_pretrained_state_dict = {}
            for k, v in pretrained_state_dict.items():
                if k in model_state_dict and v.shape == model_state_dict[k].shape:
                    new_pretrained_state_dict[k] = v
                elif '.'.join([k.split('.')[0], k]) in model_state_dict and v.shape == model_state_dict[
                    '.'.join([k.split('.')[0], k])].shape:
                    new_pretrained_state_dict['.'.join([k.split('.')[0], k])] = v
            not_inited_params = set(model_state_dict.keys()) - set(new_pretrained_state_dict.keys())
            if len(not_inited_params) > 0:
                print("Parameters [{}] were not inited".format(not_inited_params))
            network.load_state_dict(new_pretrained_state_dict, strict=False)
            del pretrained_state_dict, new_pretrained_state_dict, model_state_dict

    def load(self, model_path, strict=True):
        self.load_network(model_path, self.netG, strict)

    def save(self, epoch, filename):
        network = self.get_bare_model(self.netG)
        torch.save({'state_dict': network.state_dict(),
                    'epoch': epoch}, filename)

    def forward(self, sparse_input, raw_imu=None, do_fk=True):
        batch_size, time_length = sparse_input.shape[0], sparse_input.shape[1]

        # 步骤1: 获取原始姿态草稿
        pose_draft, pred_shapes, aggregated_features = self.netG(sparse_input)

        # --- 2. 新增：计算门控信号 ---
        gate_signals = None
        if raw_imu is not None:
            # raw_imu 的形状是 [B, T, 18]，正是CnnMSGN期望的
            gate_signals = self.msgn(raw_imu)

        # 切片门控信号，为三个部位各提取一个通道
        C_torso, C_lower, C_upper = gate_signals.split(1, dim=-1)

        pose_draft_flat = pose_draft.view(batch_size * time_length, -1)
        features_flat = aggregated_features.view(batch_size * time_length, -1)

        # 1. 修正躯干
        torso_draft_part = self._get_pose_part(pose_draft_flat, 'torso')
        # 直接将 pose 和 feature 传进去，让模块内部拼接
        torso_residual = self.torso_refiner(torso_draft_part, features_flat, C_torso)
        refined_torso_part = torso_draft_part + torso_residual
        pose_after_torso = self._update_pose_part(pose_draft_flat, refined_torso_part, 'torso')

        # 2. 修正下肢
        lower_draft_part = self._get_pose_part(pose_after_torso, 'lower')
        lower_context = torch.cat([features_flat, refined_torso_part], dim=1)
        lower_residual = self.lower_refiner(lower_draft_part, lower_context, C_lower)
        refined_lower_part = lower_draft_part + lower_residual
        pose_after_lower = self._update_pose_part(pose_after_torso, refined_lower_part, 'lower')

        # 3. 修正上肢
        upper_draft_part = self._get_pose_part(pose_after_lower, 'upper')
        refined_lower_part_from_after_lower = self._get_pose_part(pose_after_lower, 'lower')
        upper_context = torch.cat([features_flat, refined_torso_part, refined_lower_part_from_after_lower], dim=1)
        upper_residual = self.upper_refiner(upper_draft_part, upper_context, C_upper)
        refined_upper_part = upper_draft_part + upper_residual
        refined_pose_flat = self._update_pose_part(pose_after_lower, refined_upper_part, 'upper')
        # --- 修改结束 ---

        refined_pose = refined_pose_flat.view(batch_size, time_length, -1)

        # 后续FK计算
        rotation_local_matrot = utils_transform.sixd2matrot(refined_pose.reshape(-1, 6)).reshape(
            batch_size * time_length,
            22, 3, 3)
        rotation_global_matrot = self.forward_kinematics_R(rotation_local_matrot,
                                                           self.bm.kintree_table[0][:22].long()).view(batch_size,
                                                                                                      time_length, 22,
                                                                                                      3, 3)

        rotation_global_r6d = utils_transform.matrot2sixd(rotation_global_matrot.reshape(-1, 3, 3)).reshape(batch_size,
                                                                                                            time_length,
                                                                                                            22 * 6)
        if do_fk:
            pred_joint_position = self.fk_module(
                refined_pose[:, :, :6].reshape(-1, 6),
                refined_pose[:, :, 6:].reshape(-1, 21 * 6),
                pred_shapes.reshape(-1, 16)
            )
            pred_joint_position = pred_joint_position.reshape(batch_size, time_length, 22, 3)
            return refined_pose, pose_draft, pred_shapes, rotation_global_r6d, pred_joint_position, gate_signals
        else:
            return refined_pose, pose_draft, pred_shapes, gate_signals
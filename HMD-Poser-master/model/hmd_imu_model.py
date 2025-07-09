import numpy as np
import torch
import torch.nn as nn
from model.network import HMD_imu_HME_Universe
from utils import utils_transform
from human_body_prior.body_model.body_model import BodyModel
import os
from torch.nn.parallel import DataParallel, DistributedDataParallel

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
        num_betas = 16 # number of body parameters
        num_dmpls = 8 # number of DMPL parameters
        self.bm = BodyModel(bm_fname=bm_fname, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname).to(device)
    def forward_kinematics_R(self, R_local: torch.Tensor, parent):
        R_local = R_local.view(R_local.shape[0], -1, 3, 3)
        R_global = _forward_tree(R_local, parent, torch.bmm)
        return R_global
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
                elif '.'.join([k.split('.')[0], k]) in model_state_dict and v.shape == model_state_dict['.'.join([k.split('.')[0], k])].shape:
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

    def forward(self, sparse_input, do_fk=True):
        """
        模型的完整前向传播。
        当 do_fk=False 时，只执行低成本的网络推理，不进行耗费显存的前向动力学计算。
        """
        batch_size, time_length = sparse_input.shape[0], sparse_input.shape[1]

        # 步骤1: 低成本的网络推理，得到姿态和体型参数
        pred_pose, pred_shapes = self.netG(sparse_input)
        # --- 核心修改 2: 在 forward 函数内部，使用 self 来调用 ---
        rotation_local_matrot = utils_transform.sixd2matrot(pred_pose.reshape(-1, 6)).reshape(batch_size * time_length,
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
                pred_pose[:, :, :6].reshape(-1, 6),
                pred_pose[:, :, 6:].reshape(-1, 21*6),
                pred_shapes.reshape(-1, 16)
            )
            pred_joint_position = pred_joint_position.reshape(batch_size, time_length, 22, 3)
            return pred_pose, pred_shapes, rotation_global_r6d, pred_joint_position
        else:
            # 当不做FK时，我们只返回基础的网络输出
            return pred_pose, pred_shapes

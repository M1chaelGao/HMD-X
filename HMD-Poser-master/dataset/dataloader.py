import glob
import os

import torch
from torch.utils.data import Dataset


def get_path(dataset_path, split):
    data_list_path = []
    parent_data_path = glob.glob(dataset_path + "/*")
    for d in parent_data_path:
        if os.path.isdir(d):
            if os.path.exists(d + "/" + split):
                files = glob.glob(d + "/" + split + "/*pt")
                if len(files) > 0:
                    data_list_path.extend(files)
    return data_list_path


def load_data(dataset_path, split, **kwargs):
    motion_list = get_path(dataset_path, split)
    input_motion_length = kwargs["input_motion_length"]
    motion_raw_data = []
    for data_i in motion_list:
        motion_raw_data.extend(torch.load(data_i))

    valid_motion_data = []
    for data in motion_raw_data:
        if split == "train" and data["rotation_local_full_gt_list"].shape[0] < input_motion_length:
            continue
        # 这里的数据已经是我们需要的18维数据了
        input_feat = data["hmd_position_global_full_gt_list"]
        gt_local_pose = data["rotation_local_full_gt_list"]
        gt_global_pose = data["rotation_global_full_gt_list"]
        gt_positions = data["position_global_full_gt_world"]
        gt_betas = data["body_parms_list"]['betas'][1:]

        valid_motion_data.append({
            'input_feat': input_feat,
            'gt_local_pose': gt_local_pose,
            'gt_global_pose': gt_global_pose,
            'gt_positions': gt_positions,
            'gt_betas': gt_betas
        })
    return valid_motion_data


class TrainDataset(Dataset):
    def __init__(
            self,
            train_datas,
            compatible_inputs,  # 这个参数虽然保留，但在新逻辑中不再使用
            input_motion_length=40,
            train_dataset_repeat_times=1,
    ):
        self.train_dataset_repeat_times = train_dataset_repeat_times
        self.input_motion_length = input_motion_length
        self.motions = train_datas

    def __len__(self):
        return len(self.motions) * self.train_dataset_repeat_times

    def __getitem__(self, idx):
        motion_data = self.motions[idx % len(self.motions)]
        seqlen = motion_data['input_feat'].shape[0]

        if seqlen <= self.input_motion_length:
            start_idx = 0
        else:
            start_idx = torch.randint(0, int(seqlen - self.input_motion_length), (1,))[0]

        end_idx = start_idx + self.input_motion_length

        # --- 核心修改：直接切片，不再有任何置零或堆叠操作 ---
        input_feat = motion_data['input_feat'][start_idx: end_idx]
        gt_local_pose = motion_data['gt_local_pose'][start_idx: end_idx]
        gt_global_pose = motion_data['gt_global_pose'][start_idx: end_idx]
        gt_positions = motion_data['gt_positions'][start_idx: end_idx]
        gt_betas = motion_data['gt_betas'][start_idx: end_idx]

        # --- 新增：在这里加入我们为MSGN生成的状态伪标签 ---
        # (这部分逻辑我们在第一周冲刺计划中添加)
        # motion_state_labels = self.calculate_motion_state(gt_positions)

        # 直接返回张量，不再需要为HMD, HMD_2IMUs等模式进行堆叠
        # return input_feat, gt_local_pose, gt_global_pose, gt_positions, gt_betas, motion_state_labels
        return input_feat, gt_local_pose, gt_global_pose, gt_positions, gt_betas


class TestDataset(Dataset):
    def __init__(
            self,
            train_datas,
            compatible_inputs,  # 这个参数虽然保留，但在新逻辑中不再使用
            input_motion_length=40,
            train_dataset_repeat_times=1,
    ):
        self.input_motion_length = input_motion_length
        self.motions = train_datas

    def __len__(self):
        return len(self.motions)

    def __getitem__(self, idx):
        motion_data = self.motions[idx]

        # --- 核心修改：直接返回完整的序列，移除所有模拟输入的逻辑 ---
        input_feat = motion_data['input_feat']
        gt_local_pose = motion_data['gt_local_pose']
        gt_global_pose = motion_data['gt_global_pose']
        gt_positions = motion_data['gt_positions']
        gt_betas = motion_data['gt_betas']

        # 同样，后续可以在这里加入为整个测试序列生成的状态伪标签

        # 直接返回，不再有stack操作
        return input_feat, gt_local_pose, gt_global_pose, gt_positions, gt_betas
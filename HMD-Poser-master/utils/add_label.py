import torch
import os
import glob
from tqdm import tqdm


def get_all_pt_files(dataset_path):
    """查找指定路径下所有的.pt文件"""
    all_files = []
    # 确保路径拼接是正确的
    search_pattern = os.path.join(dataset_path, "**", "*.pt")
    all_files = glob.glob(search_pattern, recursive=True)
    return all_files


def add_motion_state_labels_to_file(file_path):
    """为单个.pt文件中的所有数据添加状态标签"""
    try:
        # 加载.pt文件，它应该是一个包含多个字典的列表
        motion_data_list = torch.load(file_path)

        # 检查是否是列表，如果不是，则放入列表中以便统一处理
        if not isinstance(motion_data_list, list):
            motion_data_list = [motion_data_list]

        updated_data_list = []
        for data_dict in motion_data_list:
            # --- 核心修改：使用正确的键名来获取关节位置数据 ---
            if 'position_global_full_gt_world' not in data_dict:
                print(f"Skipping a data entry in {file_path} due to missing key 'position_global_full_gt_world'")
                continue

            gt_positions = data_dict['position_global_full_gt_world']
            # --- 修改结束 ---

            # 计算帧间速度
            gt_velocities = torch.zeros_like(gt_positions)
            if gt_positions.shape[0] > 1:
                gt_velocities[1:] = gt_positions[1:] - gt_positions[:-1]

            gt_speeds = torch.linalg.norm(gt_velocities, dim=-1)

            torso_joints = [0, 1, 2, 3, 6, 9, 12, 13, 14, 15]
            lower_limbs_joints = [4, 5, 7, 8, 10, 11]
            upper_limbs_joints = [16, 17, 18, 19, 20, 21]

            velocity_threshold = 0.05

            # 注意：我们的MSGN需要的是“静止概率”，所以标签0代表静止，1代表运动
            is_static_torso = (torch.mean(gt_speeds[:, torso_joints], dim=-1) < velocity_threshold).long()
            is_static_lower = (torch.mean(gt_speeds[:, lower_limbs_joints], dim=-1) < velocity_threshold).long()
            is_static_upper = (torch.mean(gt_speeds[:, upper_limbs_joints], dim=-1) < velocity_threshold).long()

            motion_state_labels = torch.stack([is_static_torso, is_static_lower, is_static_upper], dim=-1)

            data_dict['motion_state_labels'] = motion_state_labels
            updated_data_list.append(data_dict)

        # 覆盖保存更新后的数据
        torch.save(updated_data_list, file_path)

        return True
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return False


def main():
    dataset_root = "/home/4T-2/gyf/AgilePoserdataV1/"

    print(f"Starting to upgrade dataset at: {dataset_root}")

    all_pt_files = get_all_pt_files(dataset_root)

    if not all_pt_files:
        print("No .pt files found. Please check your dataset_root path.")
        return

    print(f"Found {len(all_pt_files)} .pt files to process.")

    for file_path in tqdm(all_pt_files, desc="Upgrading .pt files"):
        add_motion_state_labels_to_file(file_path)

    print("Dataset upgrade complete!")


if __name__ == "__main__":
    main()
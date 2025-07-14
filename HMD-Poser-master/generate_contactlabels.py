import torch
import numpy as np
from tqdm import tqdm

# 1. 硬编码需要处理的所有.pt文件绝对路径
target_files = [
    '/home/4T-2/gyf/ProcessedHMDdata/MPI_HDM05/train/193.pt',
    '/home/4T-2/gyf/ProcessedHMDdata/MPI_HDM05/test/22.pt',
    '/home/4T-2/gyf/ProcessedHMDdata/BioMotionLab_NTroje/train/2754.pt',
    '/home/4T-2/gyf/ProcessedHMDdata/BioMotionLab_NTroje/test/307.pt',
    '/home/4T-2/gyf/ProcessedHMDdata/CMU/train/1867.pt',
    '/home/4T-2/gyf/ProcessedHMDdata/CMU/test/207.pt'
    # ... 添加其他所有需要处理的文件路径
]

def compute_contact_labels(pos, velocity_threshold=0.005, height_threshold=0.01):
    # pos: [T, 22, 3] (torch.Tensor or np.ndarray)
    if isinstance(pos, np.ndarray):
        pos = torch.from_numpy(pos)
    left_y = pos[:, 7, 1]  # 左脚踝Y
    right_y = pos[:, 8, 1] # 右脚踝Y

    left_vel = torch.zeros_like(left_y)
    right_vel = torch.zeros_like(right_y)
    left_vel[1:] = left_y[1:] - left_y[:-1]
    right_vel[1:] = right_y[1:] - right_y[:-1]

    min_left_height = torch.min(left_y)
    min_right_height = torch.min(right_y)

    left_contact = ((left_vel.abs() < velocity_threshold) & (left_y < min_left_height + height_threshold)).long()
    right_contact = ((right_vel.abs() < velocity_threshold) & (right_y < min_right_height + height_threshold)).long()

    return torch.stack([left_contact, right_contact], dim=-1)  # [T, 2]

if __name__ == "__main__":
    for input_path in tqdm(target_files, desc="Processing .pt files"):
        try:
            data_total = torch.load(input_path)
        except Exception as e:
            print(f"Error loading {input_path}: {e}")
            continue

        for idx, item in enumerate(data_total):
            if not isinstance(item, dict):
                print(f"Warning: Skipping non-dict item. Type: {type(item)}, Content: '{item}' in file {input_path} (index {idx})")
                continue
            if 'position_global_full_gt_world' not in item:
                print(f"Warning: Skipping dict with missing key. Available keys: {list(item.keys())} in file {input_path} (index {idx})")
                continue
            pos = item['position_global_full_gt_world']
            gt_contact = compute_contact_labels(pos)
            item['gt_contact'] = gt_contact
        # 新文件名加后缀
        output_path = input_path.replace('.pt', '_with_contact.pt')
        try:
            torch.save(data_total, output_path)
        except Exception as e:
            print(f"Error saving {output_path}: {e}")
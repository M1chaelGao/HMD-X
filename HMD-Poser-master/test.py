# 文件: test.py (最终版)

import torch
from tqdm import tqdm
import argparse
from utils.utils_config import load_config
from dataset.dataloader import load_data, TestDataset
from torch.utils.data import DataLoader
from model.hmd_imu_model import HMDIMUModel
from utils import utils_transform
import math
from utils.metrics import get_metric_function
import prettytable as pt
import numpy as np  # 确保导入numpy

# --- 指标定义与常量 (保持不变) ---
RADIANS_TO_DEGREES = 360.0 / (2 * math.pi)
METERS_TO_CENTIMETERS = 100.0

pred_metrics = [
    "mpjre",
    "mpjpe",
    "mpjve",
    "handpe",
    "upperpe",
    "lowerpe",
    "rootpe",
    "pred_jitter",
    "skating_error",
    "footjointpe",
    "footjointheelpe",
    "footjointtoepe"
]
gt_metrics = [
    "gt_jitter",
]
all_metrics = pred_metrics + gt_metrics

metrics_coeffs = {
    "mpjre": RADIANS_TO_DEGREES,
    "mpjpe": METERS_TO_CENTIMETERS,
    "mpjve": METERS_TO_CENTIMETERS,
    "handpe": METERS_TO_CENTIMETERS,
    "upperpe": METERS_TO_CENTIMETERS,
    "lowerpe": METERS_TO_CENTIMETERS,
    "rootpe": METERS_TO_CENTIMETERS,
    "skating_error": METERS_TO_CENTIMETERS,
    "footjointpe": METERS_TO_CENTIMETERS,
    "footjointheelpe": METERS_TO_CENTIMETERS,
    "footjointtoepe": METERS_TO_CENTIMETERS,
    "pred_jitter": 1.0,
    "gt_jitter": 1.0,
}


# --- 定义结束 ---


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./options/test_config.yaml",
                        help="Path, where config file is stored")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    configs = load_config(args.config)
    # 推荐将设备选择也放入配置中，但这里为了方便直接指定
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载测试数据
    test_datas = load_data(configs.dataset_path, "test",
                           input_motion_length=configs.input_motion_length)

    # 创建并加载模型
    # 【关键】确保这里的 configs 与您训练18D基线时使用的模型参数一致
    model = HMDIMUModel(configs, device)
    model.load(configs.resume_model)
    print(f"Successfully resumed checkpoint from {configs.resume_model}")
    model.eval()

    # 初始化结果表格
    tb = pt.PrettyTable()
    tb.field_names = ['Input_type'] + pred_metrics + gt_metrics

    # 循环处理不同的测试条件 (例如 HMD, HMD+2IMUs ...)
    for input_type in configs.compatible_inputs:
        print(f"Testing on {input_type}...")
        test_dataset = TestDataset(test_datas, [input_type],
                                   configs.input_motion_length, 1)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=1,  # 测试时batch_size通常为1，以处理不同长度的序列
            shuffle=False,
            num_workers=1,
            drop_last=False
        )

        log = {metric: 0.0 for metric in all_metrics}

        with torch.no_grad():
            for _, (input_feat, gt_local_pose, gt_global_pose, gt_positions, head_global_trans) in tqdm(
                    enumerate(test_dataloader)):

                # --- 数据移动与预处理 ---
                input_feat = input_feat.to(device).float()
                gt_local_pose = gt_local_pose.to(device).float()
                gt_positions = gt_positions.to(device).float()
                # flatten操作以匹配模型输入
                if len(input_feat.shape) == 4:
                    input_feat = torch.flatten(input_feat, start_dim=0, end_dim=1)
                    gt_local_pose = torch.flatten(gt_local_pose, start_dim=0, end_dim=1)
                    gt_positions = torch.flatten(gt_positions, start_dim=0, end_dim=1)

                # --- 【核心修改】模型调用与返回值处理 ---
                batch_size, time_seq = input_feat.shape[:2]

                # 1. 正确解包模型返回的4个值
                #    do_fk=True确保我们能得到 pred_joint_position
                pred_local_pose, pred_shapes, rotation_global_r6d, pred_joint_position = model(input_feat, do_fk=True)

                # --- 数据后处理，用于评估 ---
                # 2. 将6D旋转转换为轴角(axis-angle)表示，用于MPJRE计算
                pred_local_pose_aa = utils_transform.sixd2aa(pred_local_pose.reshape(-1, 6).detach()).reshape(
                    batch_size * time_seq, -1)
                gt_local_pose_aa = utils_transform.sixd2aa(gt_local_pose.reshape(-1, 6).detach()).reshape(
                    batch_size * time_seq, -1)

                # 3. 准备用于MPJPE计算的3D关节位置
                gt_positions = gt_positions.reshape(batch_size * time_seq, 22, 3).detach()
                pred_joint_position = pred_joint_position.reshape(batch_size * time_seq, 22, 3).detach()
                # 使用真实头部位置进行对齐，这是该领域的标准评估方法
                pred_joint_position = pred_joint_position - pred_joint_position[:, 15:16] + gt_positions[:, 15:16]

                # 4. 统一变量命名，供 get_metric_function 调用
                predicted_position = pred_joint_position
                gt_position = gt_positions
                predicted_angle = pred_local_pose_aa[..., 3:66]  # 身体关节旋转
                predicted_root_angle = pred_local_pose_aa[..., :3]  # 根关节旋转
                gt_angle = gt_local_pose_aa[..., 3:66]
                gt_root_angle = gt_angle_aa[..., :3]

                # 定义评估指标所需要的身体部位索引
                upper_index = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
                lower_index = [0, 1, 2, 4, 5, 7, 8, 10, 11]
                eval_log = {}

                # 5. 循环计算所有指标
                for metric in all_metrics:
                    eval_log[metric] = (
                        get_metric_function(metric)(
                            predicted_position,
                            predicted_angle,
                            predicted_root_angle,
                            gt_position,
                            gt_angle,
                            gt_root_angle,
                            upper_index,
                            lower_index,
                            fps=60,
                        )
                        .cpu()
                        .numpy()
                    )

                for key in eval_log:
                    log[key] += eval_log[key]

        # 循环结束后，计算平均值并添加到表格中
        row_data = [input_type]
        for metric in all_metrics:
            avg_metric_value = log[metric] / len(test_dataloader) * metrics_coeffs.get(metric, 1.0)
            row_data.append('%.2f' % avg_metric_value)
        tb.add_row(row_data)

    # 打印最终的结果表格
    print("\n" + "=" * 20 + " Test Results " + "=" * 20)
    print(tb)


if __name__ == "__main__":
    main()
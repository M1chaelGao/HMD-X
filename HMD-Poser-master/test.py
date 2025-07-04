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



#####################
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

RADIANS_TO_DEGREES = 360.0 / (2 * math.pi)  # 57.2958 grads
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

#####################

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path, where config file is stored")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    args.config = "./options/test_config.yaml"
    configs = load_config(args.config)
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    test_datas = load_data(configs.dataset_path, "test", 
        input_motion_length=configs.input_motion_length)

    model = HMDIMUModel(configs, device)
    model.load(configs.resume_model)
    print(f"successfully resume checkpoint in {configs.resume_model}")
    model.eval()

    # Print the value for all the metrics
    tb = pt.PrettyTable()
    tb.field_names = ['Input_type'] + pred_metrics + gt_metrics

    for input_type in configs.compatible_inputs:
        print(f"Testing on {input_type}")
        test_dataset = TestDataset(test_datas, [input_type], 
            configs.input_motion_length, 1)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            drop_last=False
        )

        log = {}
        for metric in all_metrics:
            log[metric] = 0

        # 初始化存储张量的列表
        all_gt_angles = []
        all_predicted_angles = []
        all_gt_trans = []
        all_predicted_trans = []

        with torch.no_grad():
            for _, (input_feat, gt_local_pose, gt_global_pose, gt_positions, head_global_trans) in tqdm(enumerate(test_dataloader)):
                input_feat, gt_local_pose, gt_global_pose, gt_positions, head_global_trans = input_feat.to(device).float(), \
                    gt_local_pose.to(device).float(), gt_global_pose.to(device).float(), \
                        gt_positions.to(device).float(), head_global_trans.to(device).float()
                
                if len(input_feat.shape) == 4:
                    input_feat = torch.flatten(input_feat, start_dim=0, end_dim=1)
                    gt_local_pose = torch.flatten(gt_local_pose, start_dim=0, end_dim=1)
                    gt_global_pose = torch.flatten(gt_global_pose, start_dim=0, end_dim=1)
                    gt_positions = torch.flatten(gt_positions, start_dim=0, end_dim=1)
                    head_global_trans = torch.flatten(head_global_trans, start_dim=0, end_dim=1)

                batch_size, time_seq = input_feat.shape[0], input_feat.shape[1]
                pred_local_pose, _, rotation_global_r6d, pred_joint_position = model(input_feat)
                
                pred_local_pose_aa = utils_transform.sixd2aa(pred_local_pose.reshape(-1, 6).detach()).reshape(batch_size*time_seq, 22*3)
                gt_local_pose_aa = utils_transform.sixd2aa(gt_local_pose.reshape(-1, 6).detach()).reshape(batch_size*time_seq, 22*3)
                gt_positions = gt_positions.reshape(batch_size*time_seq, 22, 3).detach()
                pred_joint_position = pred_joint_position.reshape(batch_size*time_seq, 22, 3).detach()
                pred_joint_position = pred_joint_position - pred_joint_position[:, 15:16] + gt_positions[:, 15:16]

                # skate = skating_error(predicted_position, gt_position) * 100
                # error_dict['Skate'].append(skate_error_)
                predicted_angle = pred_local_pose_aa[...,3:66]
                predicted_root_angle = pred_local_pose_aa[...,:3]
                predicted_position = pred_joint_position

                gt_angle = gt_local_pose_aa[...,3:66]
                gt_root_angle = gt_local_pose_aa[...,:3]
                gt_position = gt_positions

                upper_index = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
                lower_index = [0, 1, 2, 4, 5, 7, 8, 10, 11]
                eval_log = {}

                # 将 root_orient 添加到 gt_angle 和 predicted_angle
                gt_angle_0 = torch.cat((gt_root_angle, gt_angle), dim=1)  # (seq, 66)
                predicted_angle_0 =  pred_local_pose_aa[...,:66]  # (seq, 66)

                # 拼接 3x2 的零向量，使其变为 (seq, 72)
                zeros_to_add = torch.zeros(gt_angle.size(0), 6, device=gt_angle.device)  # (seq, 6)
                gt_angle_0 = torch.cat((gt_angle_0, zeros_to_add), dim=1)  # (seq, 72)
                predicted_angle_0 = torch.cat((predicted_angle_0, zeros_to_add), dim=1)  # (seq, 72)
                gt_trans = torch.zeros(gt_angle.size(0), 1, device=gt_angle.device)

                # 将张量添加到对应的列表中
                all_gt_angles.append(gt_angle_0)
                all_predicted_angles.append(predicted_angle_0)
                all_gt_trans.append(gt_trans)
                all_predicted_trans.append(gt_trans)

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
        # # 将这些张量存入字典
        # results = {
        #     'gt_angle': all_gt_angles,
        #     'predicted_angle': all_predicted_angles,
        #     'gt_trans': all_gt_trans,
        #     'predicted_trans': all_predicted_trans
        # }
        # # 动态生成新的文件名
        # new_filename = f"{input_type}_vis.pt"
        # # 保存为 .pt 文件
        # torch.save(results, new_filename)
        tb.add_row([input_type] + 
            [ '%.2f' % (log[metric] / len(test_dataloader) * metrics_coeffs[metric]) for metric in pred_metrics] +
            [ '%.2f' % (log[metric] / len(test_dataloader) * metrics_coeffs[metric]) for metric in gt_metrics]
        )
    print(tb)

if __name__ == "__main__":
    main()
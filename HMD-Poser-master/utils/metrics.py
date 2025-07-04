# Copyright (c) Meta Platforms, Inc. All Rights Reserved
# Metric functions with same inputs

import numpy as np
import torch


def pred_jitter(
    predicted_position,
    predicted_angle,
    predicted_root_angle,
    gt_position,
    gt_angle,
    gt_root_angle,
    upper_index,
    lower_index,
    fps,
):
    pred_jitter = (
        (
            (
                predicted_position[3:]
                - 3 * predicted_position[2:-1]
                + 3 * predicted_position[1:-2]
                - predicted_position[:-3]
            )
            * (fps**3)
        )
        .norm(dim=2)
        .mean()
    )
    return pred_jitter


def gt_jitter(
    predicted_position,
    predicted_angle,
    predicted_root_angle,
    gt_position,
    gt_angle,
    gt_root_angle,
    upper_index,
    lower_index,
    fps,
):
    gt_jitter = (
        (
            (
                gt_position[3:]
                - 3 * gt_position[2:-1]
                + 3 * gt_position[1:-2]
                - gt_position[:-3]
            )
            * (fps**3)
        )
        .norm(dim=2)
        .mean()
    )
    return gt_jitter


def mpjre(
    predicted_position,
    predicted_angle,
    predicted_root_angle,
    gt_position,
    gt_angle,
    gt_root_angle,
    upper_index,
    lower_index,
    fps,
):
    diff = gt_angle - predicted_angle
    diff[diff > np.pi] = diff[diff > np.pi] - 2 * np.pi
    diff[diff < -np.pi] = diff[diff < -np.pi] + 2 * np.pi
    rot_error = torch.mean(torch.absolute(diff))
    return rot_error


def mpjpe(
    predicted_position,
    predicted_angle,
    predicted_root_angle,
    gt_position,
    gt_angle,
    gt_root_angle,
    upper_index,
    lower_index,
    fps,
):
    pos_error = torch.mean(
        torch.sqrt(torch.sum(torch.square(gt_position - predicted_position), axis=-1))
    )
    return pos_error


def handpe(
    predicted_position,
    predicted_angle,
    predicted_root_angle,
    gt_position,
    gt_angle,
    gt_root_angle,
    upper_index,
    lower_index,
    fps,
):
    pos_error_hands = torch.mean(
        torch.sqrt(torch.sum(torch.square(gt_position - predicted_position), axis=-1))[
            ..., [20, 21]
        ]
    )
    return pos_error_hands


def upperpe(
    predicted_position,
    predicted_angle,
    predicted_root_angle,
    gt_position,
    gt_angle,
    gt_root_angle,
    upper_index,
    lower_index,
    fps,
):
    upper_body_error = torch.mean(
        torch.sqrt(torch.sum(torch.square(gt_position - predicted_position), axis=-1))[
            ..., upper_index
        ]
    )
    return upper_body_error


def lowerpe(
    predicted_position,
    predicted_angle,
    predicted_root_angle,
    gt_position,
    gt_angle,
    gt_root_angle,
    upper_index,
    lower_index,
    fps,
):
    lower_body_error = torch.mean(
        torch.sqrt(torch.sum(torch.square(gt_position - predicted_position), axis=-1))[
            ..., lower_index
        ]
    )
    return lower_body_error


def rootpe(
    predicted_position,
    predicted_angle,
    predicted_root_angle,
    gt_position,
    gt_angle,
    gt_root_angle,
    upper_index,
    lower_index,
    fps,
):
    pos_error_root = torch.mean(
        torch.sqrt(torch.sum(torch.square(gt_position - predicted_position), axis=-1))[
            ..., [0]
        ]
    )
    return pos_error_root


def mpjve(
    predicted_position,
    predicted_angle,
    predicted_root_angle,
    gt_position,
    gt_angle,
    gt_root_angle,
    upper_index,
    lower_index,
    fps,
):
    gt_velocity = (gt_position[1:, ...] - gt_position[:-1, ...]) * fps
    predicted_velocity = (
        predicted_position[1:, ...] - predicted_position[:-1, ...]
    ) * fps
    vel_error = torch.mean(
        torch.sqrt(torch.sum(torch.square(gt_velocity - predicted_velocity), axis=-1))
    )
    return vel_error

def footjointpe(
    predicted_position,
    predicted_angle,
    predicted_root_angle,
    gt_position,
    gt_angle,
    gt_root_angle,
    upper_index,
    lower_index,
    fps,
):
    foot_joint_error = torch.mean(
        torch.sqrt(torch.sum(torch.square(gt_position - predicted_position), axis=-1))[
            ..., [7,8,10,11]
        ]
    )
    return foot_joint_error

def footjointheelpe(
    predicted_position,
    predicted_angle,
    predicted_root_angle,
    gt_position,
    gt_angle,
    gt_root_angle,
    upper_index,
    lower_index,
    fps,
):
    foot_joint_heel_error = torch.mean(
        torch.sqrt(torch.sum(torch.square(gt_position - predicted_position), axis=-1))[
            ..., [7,8]
        ]
    )
    return foot_joint_heel_error

def footjointtoepe(
    predicted_position,
    predicted_angle,
    predicted_root_angle,
    gt_position,
    gt_angle,
    gt_root_angle,
    upper_index,
    lower_index,
    fps,
):
    foot_joint_toe_error = torch.mean(
        torch.sqrt(torch.sum(torch.square(gt_position - predicted_position), axis=-1))[
            ..., [10,11]
        ]
    )
    return foot_joint_toe_error


def skating_error(
    predicted_position,
    predicted_angle,
    predicted_root_angle,
    gt_position,
    gt_angle,
    gt_root_angle,
    upper_index,
    lower_index,
    fps,):
    '''
    pred_mesh: (batch, v_num, 3)
    '''
    seq_len = predicted_position.shape[0]
    batch = predicted_position.shape[0] // seq_len

    # batch, seq_len = pred.shape[0], pred.shape[1]
    predicted_position = predicted_position.reshape(batch, seq_len, -1)
    gt_position = gt_position.reshape(batch, seq_len, -1)
    predicted_position = predicted_position[:, :, :22 * 3].reshape(batch, seq_len, 22, 3)
    gt_position = gt_position[:, :, :22 * 3].reshape(batch, seq_len, 22, 3)

    # 'L_Ankle', # 7, 'R_Ankle', # 8 , 'L_Foot', # 10, 'R_Foot', # 11
    l_ankle_idx, r_ankle_idx, l_foot_idx, r_foot_idx = 7, 8, 10, 11
    relevant_joints = [l_ankle_idx, l_foot_idx, r_ankle_idx, r_foot_idx]
    gt_joint_xyz = gt_position[:, :, relevant_joints, :] # [BatchSize, 4, 3, Frames]
    gt_joint_vel = torch.linalg.norm(gt_joint_xyz[:, 1:, :, :] - gt_joint_xyz[:, :-1, :, :], dim=-1) # [BatchSize, 4, Frames]
    fc_mask = torch.unsqueeze((gt_joint_vel <= 0.01), dim=-1).repeat(1, 1, 1, 3)
    pred_joint_xyz = predicted_position[:, :, relevant_joints, :] # [BatchSize, 4, 3, Frames]
    pred_vel = pred_joint_xyz[:, 1:, :, :] - pred_joint_xyz[:, :-1, :, :]
    pred_vel[~fc_mask] = 0
    foot_concat_loss = torch.abs(torch.zeros(pred_vel.shape, device=pred_vel.device) - pred_vel).mean()
    return foot_concat_loss


metric_funcs_dict = {
    "mpjre": mpjre,
    "mpjpe": mpjpe,
    "mpjve": mpjve,
    "handpe": handpe,
    "upperpe": upperpe,
    "lowerpe": lowerpe,
    "rootpe": rootpe,
    "pred_jitter": pred_jitter,
    "gt_jitter": gt_jitter,
    "skating_error": skating_error,
    "footjointpe": footjointpe,
    "footjointheelpe": footjointheelpe,
    "footjointtoepe": footjointtoepe
}

def get_metric_function(metric):
    return metric_funcs_dict[metric]

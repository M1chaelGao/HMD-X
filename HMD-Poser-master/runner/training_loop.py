import torch
import os
import time
from tqdm import tqdm
import numpy as np
from utils import utils_transform
import math

RADIANS_TO_DEGREES = 360.0 / (2 * math.pi)
METERS_TO_CENTIMETERS = 100.0

def train_loop(configs, device, model, loss_func, optimizer, \
               lr_scheduler, train_loader, test_loader, LOG, train_summary_writer, out_dir, total_loss_main=None):
    global_step = 0
    best_train_loss, best_test_loss, best_position, best_local_pose = float("inf"), float("inf"), float("inf"), float("inf")
    for epoch in range(configs.train_config.epochs):
        model.train()
        current_lr = optimizer.param_groups[0]['lr']
        LOG.info(f'Training epoch: {epoch + 1}/{configs.train_config.epochs}, LR: {current_lr:.6f}')

        one_epoch_root_loss, one_epoch_lpose_loss, one_epoch_gpose_loss, one_epoch_joint_loss, \
            one_epoch_acc_loss, one_epoch_shape_loss, one_epoch_total_loss = [], [], [], [], [], [], []
        for batch_idx, (input_feat, gt_local_pose, gt_global_pose, gt_positions, gt_betas) in enumerate(train_loader):
            global_step += 1
            optimizer.zero_grad()
            batch_start = time.time()
            input_feat, gt_local_pose, gt_global_pose, gt_positions, gt_betas = input_feat.to(device).float(), \
                gt_local_pose.to(device).float(), gt_global_pose.to(device).float(), \
                    gt_positions.to(device).float(), gt_betas.to(device).float()
            if len(input_feat.shape) == 4:
                input_feat = torch.flatten(input_feat, start_dim=0, end_dim=1) # （256,3,40,135）-》（768,40,135） 3种模态的输入
                gt_local_pose = torch.flatten(gt_local_pose, start_dim=0, end_dim=1)
                gt_global_pose = torch.flatten(gt_global_pose, start_dim=0, end_dim=1)
                gt_positions = torch.flatten(gt_positions, start_dim=0, end_dim=1)
                gt_betas = torch.flatten(gt_betas, start_dim=0, end_dim=1)

            refined_pose, pose_draft, pred_shapes, rotation_global_r6d, pred_joint_position = model(input_feat)
            pred_local_pose = refined_pose
            pred_betas = pred_shapes
            pred_joint_position_head_centered = pred_joint_position - pred_joint_position[:, :, 15:16] + gt_positions[:, :, 15:16]  # 把输入的头部位置作为最终的
            gt_positions_head_centered = gt_positions

            root_orientation_loss, local_pose_loss, global_pose_loss, joint_position_loss, accel_loss, shape_loss, total_loss_main = \
                loss_func(pred_local_pose[:, :, :6], pred_local_pose[:, :, 6:], rotation_global_r6d, pred_joint_position_head_centered, pred_betas, \
                          gt_local_pose[:, :, :6], gt_local_pose[:, :, 6:], gt_global_pose, gt_positions_head_centered, gt_betas)
            # 2. 计算辅助损失 (针对原始草稿的L1损失)
            #    注意：这里的gt_local_pose是和pose_draft维度完全匹配的真值
            loss_draft = torch.nn.functional.l1_loss(pose_draft, gt_local_pose)
            # 3. 合并成最终的总损失
            alpha = 0.3  # 这是一个超参数，您可以根据实验效果调整
            total_loss = total_loss_main + alpha * loss_draft
            # --- 修改结束 ---
            # loss是l1损失
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad),
                max_norm=1.0
            )
            optimizer.step()
            lr_scheduler.step()
            lr = optimizer.param_groups[0]['lr']

            one_epoch_root_loss.append(root_orientation_loss.item())
            one_epoch_lpose_loss.append(local_pose_loss.item())
            one_epoch_gpose_loss.append(global_pose_loss.item())
            one_epoch_joint_loss.append(joint_position_loss.item())
            one_epoch_acc_loss.append(accel_loss.item())
            one_epoch_shape_loss.append(shape_loss.item())
            one_epoch_total_loss.append(total_loss.item())
            batch_time = time.time() - batch_start
            
            if batch_idx % configs.train_config.log_interval == 0:
                batch_info = {
                    'type': 'train',
                    'epoch': epoch + 1, 'batch': batch_idx, 'n_batches': len(train_loader),
                    'time': round(batch_time, 5),
                    'lr': round(lr, 8),
                    'loss_total': round(float(total_loss.item()), 3),
                    'root_orientation_loss': round(float(root_orientation_loss.item()), 3),
                    'local_pose_loss': round(float(local_pose_loss.item()), 3),
                    'global_pose_loss': round(float(global_pose_loss.item()), 3),
                    'joint_3d_loss': round(float(joint_position_loss.item()), 3),
                    'smooth_loss': round(float(accel_loss.item()), 3),
                    'shape_loss': round(float(shape_loss.item()), 3),
                    'loss_draft': round(float(loss_draft.item()), 3)
                }
                LOG.info(batch_info)
        one_epoch_root_loss = torch.tensor(one_epoch_root_loss).mean().item()
        one_epoch_lpose_loss = torch.tensor(one_epoch_lpose_loss).mean().item()
        one_epoch_gpose_loss = torch.tensor(one_epoch_gpose_loss).mean().item()
        one_epoch_joint_loss = torch.tensor(one_epoch_joint_loss).mean().item()
        one_epoch_acc_loss = torch.tensor(one_epoch_acc_loss).mean().item()
        one_epoch_shape_loss = torch.tensor(one_epoch_shape_loss).mean().item()
        one_epoch_total_loss = torch.tensor(one_epoch_total_loss).mean().item()

        train_summary_writer.add_scalar(
            'train_epoch_loss/loss_total', one_epoch_total_loss, epoch)
        train_summary_writer.add_scalar(
            'train_epoch_loss/root_orientation_loss', one_epoch_root_loss, epoch)
        train_summary_writer.add_scalar(
            'train_epoch_loss/local_pose_loss', one_epoch_lpose_loss, epoch)
        train_summary_writer.add_scalar(
            'train_epoch_loss/global_pose_loss', one_epoch_gpose_loss, epoch)
        train_summary_writer.add_scalar(
            'train_epoch_loss/joint_3d_loss', one_epoch_joint_loss, epoch)
        train_summary_writer.add_scalar(
            'train_epoch_loss/smooth_loss', one_epoch_acc_loss, epoch)
        train_summary_writer.add_scalar(
            'train_epoch_loss/shape_loss', one_epoch_shape_loss, epoch)

        epoch_info = {
            'type': 'train',
            'epoch': epoch + 1,
            'loss_total': round(float(one_epoch_total_loss), 3),
            'root_orientation_loss': round(float(one_epoch_root_loss), 3),
            'local_pose_loss': round(float(one_epoch_lpose_loss), 3),
            'global_pose_loss': round(float(one_epoch_gpose_loss), 3),
            'joint_3d_loss': round(float(one_epoch_joint_loss), 3),
            'smooth_loss': round(float(one_epoch_acc_loss), 3),
            'shape_loss': round(float(one_epoch_shape_loss), 3)
        }
        LOG.info(epoch_info)

        if one_epoch_total_loss < best_train_loss:
            LOG.info("Saving model with best train loss in epoch {}".format(epoch+1))
            filename = os.path.join(out_dir, "epoch_with_best_trainloss.pt")
            if os.path.exists(filename):
                os.remove(filename)
            model.save(epoch, filename)
            best_train_loss = one_epoch_total_loss

        if epoch % configs.train_config.val_interval == 0:
            model.eval()
            torch.cuda.empty_cache()  # 增加一个好习惯：在验证前清空缓存
            val_metrics = {
                'total_loss': [], 'root_loss': [], 'lpose_loss': [], 'gpose_loss': [],
                'joint_loss': [], 'acc_loss': [], 'shape_loss': [],
                'pos_error': [], 'rot_error': []
            }
            with torch.no_grad():
                for _, (input_feat, gt_local_pose, gt_global_pose, gt_positions, gt_betas) in tqdm(
                        enumerate(test_loader)):
                    input_feat, gt_local_pose, gt_global_pose, gt_positions, gt_betas = input_feat.to(device).float(), \
                        gt_local_pose.to(device).float(), gt_global_pose.to(device).float(), \
                        gt_positions.to(device).float(), gt_betas.to(device).float()
                    if len(input_feat.shape) == 4:
                        input_feat = torch.flatten(input_feat, start_dim=0, end_dim=1)
                        gt_local_pose = torch.flatten(gt_local_pose, start_dim=0, end_dim=1)
                        gt_global_pose = torch.flatten(gt_global_pose, start_dim=0, end_dim=1)
                        gt_positions = torch.flatten(gt_positions, start_dim=0, end_dim=1)
                        gt_betas = torch.flatten(gt_betas, start_dim=0, end_dim=1)

                    # ------------------- 核心修改：滑动窗口验证 -------------------

                    full_seq_len = input_feat.shape[1]
                    chunk_size = configs.input_motion_length

                    if full_seq_len <= chunk_size:
                        pred_local_pose, _, pred_betas = model(input_feat, do_fk=False)
                    else:
                        stride = chunk_size // 2
                        pred_pose_chunks = []
                        pred_betas_chunks = []

                        num_chunks = (full_seq_len - chunk_size + stride) // stride
                        for i in range(num_chunks):
                            start_idx = i * stride
                            end_idx = start_idx + chunk_size
                            if end_idx > full_seq_len:
                                end_idx = full_seq_len
                                start_idx = end_idx - chunk_size

                            input_chunk = input_feat[:, start_idx:end_idx]
                            pred_pose_chunk, _, pred_betas_chunk = model(input_chunk, do_fk=False)
                            pred_pose_chunks.append(pred_pose_chunk)
                            pred_betas_chunks.append(pred_betas_chunk)

                            if end_idx == full_seq_len:
                                break

                        # --- 核心修改：为pose和betas创建独立的、形状正确的计数器 ---
                        pred_local_pose = torch.zeros(gt_local_pose.shape[0], full_seq_len, gt_local_pose.shape[2],
                                                      device=device)
                        pred_betas = torch.zeros(gt_betas.shape[0], full_seq_len, gt_betas.shape[2], device=device)

                        pose_count_matrix = torch.zeros_like(pred_local_pose)
                        betas_count_matrix = torch.zeros_like(pred_betas)
                        # --- 修改结束 ---

                        for i, chunk_idx in enumerate(range(num_chunks)):
                            start_idx = chunk_idx * stride
                            end_idx = start_idx + chunk_size
                            if end_idx > full_seq_len:
                                end_idx = full_seq_len
                                start_idx = end_idx - chunk_size

                            pose_count_matrix[:, start_idx:end_idx] += 1
                            pred_local_pose[:, start_idx:end_idx] += pred_pose_chunks[i]

                            betas_count_matrix[:, start_idx:end_idx] += 1
                            pred_betas[:, start_idx:end_idx] += pred_betas_chunks[i]

                            if end_idx == full_seq_len:
                                break

                        # 使用各自的计数器进行平均
                        epsilon = 1e-8
                        pred_local_pose /= (pose_count_matrix + epsilon)
                        pred_betas /= (betas_count_matrix + epsilon)
                    # ------------------- 滑动窗口结束，后续计算不变 -------------------

                    # 在得到完整的、与GT对齐的预测序列后，再统一进行FK计算和误差评估
                    with torch.no_grad():
                        batch_size, final_pred_len = pred_local_pose.shape[:2]
                        rotation_local_matrot = utils_transform.sixd2matrot(pred_local_pose.reshape(-1, 6)).reshape(
                            batch_size * final_pred_len, 22, 3, 3)
                        rotation_global_matrot = model.forward_kinematics_R(rotation_local_matrot,
                                                                            model.bm.kintree_table[0][:22].long()).view(
                            batch_size, final_pred_len, 22, 3, 3)
                        rotation_global_r6d = utils_transform.matrot2sixd(
                            rotation_global_matrot.reshape(-1, 3, 3)).reshape(batch_size, final_pred_len, 22 * 6)
                        pred_joint_position = model.fk_module(pred_local_pose[:, :, :6].reshape(-1, 6),
                                                              pred_local_pose[:, :, 6:].reshape(-1, 21 * 6),
                                                              pred_betas.reshape(-1, 16))
                        pred_joint_position = pred_joint_position.reshape(batch_size, final_pred_len, 22, 3)

                    pred_joint_position_head_centered = pred_joint_position - pred_joint_position[:, :,
                                                                              15:16] + gt_positions[:, :, 15:16]
                    gt_positions_head_centered = gt_positions

                    root_orientation_loss, local_pose_loss, global_pose_loss, joint_position_loss, accel_loss, shape_loss, total_loss_main = \
                        loss_func(pred_local_pose[:, :, :6], pred_local_pose[:, :, 6:], rotation_global_r6d,
                                  pred_joint_position_head_centered, pred_betas, \
                                  gt_local_pose[:, :, :6], gt_local_pose[:, :, 6:], gt_global_pose,
                                  gt_positions_head_centered, gt_betas)

                    pos_error = torch.mean(torch.sqrt(
                        torch.sum(torch.square(gt_positions_head_centered - pred_joint_position_head_centered),
                                  axis=-1)))
                    pred_local_pose_aa = utils_transform.sixd2aa(pred_local_pose.reshape(-1, 6).detach()).reshape(-1,
                                                                                                                  22 * 3)
                    gt_local_pose_aa = utils_transform.sixd2aa(gt_local_pose.reshape(-1, 6).detach()).reshape(-1,
                                                                                                              22 * 3)
                    diff = gt_local_pose_aa - pred_local_pose_aa
                    diff[diff > np.pi] -= 2 * np.pi
                    diff[diff < -np.pi] += 2 * np.pi
                    rot_error = torch.mean(torch.absolute(diff))

                    val_metrics['total_loss'].append(total_loss_main.item())
                    val_metrics['root_loss'].append(root_orientation_loss.item())
                    val_metrics['lpose_loss'].append(local_pose_loss.item())
                    val_metrics['gpose_loss'].append(global_pose_loss.item())
                    val_metrics['joint_loss'].append(joint_position_loss.item())
                    val_metrics['acc_loss'].append(accel_loss.item())
                    val_metrics['shape_loss'].append(shape_loss.item())
                    val_metrics['pos_error'].append(pos_error.item() * METERS_TO_CENTIMETERS)
                    val_metrics['rot_error'].append(rot_error.item() * RADIANS_TO_DEGREES)
            
            avg_total_loss = np.mean(val_metrics['total_loss'])
            avg_root_loss = np.mean(val_metrics['root_loss'])
            avg_lpose_loss = np.mean(val_metrics['lpose_loss'])
            avg_gpose_loss = np.mean(val_metrics['gpose_loss'])
            avg_joint_loss = np.mean(val_metrics['joint_loss'])
            avg_acc_loss = np.mean(val_metrics['acc_loss'])
            avg_shape_loss = np.mean(val_metrics['shape_loss'])
            avg_pos_error = np.mean(val_metrics['pos_error'])
            avg_rot_error = np.mean(val_metrics['rot_error'])

            train_summary_writer.add_scalar(
                'val_epoch_loss/loss_total', avg_total_loss, epoch)
            train_summary_writer.add_scalar(
                'val_epoch_loss/root_orientation_loss', avg_root_loss, epoch)
            train_summary_writer.add_scalar(
                'val_epoch_loss/local_pose_loss', avg_lpose_loss, epoch)
            train_summary_writer.add_scalar(
                'val_epoch_loss/global_pose_loss', avg_gpose_loss, epoch)
            train_summary_writer.add_scalar(
                'val_epoch_loss/joint_3d_loss', avg_joint_loss, epoch)
            train_summary_writer.add_scalar(
                'val_epoch_loss/smooth_loss', avg_acc_loss, epoch)
            train_summary_writer.add_scalar(
                'val_epoch_loss/shape_loss', avg_shape_loss, epoch)
            train_summary_writer.add_scalar(
                'val_epoch_loss/position_error', avg_pos_error, epoch)
            train_summary_writer.add_scalar(
                'val_epoch_loss/local_pose_error', avg_rot_error, epoch)

            epoch_info = {
                'type': 'test',
                'epoch': epoch + 1,
                'loss_total': round(float(avg_total_loss), 3),
                'root_orientation_loss': round(float(avg_root_loss), 3),
                'local_pose_loss': round(float(avg_lpose_loss), 3),
                'global_pose_loss': round(float(avg_gpose_loss), 3),
                'joint_3d_loss': round(float(avg_joint_loss), 3),
                'smooth_loss': round(float(avg_acc_loss), 3),
                'shape_loss': round(float(avg_shape_loss), 3),
                'MPJPE': round(float(avg_pos_error), 3),
                'MPJRE_with_Root': round(float(avg_rot_error), 3)
            }
            LOG.info(epoch_info)
            model.train()

            if avg_total_loss < best_test_loss:
                LOG.info("Saving model with lowest test loss in epoch {}".format(epoch+1))
                filename = os.path.join(out_dir, "epoch_with_best_testloss.pt")
                if os.path.exists(filename):
                    os.remove(filename)
                model.save(epoch, filename)
                best_test_loss = avg_total_loss
            
            if avg_pos_error < best_position:
                best_position = avg_pos_error
                LOG.info("Lowest MPJPE {} in epoch {}".format(best_position, epoch+1))
            
            if avg_rot_error < best_local_pose:
                best_local_pose = avg_rot_error
                LOG.info("Lowest MPJRE_(including root) {} in epoch {}".format(best_local_pose, epoch+1))

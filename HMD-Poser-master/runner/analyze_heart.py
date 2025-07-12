import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from human_body_prior.body_model.body_model import BodyModel

# 优先使用真实的aa2sixd函数
try:
    from utils.utils_transform import aa2sixd
except ImportError:
    # 如果找不到，定义一个能工作的占位或简单实现
    def aa2sixd(aa):
        # 这是一个简化的、非torchscript兼容的实现，仅用于本脚本
        mat = torch.zeros((aa.shape[0], 3, 3), dtype=aa.dtype, device=aa.device)
        theta = torch.linalg.norm(aa, dim=1, keepdim=True)
        axis = aa / (theta + 1e-8)
        c = torch.cos(theta)
        s = torch.sin(theta)
        c1 = 1. - c
        ix, iy, iz = axis.T
        mat[:, 0, 0] = c + ix * ix * c1
        mat[:, 0, 1] = ix * iy * c1 - iz * s
        mat[:, 0, 2] = ix * iz * c1 + iy * s
        mat[:, 1, 0] = iy * ix * c1 + iz * s
        mat[:, 1, 1] = c + iy * iy * c1
        mat[:, 1, 2] = iy * iz * c1 - ix * s
        mat[:, 2, 0] = iz * ix * c1 - iy * s
        mat[:, 2, 1] = iz * iy * c1 + ix * s
        mat[:, 2, 2] = c + iz * iz * c1
        return mat[:, :, :2].reshape(aa.shape[0], 6)
# 同时，我们也需要 local2global_pose
try:
    from human_body_prior.tools.rotation_tools import aa2matrot
except ImportError:
    # 临时实现
    def aa2matrot(pose_aa):
        import torch
        from scipy.spatial.transform import Rotation as R
        pose_aa_np = pose_aa.detach().cpu().numpy()
        mats = R.from_rotvec(pose_aa_np).as_matrix()
        return torch.tensor(mats, dtype=pose_aa.dtype, device=pose_aa.device)
try:
    # 假设它也可能在utils里
    from utils.utils_transform import local2global_pose
except ImportError:
    # 如果没有，我们需要一个全局的 _forward_tree
    def _forward_tree(x_local, parent, reduction_fn):
        x_global = [x_local[:, 0]]
        for i in range(1, len(parent)):
            x_global.append(reduction_fn(x_global[parent[i]], x_local[:, i]))
        return torch.stack(x_global, dim=1)


    def local2global_pose(R_local, parent_list):
        R_local = R_local.view(R_local.shape[0], -1, 3, 3)
        return _forward_tree(R_local, parent_list, torch.bmm)

SAMPLING_RATE = 60
BREATHING_BAND = [0.2, 0.5]
HEARTBEAT_BAND = [1.0, 2.5]

def _syn_acc(v, smooth_n=4):
    """
    Synthesize accelerations from vertex positions.
    v: [N, K, 3]  (N帧, K顶点, 3坐标)
    返回: [N-1, K, 3] (差分后帧数减少1)
    """
    if isinstance(v, np.ndarray):
        v = torch.from_numpy(v).float()
    elif isinstance(v, torch.Tensor):
        v = v.clone().detach()  # 推荐写法
    else:
        raise TypeError("Input v must be torch.Tensor or np.ndarray")
    # 后续逻辑保持不变
    mid = smooth_n // 2
    acc = torch.stack([(v[i] + v[i + 2] - 2 * v[i + 1]) * 3600 for i in range(0, v.shape[0] - 2)])
    acc = torch.cat((torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])))
    if mid != 0 and v.shape[0] > smooth_n * 2:
        acc[smooth_n:-smooth_n] = torch.stack(
            [(v[i] + v[i + smooth_n * 2] - 2 * v[i + smooth_n]) * 3600 / smooth_n ** 2
                for i in range(0, v.shape[0] - smooth_n * 2)])
    return acc


def generate_input_features(bdata_poses, body_pose_world, bm):
    """
    参考prepare_data_me.py的逻辑，生成18维IMU特征
    返回: dict，包含'hmd_position_global_full_gt_list'
    """
    bdata_poses_tensor = torch.tensor(bdata_poses, dtype=torch.float32)

    # --- 【核心修正】我们不再从body_pose_world中取rot_mats ---
    # 1. 首先，将轴角(axis-angle)的poses转换为局部旋转矩阵
    rotation_local_matrot = aa2matrot(bdata_poses_tensor.reshape(-1, 3)).reshape(bdata_poses_tensor.shape[0], -1, 3, 3)

    # 2. 然后，手动进行FK，计算全局旋转矩阵
    #    我们使用bm.kintree_table作为父关节信息
    parent_list_cpu = bm.kintree_table[0].long().tolist()
    rotation_global_matrot = local2global_pose(rotation_local_matrot[:, :22], parent_list_cpu[:22])

    # --- 修正结束，后续逻辑可以保持，因为它们都依赖rotation_global_matrot ---

    # 提取头部旋转
    head_rotation_global_matrot = rotation_global_matrot[:, 15, :, :]  # [N, 3, 3]
    head_rot_mat = head_rotation_global_matrot

    # 头部旋转6D表示
    head_rot_6d = head_rot_mat[:, :, :2].contiguous().view(head_rot_mat.shape[0], 6)  # [N, 6]

    # 头部角速度
    # 为了避免GPU-CPU问题，所有计算都在tensor上进行
    head_rot_mat_prev = head_rotation_global_matrot[:-1]
    head_rot_mat_next = head_rotation_global_matrot[1:]
    # 使用克隆来确保张量是连续的，有时可以避免奇怪的错误
    head_vel_mat = torch.bmm(head_rot_mat_prev.transpose(1, 2).clone(), head_rot_mat_next.clone())
    head_vel_6d = head_vel_mat[:, :, :2].contiguous().view(head_vel_mat.shape[0], 6)
    head_vel_6d = torch.cat([torch.zeros((1, 6)), head_vel_6d], dim=0)

    # 耳朵加速度
    ear_indices = [4071, 516]
    vertices = body_pose_world.v
    ear_accs = _syn_acc(vertices[:, ear_indices])  # <-- 这里直接用Tensor，不要numpy
    ear_accs_flat = ear_accs.reshape(ear_accs.shape[0], -1)
    ear_accs_flat = torch.cat([torch.zeros((1, 6)), ear_accs_flat], dim=0)

    # 对齐所有特征的时间长度 (因为差分会减少一帧)
    num_frames = head_rot_6d.shape[0] - 1
    head_rot_6d = head_rot_6d[:num_frames]
    head_vel_6d = head_vel_6d[:num_frames]
    ear_accs_flat = ear_accs_flat[:num_frames]

    # 组合
    hmd_position_global_full_gt_list = torch.cat(
        [head_rot_6d, head_vel_6d, ear_accs_flat], dim=-1
    )

    return {"hmd_position_global_full_gt_list": hmd_position_global_full_gt_list}

def plot_spectrum(signal, title):
    N = len(signal)
    fft_vals = torch.fft.fft(signal)
    fft_freq = torch.fft.fftfreq(N, 1.0 / SAMPLING_RATE)
    fft_magnitude = torch.abs(fft_vals)
    positive_freq_indices = fft_freq > 0

    plt.plot(fft_freq[positive_freq_indices], fft_magnitude[positive_freq_indices], label="Signal Spectrum")
    plt.axvspan(BREATHING_BAND[0], BREATHING_BAND[1], alpha=0.2, color='r', label="Breathing Band")
    plt.axvspan(HEARTBEAT_BAND[0], HEARTBEAT_BAND[1], alpha=0.2, color='g', label="Heartbeat Band")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze frequency spectrum of AMASS data')
    parser.add_argument('--amass_root', type=str, required=True, help='Root directory of AMASS .npz files')
    parser.add_argument('--support_dir', type=str, required=True, help='Directory containing SMPL models')
    args = parser.parse_args()

    # 代表性序列
    files_to_analyze = {
        "Static - T-Pose (CMU)": "CMU/01/01_01_poses.npz",
        "Dynamic - Running (CMU)": "CMU/09/09_02_poses.npz",
        "Decoupled - Sitting Talking (CMU)": "CMU/16/16_38_poses.npz",
        "Dynamic - Jumping Jacks (CMU)": "CMU/13/13_21_poses.npz"
    }

    # 加载SMPLH模型（用neutral性别）
    subject_gender = "neutral"
    bm_fname = os.path.join(args.support_dir, 'smplh/{}/model.npz'.format(subject_gender))
    dmpl_fname = os.path.join(args.support_dir, 'dmpls/{}/model.npz'.format(subject_gender))
    num_betas = 16
    num_dmpls = 8
    bm = BodyModel(bm_fname=bm_fname, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname)

    plt.figure(figsize=(20, 15))

    for i, (description, rel_path) in enumerate(files_to_analyze.items()):
        file_path = os.path.join(args.amass_root, rel_path)
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} does not exist. Skipping.")
            continue
        print(f"Processing {description} from {file_path}")
        try:
            npz_data = np.load(file_path, allow_pickle=True)
            # 参考prepare_data_me.py，采样率相关
            framerate = npz_data.get('mocap_framerate', 60)
            stride = 2 if framerate == 120 else 1
            bdata_poses = npz_data['poses'][::stride, ...]
            bdata_trans = npz_data['trans'][::stride, ...]
            bdata_betas = npz_data['betas']
            # 只用neutral模型
            body_parms = {
                "root_orient": torch.Tensor(bdata_poses[:, :3]),
                "pose_body": torch.Tensor(bdata_poses[:, 3:66]),
                "trans": torch.Tensor(bdata_trans),
                'betas': torch.Tensor(bdata_betas).repeat(bdata_poses.shape[0], 1),
            }
            body_pose_world = bm(
                **{k: v for k, v in body_parms.items() if k in ["pose_body", "root_orient", "trans", 'betas']}
            )
            features = generate_input_features(bdata_poses, body_pose_world, bm)
            # 取左耳Y轴加速度（prepare_data_me.py逻辑：ear_accs_flat[:, 1]，即4071顶点的Y分量）
            left_ear_y_acc = features["hmd_position_global_full_gt_list"][:,12+1]
            plt.subplot(2, 2, i+1)
            plot_spectrum(left_ear_y_acc, description)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    plt.suptitle("AMASS Data Frequency Spectrum Analysis - Potential Physiological Signals", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()
    plt.savefig("amass_spectrum_analysis.png", dpi=300)
    print("Analysis complete. Results saved to amass_spectrum_analysis.png")
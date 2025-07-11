import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import signal

# 全局参数设置
SAMPLING_RATE = 60  # 假设IMU数据采样率为60Hz
BREATHING_BAND = [0.2, 0.5]  # 呼吸频率范围(Hz)
HEARTBEAT_BAND = [1.0, 2.5]  # 心跳频率范围(Hz)


def load_imu_data(sequence_length, noise_level):
    """
    模拟从AMASS数据集加载IMU数据

    参数:
    sequence_length (int): 序列长度（帧数）
    noise_level (float): 噪声水平

    返回:
    torch.Tensor: 模拟的单轴加速度信号
    """
    # 创建时间轴
    t = torch.linspace(0, sequence_length / SAMPLING_RATE, sequence_length)

    # 生成模拟的呼吸信号 (0.3Hz)
    breathing_signal = 0.01 * torch.sin(2 * torch.pi * 0.3 * t)

    # 生成模拟的心跳信号 (1.5Hz)
    heartbeat_signal = 0.005 * torch.sin(2 * torch.pi * 1.5 * t)

    # 生成模拟的运动噪声
    motion_noise = torch.randn(sequence_length) * noise_level

    # 组合信号
    accel_y = breathing_signal + heartbeat_signal + motion_noise

    return accel_y


def plot_spectrum(signal, title):
    """
    计算并绘制信号的频谱

    参数:
    signal (torch.Tensor): 输入信号
    title (str): 图表标题
    """
    # 计算信号长度
    N = len(signal)

    # 执行FFT
    fft_vals = torch.fft.fft(signal)

    # 计算频率轴
    fft_freq = torch.fft.fftfreq(N, 1.0 / SAMPLING_RATE)

    # 计算频谱幅度
    fft_magnitude = torch.abs(fft_vals)

    # 找到正频率的索引
    positive_freq_indices = fft_freq > 0

    # 绘制频谱
    plt.plot(fft_freq[positive_freq_indices], fft_magnitude[positive_freq_indices], label="Signal Spectrum")

    # 标记呼吸频带
    plt.axvline(BREATHING_BAND[0], color='r', linestyle='--', alpha=0.7)
    plt.axvline(BREATHING_BAND[1], color='r', linestyle='--', alpha=0.7)
    plt.axvspan(BREATHING_BAND[0], BREATHING_BAND[1], alpha=0.2, color='r', label="Breathing Band")

    # 标记心跳频带
    plt.axvline(HEARTBEAT_BAND[0], color='g', linestyle='--', alpha=0.7)
    plt.axvline(HEARTBEAT_BAND[1], color='g', linestyle='--', alpha=0.7)
    plt.axvspan(HEARTBEAT_BAND[0], HEARTBEAT_BAND[1], alpha=0.2, color='g', label="Heartbeat Band")

    # 设置图表属性
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)


if __name__ == "__main__":
    # 设置图表大小
    plt.figure(figsize=(20, 10))

    # 生成并分析"静止"场景的信号
    static_signal = load_imu_data(sequence_length=3000, noise_level=0.001)
    plt.subplot(2, 1, 1)
    plot_spectrum(static_signal, "Spectrum of 'Static' Scenario (Low Noise)")

    # 生成并分析"运动"场景的信号
    motion_signal = load_imu_data(sequence_length=3000, noise_level=0.05)
    plt.subplot(2, 1, 2)
    plot_spectrum(motion_signal, "Spectrum of 'Motion' Scenario (High Noise)")

    # 调整布局并显示
    plt.tight_layout()
    plt.show()
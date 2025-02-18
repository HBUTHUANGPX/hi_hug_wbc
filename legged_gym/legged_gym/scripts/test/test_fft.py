import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# 1. 读取 CSV 文件
df = pd.read_csv('/home/pi/HPX_Loco/DreamWaQ-Pi-Sim2sim/kpf_kdf_data.csv', header=None)

# 假设采样频率（根据你的实际采样情况来设置）
fs = 1000.0      # 这里假设 100 Hz
dt = 1.0 / fs

# 2. 选取需要滤波的信号列，示例取第 3 列
col_signal = 3  
# 只取数据的 2000:2500 区间作演示，你可根据需要修改
data = df.iloc[1000:2500, col_signal].values

N = len(data)
t = np.arange(N) * dt

# ==============  频域分析（原始信号）  ==============
freqs = np.fft.fftfreq(N, d=dt)
fft_vals = np.fft.fft(data)

# 只关注正频率
pos_idx = freqs >= 0
freqs_pos = freqs[pos_idx]
fft_vals_pos = fft_vals[pos_idx]
fft_amp = np.abs(fft_vals_pos)

# ==============  设计并应用低通滤波器  ==============
# 示例：低通滤波器截止频率设为 5 Hz，二阶
cutoff = 50      # 截止频率（Hz）
order = 2         # 二阶
nyquist = 0.5 * fs
normal_cutoff = cutoff / nyquist  # 归一化截止频率

# 构造低通滤波器
b, a = butter(order, normal_cutoff, btype='low', analog=False)

# 使用 filtfilt 进行双向滤波（零相位滤波）
filtered_data = filtfilt(b, a, data)

# ==============  滤波后信号的 FFT 分析  ==============
fft_vals_filtered = np.fft.fft(filtered_data)
fft_vals_filtered_pos = fft_vals_filtered[pos_idx]
fft_amp_filtered = np.abs(fft_vals_filtered_pos)

# ==============  绘图比较  ==============
plt.figure(figsize=(12, 8))

# 1) 原始信号（时域）
plt.subplot(2, 2, 1)
plt.plot(t, data, label='Original')
plt.title(f'原始信号 (Column {col_signal})')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()

# 2) 滤波后信号（时域）
plt.subplot(2, 2, 2)
plt.plot(t, filtered_data, 'r', label='Filtered (Low-pass)')
plt.title('滤波后信号')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()

# 3) 原始信号（频域）
plt.subplot(2, 2, 3)
plt.stem(freqs_pos, fft_amp, use_line_collection=True)
plt.title('原始信号 FFT 幅值谱')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude')
plt.xlim([0, fs/2])
plt.grid(True)

# 4) 滤波后信号（频域）
plt.subplot(2, 2, 4)
plt.stem(freqs_pos, fft_amp_filtered, use_line_collection=True, linefmt='C1-', markerfmt='C1o')
plt.title('滤波后信号 FFT 幅值谱')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude')
plt.xlim([0, fs/2])
plt.grid(True)

plt.tight_layout()
plt.show()

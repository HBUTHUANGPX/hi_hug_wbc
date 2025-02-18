import numpy as np
from scipy.signal import butter
from rtf import RealTimeFilter

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # 假设我们有一些带噪声的测试信号 (示例正弦信号 + 高频噪声)
    fs = 1000.0
    t = np.arange(0, 1, 1/fs)  # 1秒，100点
    freq_signal = 0.6          # 3 Hz 基波
    x_clean = np.sin(2*np.pi*freq_signal * t)
    noise = 0.5 * np.random.randn(len(t))
    x_noisy = x_clean + noise

    # 1. 创建一个二阶低通滤波器，截止频率 5 Hz
    my_filter = RealTimeFilter(fs=fs, cutoff=50.0, btype='low', order=2)

    # 2. 逐点输入 x_noisy，实时获取滤波输出
    y_filtered = []
    for sample in x_noisy:
        y_out = my_filter.filter_sample(sample)
        y_filtered.append(y_out)

    y_filtered = np.array(y_filtered)

    # 3. 画图对比
    plt.figure(figsize=(8,4))
    plt.plot(t, x_noisy, label='Noisy input')
    plt.plot(t, y_filtered, label='Filtered output', color='red', linewidth=2)
    plt.title("Real-time 2nd-order Low-pass Filter Demo")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.show()

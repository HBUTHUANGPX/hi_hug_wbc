import numpy as np
from scipy.signal import butter

class RealTimeFilter:
    """
    二阶IIR滤波器，用于实时滤波单点数据。

    参数:
    ----
    fs : float
        采样频率 (Hz)
    cutoff : float
        滤波器截止频率 (Hz)，若为 lowpass 即表示保留 cutoff 以下的频率
    btype : str
        滤波器类型，可选 'low', 'high', 'bandpass', 'bandstop'。
    order : int
        滤波器阶数，这里默认 2。
    """

    def __init__(self, fs, cutoff, btype='low', order=2):
        self.fs = fs
        self.cutoff = cutoff
        self.btype = btype
        self.order = order

        # 计算滤波器系数
        # butter() 返回 (b, a), 其中 a[0] = 1
        nyq = 0.5 * fs
        normal_cutoff = np.array(cutoff) / nyq  # 若 bandpass 需要传递数组 [low, high]
        self.b, self.a = butter(order, normal_cutoff, btype=btype, analog=False)

        # 仅保留二阶情况演示: a, b 的长度 = order+1 = 3
        # 一般 a[0] = 1, a[1], a[2], b[0], b[1], b[2]
        # 状态初始化: (x[n-1], x[n-2]) 和 (y[n-1], y[n-2])
        self.x_history = [0.0, 0.0]  # 保存前两次输入
        self.y_history = [0.0, 0.0]  # 保存前两次输出

    def filter_sample(self, x_now):
        """
        实时滤波单个输入 x_now, 返回滤波后的单个输出 y_now。
        """
        # 取出系数 (长度均为 3)
        b0, b1, b2 = self.b
        a0, a1, a2 = self.a

        # 当前输入记为 x[n], 历史输入为 x[n-1], x[n-2]
        x1, x2 = self.x_history
        y1, y2 = self.y_history

        # 差分方程：
        # y[n] = (b0*x[n] + b1*x[n-1] + b2*x[n-2] 
        #         - a1*y[n-1] - a2*y[n-2]) / a0
        # 因为 a0 通常 = 1 (Butterworth 默认如此)
        y_now = (b0*x_now + b1*x1 + b2*x2 - a1*y1 - a2*y2) / a0

        # 更新历史状态:
        # 把 x[n], y[n] 变成下一次的 x[n-1], y[n-1]
        self.x_history = [x_now, x1]
        self.y_history = [y_now, y1]

        return y_now

    def reset_state(self):
        """
        重置滤波器状态，清空 x_history, y_history。
        """
        self.x_history = [0.0, 0.0]
        self.y_history = [0.0, 0.0]

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib.widgets import Slider

def simulate_overshoot_response(omega_n, zeta, t_end, step_amplitude):
    # 定义时间向量
    t = np.linspace(0, t_end, num=1000)

    # 定义二阶系统传递函数 H(s) = omega_n^2 / (s^2 + 2*zeta*omega_n*s + omega_n^2)
    system = signal.TransferFunction([omega_n**2], [1, 2*zeta*omega_n, omega_n**2])

    # 模拟阶跃响应
    t, response = signal.step(system, T=t)

    # 调整阶跃幅度
    response *= step_amplitude

    return t, response

def update(val):
    # 获取滑动条的当前值
    zeta = zeta_slider.val
    omega_n = omega_n_slider.val

    # 计算新的响应
    t, actual_response = simulate_overshoot_response(omega_n, zeta, t_end, step_amplitude)

    # 更新图像数据
    line.set_ydata(actual_response)
    fig.canvas.draw_idle()

# 二阶系统参数
t_end = 0.1  # 模拟结束时间
step_amplitude = 1.0  # 阶跃信号幅度

# 初始参数
initial_zeta = 0.3
initial_omega_n = 40.0

# 模拟初始系统响应
t, actual_response = simulate_overshoot_response(initial_omega_n, initial_zeta, t_end, step_amplitude)

# 创建图形和滑动条
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.35)  # 调整底部空间以容纳两个滑动条
desired_response = np.ones_like(t) * step_amplitude
line, = plt.plot(t, actual_response, 'b-', label='Actual Response')
plt.plot(t, desired_response, 'r--', label='Desired Output')
plt.xlabel('Time [s]')
plt.ylabel('Response')
plt.title('Second Order System Response with Overshoot')
plt.legend()
plt.grid(True)

# 阻尼比滑动条设置
axcolor = 'lightgoldenrodyellow'
ax_zeta_slider = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
zeta_slider = Slider(ax_zeta_slider, 'Damping Ratio', 0.01, 1.0, valinit=initial_zeta)

# 自然频率滑动条设置
ax_omega_n_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
omega_n_slider = Slider(ax_omega_n_slider, 'Natural Frequency', 100, 200.0, valinit=initial_omega_n)

# 绑定滑动条更新事件
zeta_slider.on_changed(update)
omega_n_slider.on_changed(update)

# 显示图形
plt.show()

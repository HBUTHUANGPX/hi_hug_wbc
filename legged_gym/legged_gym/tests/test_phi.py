import numpy as np
import matplotlib.pyplot as plt

# 定义分段函数
def normalized_phase(phi_i, phi_stance):
    """
    计算归一化相位变量 $\bar{\phi_i}$
    :param phi_i: 原始相位变量
    :param phi_stance: 支撑相位结束点
    :return: 归一化相位变量 $\bar{\phi_i}$
    """
    if phi_i < phi_stance:
        return 0.5 * (phi_i / phi_stance)
    else:
        return 0.5 + 0.5 * ((phi_i - phi_stance) / (1 - phi_stance))

# 定义参数
phi_stance = 0.6 # 支撑相位结束点
phi_i_values = np.linspace(0, 1, 1000)  # 生成从 0 到 1 的相位变量

# 计算归一化相位变量
phi_bar_values = [np.sin(2*np.pi*normalized_phase(phi_i, phi_stance))for phi_i in phi_i_values]

# 绘制分段函数
plt.figure(figsize=(8, 6))
plt.plot(phi_i_values, phi_bar_values, label="$\\bar{\\phi_i}$", color='blue')
plt.axvline(x=phi_stance, color='red', linestyle='--', label="$\\phi_{{stance}}$")
plt.xlabel("$\\phi_i$", fontsize=12)
plt.ylabel("$\\bar{\\phi_i}$", fontsize=12)
plt.title('Normalized Phase Variable $\\bar{\phi_i}$', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()

import torch
import numpy as np
import matplotlib.pyplot as plt
def normalized_phase(phi_i, phi_stance):
    phy_i_bar = torch.zeros_like(phi_i)
    mask_sma = phi_i<phi_stance
    if torch.any(mask_sma):
        phy_i_bar[mask_sma] = (0.5 * (phi_i / phi_stance))[mask_sma]
    mask_big = phi_i>=phi_stance
    if torch.any(mask_big):
        phy_i_bar[mask_big] = (0.5 + 0.5 * ((phi_i - phi_stance)/(1 - phi_stance)))[mask_big]
    return phy_i_bar

# 假设我们有一个n×1的向量
n = 4096  # 示例大小
phi_stance = 0.5 * torch.ones(n, 1)
phi_0_5 = normalized_phase(torch.ones(n, 1) * 0.5,phi_stance)  # 随机生成一个n×1的phi_0_5
phi_0_75 = normalized_phase(torch.ones(n, 1) * 0.75,phi_stance)  # 随机生成一个n×1的phi_0_75
phi_1_0 = normalized_phase(torch.ones(n, 1),phi_stance)  # 固定为1

p_s_z = torch.zeros(n, 1)  # 随机生成一个n×1的p_s_z
p_e_z = torch.zeros(n, 1)  # 随机生成一个n×1的p_e_z
l_t = torch.ones(n, 1)*0.5  # 随机生成一个n×1的l_t
print("p_s_z: ",p_s_z.size())
def return_A(s_phi,e_phi):
    # 定义多项式的系数矩阵
    A1_1 = torch.cat([
        s_phi**5, s_phi**4, s_phi**3, s_phi**2, s_phi, torch.ones(n, 1)
    ], dim=1)  # n×6 矩阵
    A1_2 = torch.cat([
        e_phi**5, e_phi**4, e_phi**3, e_phi**2, e_phi, torch.ones(n, 1)
    ], dim=1)  # n×6 矩阵
    A1_3 = torch.cat([
        5*s_phi**4, 4*s_phi**3, 3*s_phi**2, 2*s_phi, torch.ones(n, 1), torch.zeros(n, 1)
    ], dim=1)  # n×6 矩阵
    A1_4 = torch.cat([
        5*e_phi**4, 4*e_phi**3, 3*e_phi**2, 2*e_phi, torch.ones(n, 1), torch.zeros(n, 1)
    ], dim=1)  # n×6 矩阵
    A1_5 = torch.cat([
        20*s_phi**3, 12*s_phi**2, 6*s_phi, 2*torch.ones(n, 1), torch.zeros(n, 1), torch.zeros(n, 1)
    ], dim=1)  # n×6 矩阵
    A1_6 = torch.cat([
        20*e_phi**3, 12*e_phi**2, 6*e_phi, 2*torch.ones(n, 1), torch.zeros(n, 1), torch.zeros(n, 1)
    ], dim=1)  # n×6 矩阵
    return torch.stack([A1_1,A1_2,A1_3,A1_4,A1_5,A1_6],dim=1)# n×6×6 矩阵

A1 = return_A(phi_0_5,phi_0_75)
A2 = return_A(phi_0_75,phi_1_0)
# 定义常数项矩阵
b1 = torch.stack([p_s_z, l_t, torch.zeros(n, 1), torch.zeros(n, 1), torch.zeros(n, 1), torch.zeros(n, 1)], dim=1)
b2 = torch.stack([l_t, p_e_z, torch.zeros(n, 1), torch.zeros(n, 1), torch.zeros(n, 1), torch.zeros(n, 1)], dim=1)


# 求解系数矩阵a1和a2
a1 = torch.linalg.solve(A1, b1)
a2 = torch.linalg.solve(A2, b2)

# 计算目标轨迹
phi = torch.linspace(0, 1.0, n).reshape(-1, 1)  # 从0.5到1.0的phi值
nm_phi = normalized_phase(phi,phi_stance)
print(a1[:, 0].size())
print("phi: ",phi.size())
# 
a = a1[:, 0] * nm_phi**5 + a1[:, 1] * nm_phi**4 + a1[:, 2] * nm_phi**3 + a1[:, 3] * nm_phi**2 + a1[:, 4] * nm_phi + a1[:, 5]
b = a2[:, 0] * nm_phi**5 + a2[:, 1] * nm_phi**4 + a2[:, 2] * nm_phi**3 + a2[:, 3] * nm_phi**2 + a2[:, 4] * nm_phi + a2[:, 5]
print("a: ",a.size())

l_t_target = torch.where(
    phi < 0.75,
    a,
    b,
    # a1[:, 0:1] * nm_phi**5 + a1[:, 1:2] * nm_phi**4 + a1[:, 2:3] * nm_phi**3 + a1[:, 3:4] * nm_phi**2 + a1[:, 4:5] * nm_phi + a1[:, 5:6],
    # a2[:, 0:1] * nm_phi**5 + a2[:, 1:2] * nm_phi**4 + a2[:, 2:3] * nm_phi**3 + a2[:, 3:4] * nm_phi**2 + a2[:, 4:5] * nm_phi + a2[:, 5:6]
)
l_t_target[phi < 0.5] = 0
print("l_t_target: ",l_t_target.size())

# 可视化目标轨迹
l_t_target_np = l_t_target.detach().numpy()
print("l_t_target: ",l_t_target_np.shape)
print("l_t_target_np[0,...]: ",l_t_target_np.shape)
print("l_t_target_np[0,...]: ",l_t_target_np.shape)

plt.plot(phi.numpy(), l_t_target_np)
plt.xlabel("Normalized Phase ($\\phi$)")
plt.ylabel("Foot Height ($l_t^{target}$)")
plt.title("Foot Trajectory Planning for Multiple Phases")
plt.grid(True)
plt.show()

import torch
import matplotlib.pyplot as plt
import numpy as np


def rotate_around_y_axis(angle):
    # print(angle.dim())
    if angle.dim() == 1:
        rt_matrices = torch.tensor(
            [
                [torch.cos(angle), 0, torch.sin(angle), 0],
                [0, 1, 0, 0],
                [-torch.sin(angle), 0, torch.cos(angle), 0],
                [0, 0, 0, 1],
            ]
        )
    elif angle.dim() == 2:
        rt_matrices = torch.zeros((angle.size()[0], angle.size()[1], 4, 4))
        # print(torch.cos(angle).size())
        # print(rt_matrices[:, :, 0, 0].size())
        rt_matrices[:, :, 0, 0] = torch.cos(angle)
        rt_matrices[:, :, 0, 2] = torch.sin(angle)
        rt_matrices[:, :, 1, 1] = 1
        rt_matrices[:, :, 2, 0] = -torch.sin(angle)
        rt_matrices[:, :, 2, 2] = torch.cos(angle)
        rt_matrices[:, :, 3, 3] = 1
    return rt_matrices


def translate(point):
    tr_matrix = torch.tensor(
        [
            [1, 0, 0, point[0]],
            [0, 1, 0, point[1]],
            [0, 0, 1, point[2]],
            [0, 0, 0, 1],
        ]
    )
    return tr_matrix


p1 = torch.tensor([0.0, 0.0, -0.15])
p2 = torch.tensor([0.0, 0.0, -0.15])
r1 = torch.tensor([torch.pi / 6])
r2 = torch.tensor([-torch.pi / 3])

rot1 = rotate_around_y_axis(r1)
rot2 = rotate_around_y_axis(r2)
tra1 = translate(p1)
tra2 = translate(p2)

homo1 = rot1 @ tra1
homo2 = rot1 @ tra1 @ rot2 @ tra2
print(homo2)
# 假设一些示例数据
num_envs = 10
num_intervals = 2000
episode_length_buf = (
    torch.arange(num_intervals, dtype=torch.float32).unsqueeze(0).repeat(num_envs, 1)
)
dt = 0.001  # 每个时间步的时间间隔
cycle_time = 0.4  # 一个完整步态周期的时间
bias = 0.3
y_bias =0.9
eps = 0.3
scale = 0.14 
device = "cpu"  # 或者 'cuda'，视情况而定

# 计算相位
phase = episode_length_buf * dt / cycle_time
# print(phase)
# 计算正弦位置
sin_pos = torch.sin(2 * torch.pi * phase)

# 初始化支撑相掩码
stance_mask = torch.zeros((num_envs, num_intervals, 2), device=device)

# 左脚支撑相
stance_mask[:, :, 0] = sin_pos >= 0
# 右脚支撑相
stance_mask[:, :, 1] = sin_pos < 0
# 双支撑相
stance_mask[torch.abs(sin_pos) < bias] = 1


p1 = torch.tensor([0.0, 0.0, -0.15])
p2 = torch.tensor([0.0, 0.0, -0.15])
tra1 = translate(p1)
tra2 = translate(p2)



ss = sin_pos.clone() + bias
l_angle = ss / torch.sqrt(ss**2.0 + eps**2.0) - y_bias
# l_angle[l_angle > 0] = 0
# l_angle += y_bias
# l_angle *= 0
r1_l = -l_angle * scale + 0.25
r2_l = l_angle * scale* 2 - 0.65
rot1_l = rotate_around_y_axis(r1_l)
rot2_l = rotate_around_y_axis(r2_l)
homo_l = rot1_l @ tra1 @ rot2_l @ tra2

ss = sin_pos.clone() - bias
angle = ss / torch.sqrt(ss**2.0 + eps**2.0) + y_bias
# angle[angle < 0] = 0
# angle -= y_bias
# angle[torch.abs(sin_pos) < 0.5]= 0
r1_r = angle * scale + 0.25
r2_r = -angle * scale * 2 - 0.65
rot1_r = rotate_around_y_axis(r1_r)
rot2_r = rotate_around_y_axis(r2_r)
homo_r = rot1_r @ tra1 @ rot2_r @ tra2


# 转换为 numpy 数组以便于可视化
stance_mask_np = stance_mask.cpu().numpy()
sin_pos_np = sin_pos.cpu().numpy()
end_pos_l_np = homo_l[:, :, 2, 3].cpu().numpy()
end_pos_r_np = homo_r[:, :, 2, 3].cpu().numpy()
angle = angle.cpu().numpy()
t_np = (episode_length_buf * dt).cpu().numpy()
# 创建一个新的图形
plt.figure(figsize=(10, 5))

# 绘制左脚支撑相
# plt.plot(t_np[0,:],(stance_mask_np[0, :, 0]-0.5)*0.05, label="Left Foot Stance", marker=".")

# 绘制右脚支撑相
# plt.plot(t_np[0,:],(stance_mask_np[0, :, 1]-0.5)*0.05, label="Right Foot Stance", marker=".")

# plt.plot(t_np[0,:],sin_pos_np[0, :] * 0.05, label="Sin Pos", marker=".")
# plt.plot(t_np[0,:],angle[0, :] * 0.1 , label="r1_r", marker=".")
plt.plot(t_np[0, :], end_pos_l_np[0, :], label="End pos l", marker=".")
# plt.plot(t_np[0,:],end_pos_l_np[0, :] - np.min(end_pos_l_np[0, :]), label="End pos l", marker=".")
plt.plot(t_np[0, :], end_pos_r_np[0, :], label="End pos r", marker=".")
# plt.plot(t_np[0,:],np.abs(end_pos_l_np[0, :] - end_pos_r_np[0, :]), label="derta", marker=".")
# 添加图例
print(np.min(end_pos_l_np[0, :])-np.min(end_pos_r_np[0, :]))
plt.legend()

# 添加标题和标签
plt.title("Stance Mask Visualization")
plt.xlabel("Environment Index")
plt.ylabel("Stance Value")

# 显示图形
plt.show()
# print(np.arccos(0.5)/np.pi*180)
# print(np.arccos((1-(0.08*0.08)/(2*0.285*0.285)))/np.pi*180)
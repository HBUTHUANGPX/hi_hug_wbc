import torch
import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_envs = 4096
window_size = 400
tau_samples = 250

alpha = 1.0
beta = 1.0

total_time_steps = 50000
time_step = 0

dt = 0.001
t = 0.0

buffer_left = torch.zeros(num_envs, window_size)
buffer_right = torch.zeros(num_envs, window_size)


def update_reward(foot_pos_z_left, foot_pos_z_right, reward_instantaneous,plot):
    global buffer_left ,buffer_right,buffer_left_sub,buffer_right_sub
    # 更新缓冲区
    buffer_left = torch.roll(buffer_left, shifts=-1, dims=1)
    buffer_left[:, -1] = foot_pos_z_left
    buffer_right = torch.roll(buffer_right, shifts=-1, dims=1)
    buffer_right[:, -1] = foot_pos_z_right
    # 计算均值并减去
    mean_left = torch.mean(buffer_left, dim=1, keepdim=True)
    buffer_left_sub = buffer_left# - mean_left
    mean_right = torch.mean(buffer_right, dim=1, keepdim=True)
    buffer_right_sub = buffer_right# - mean_right

    # 计算FFT
    X_l = torch.fft.fft(buffer_left_sub, dim=1)
    X_r = torch.fft.fft(buffer_right_sub, dim=1)

    # 计算互相关
    cross_power = X_l * torch.conj(X_r)
    R = torch.fft.ifft(cross_power, dim=1)
    print(R.size())
    correlation_value = R[:, tau_samples].real

    # 计算幅度谱差
    magnitude_spectrum_l = torch.abs(X_l)
    magnitude_spectrum_r = torch.abs(X_r)
    diff_magnitude = torch.mean(
        (magnitude_spectrum_l - magnitude_spectrum_r) ** 2, dim=1
    )

    # 计算频域奖励
    reward_frequency = alpha * correlation_value - beta * diff_magnitude

    # 总奖励
    total_reward = reward_instantaneous + reward_frequency / window_size
    if plot:
        return diff_magnitude,correlation_value,total_reward
    else:
        return total_reward

dec = 10
cnt = 0
import time,queue,threading
data_queue = queue.Queue()
plot_num = 3
def plot_data(data_queue):
    print("plot_data")
    plt.ion()  # 开启交互模式
    fig, axs = plt.subplots(plot_num, 1, figsize=(10, 12))  # 创建 8 个子图
    lines = [ax.plot([], [])[0] for ax in axs]  # 初始化每个子图的线条
    xdata = [[] for _ in range(plot_num)]  # 存储每个子图的 x 数据
    ydata = [[] for _ in range(plot_num)]  # 存储每个子图的 y 数据

    while True:
        if not data_queue.empty():
            merged_tensor = data_queue.get()
            # print("bb")
            for i in range(plot_num):
                xdata[i].append(len(xdata[i]))
                ydata[i].append(merged_tensor[i].item())
                lines[i].set_data(xdata[i], ydata[i])
                axs[i].relim()
                axs[i].autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()
        else:
            # print("cc")
            time.sleep(0.1)
      
if __name__ == "__main__":
   
    plot_thread = threading.Thread(target=plot_data, args=(data_queue,))
    plot_thread.daemon = True
    plot_thread.start()
    while time_step < total_time_steps:
        time_step += 1
        cnt+=1
        t = time_step * dt * torch.ones(num_envs)
        foot_pos_z_left = torch.sin(t*2*torch.pi)
        foot_pos_z_right = torch.sin(t*2*torch.pi - torch.pi)*0
        if cnt >10:
            diff_magnitude,correlation_value,rew = update_reward(foot_pos_z_left,foot_pos_z_right,0,True)
            cnt = 0
            # print(diff_magnitude.size())
            merged_tensor = torch.cat([
                diff_magnitude.unsqueeze(1),
                correlation_value.unsqueeze(1),
                rew.unsqueeze(1),], dim=1)[0,:]
            data_queue.put(merged_tensor)  
            print(time_step)
    
    
    

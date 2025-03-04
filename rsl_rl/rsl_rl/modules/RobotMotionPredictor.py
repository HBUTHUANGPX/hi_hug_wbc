import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class PeriodicMotionEncoder(nn.Module):
    """机器人运动预测网络，结合PAE周期性特性和VAE变分特性"""
    
    def __init__(self, robot_dim=56, time_dim=101, latent_dim=16, velocity_dim=3, beta=0.1):
        """
        参数:
            robot_dim: 观测量维度
            time_dim: 时间帧数 (历史帧+当前帧)
            latent_dim: 隐藏空间维度
            velocity_dim: 速度估计维度
            beta: VAE损失中的beta权重
        """
        super(PeriodicMotionEncoder, self).__init__()
        
        self.robot_dim = robot_dim
        self.time_dim = time_dim
        self.latent_dim = latent_dim
        self.velocity_dim = velocity_dim
        self.beta = beta
        
        # 常量参数
        self.tpi = Parameter(torch.tensor([2.0*np.pi], dtype=torch.float32), requires_grad=False)
        self.args = Parameter(torch.from_numpy(np.linspace(0, 1, time_dim, dtype=np.float32)), requires_grad=False)
        self.freqs = Parameter(torch.fft.rfftfreq(time_dim)[1:] * time_dim, requires_grad=False)  # 移除DC频率
        
        # 相位预测网络 (fc)
        self.phase_net = nn.Sequential(
            nn.Linear(robot_dim * time_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 2)  # 输出sin和cos分量
        )
        
        # 信号解码网络 (fdecode)
        self.signal_decode = nn.Sequential(
            nn.Linear(robot_dim * time_dim, 1024),
            nn.ELU(),
            nn.Linear(1024, 512),
            nn.ELU(),
            nn.Linear(512, robot_dim * time_dim)
        )
        
        # 编码网络 (encode)
        self.encode = nn.Sequential(
            nn.Linear(robot_dim * (time_dim-1), 512),  # time_dim-1 因为只用前100帧
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU()
        )
        
        # VAE的均值和方差预测层
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_var = nn.Linear(256, latent_dim)
        
        # 速度预测层
        self.fc_velocity = nn.Linear(256, velocity_dim)
        
        # 解码网络 (decode)
        self.decode = nn.Sequential(
            nn.Linear(latent_dim + velocity_dim, 256),
            nn.ELU(),
            nn.Linear(256, 512),
            nn.ELU(),
            nn.Linear(512, robot_dim)  # 预测最后一帧
        )
    
    def FFT(self, x, dim):
        """
        计算输入的频域参数
        
        参数:
            x: 输入张量
            dim: 执行FFT的维度
            
        返回:
            freq: 频率
            amp: 振幅
            offset: 偏移
        """
        batch_size = x.shape[0]
        
        # 调整张量形状以便对每个机器人的每个观测通道进行FFT
        x_reshaped = x.reshape(batch_size * self.robot_dim, self.time_dim)
        
        # 执行FFT
        rfft = torch.fft.rfft(x_reshaped, dim=1)
        magnitudes = rfft.abs()
        spectrum = magnitudes[:, 1:]  # 去除DC分量
        power = spectrum**2
        
        # 计算频率 (加权平均)
        freq = torch.sum(self.freqs * power, dim=1) / (torch.sum(power, dim=1) + 1e-8)
        freq = freq.reshape(batch_size, self.robot_dim)
        
        # 计算振幅
        amp = 2 * torch.sqrt(torch.sum(power, dim=1)) / self.time_dim
        amp = amp.reshape(batch_size, self.robot_dim)
        
        # 计算偏移 (DC分量)
        offset = rfft.real[:, 0] / self.time_dim
        offset = offset.reshape(batch_size, self.robot_dim)
        
        return freq, amp, offset
    
    def reparameterize(self, mu, logvar):
        """VAE重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入张量 [n, 101, 56]
            
        返回:
            y_e: 重建的信号 [n, 101, 56]
            v_e: 预测的速度 [n, 3]
            latent: 隐藏表示 [n, 16]
            obs_e: 预测的下一帧观测 [n, 56]
            mu: VAE均值
            logvar: VAE对数方差
        """
        batch_size = x.shape[0]
        
        # 第一步: FFT变换和信号重建
        x_flat = x.reshape(batch_size, -1)  # [n, 101*56]
        
        # 提取频率参数
        freq, amp, offset = self.FFT(x, dim=1)
        
        # 预测相位
        phase_output = self.phase_net(x_flat)
        phase = torch.atan2(phase_output[:, 1], phase_output[:, 0]) / self.tpi
        phase = phase.unsqueeze(1).repeat(1, self.robot_dim)
        
        # 构建重建信号 (4096为最大批次大小)
        signal = torch.zeros((batch_size, self.time_dim, self.robot_dim), device=x.device)
        
        # 对每个机器人的每个时间步生成信号
        for t in range(self.time_dim):
            # a * sin(2π * (f * t + p)) + b
            signal[:, t, :] = amp * torch.sin(self.tpi * (freq * self.args[t] + phase)) + offset
        
        # 转换信号维度
        signal_flat = signal.reshape(batch_size, -1)  # [n, 101*56]
        
        # 解码信号
        y_e_flat = self.signal_decode(signal_flat)
        y_e = y_e_flat.reshape(batch_size, self.time_dim, self.robot_dim)  # [n, 101, 56]
        
        # 第二步: 编码前100帧
        history = y_e[:, :self.time_dim-1, :].reshape(batch_size, -1)  # [n, 100*56]
        encoded = self.encode(history)
        
        # VAE参数
        mu = self.fc_mu(encoded)
        logvar = self.fc_var(encoded)
        
        # 重参数化
        latent = self.reparameterize(mu, logvar)
        
        # 速度预测
        v_e = self.fc_velocity(encoded)
        
        # 第三步: 解码预测下一帧
        decoder_input = torch.cat([latent, v_e], dim=1)
        obs_e = self.decode(decoder_input)
        
        return y_e, v_e, latent, obs_e, mu, logvar
    
    def loss_function(self, y_e, x, v_e, v_true, obs_e, mu, logvar):
        """
        计算总损失
        
        参数:
            y_e: 重建的信号 [n, 101, 56]
            x: 原始输入 [n, 101, 56]
            v_e: 预测的速度 [n, 3]
            v_true: 真实速度 [n, 3]
            obs_e: 预测的下一帧观测 [n, 56]
            mu: VAE均值
            logvar: VAE对数方差
            
        返回:
            total_loss: 总损失
            loss_dict: 各部分损失的字典
        """
        # 重建损失 (y_e 和 x)
        recon_loss = F.mse_loss(y_e, x)
        
        # 速度预测损失
        velocity_loss = F.mse_loss(v_e, v_true)
        
        # VAE KL散度损失
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / x.shape[0]  # 归一化
        
        # 下一帧预测损失
        next_frame_loss = F.mse_loss(obs_e, x[:, -1, :])
        
        # 总损失
        total_loss = recon_loss + velocity_loss + self.beta * kl_loss + next_frame_loss
        
        loss_dict = {
            'total': total_loss.item(),
            'reconstruction': recon_loss.item(),
            'velocity': velocity_loss.item(),
            'kl': kl_loss.item(),
            'next_frame': next_frame_loss.item()
        }
        
        return total_loss, loss_dict


# 使用示例
if __name__ == "__main__":
    # 创建模型
    model = PeriodicMotionEncoder()
    
    # 生成测试数据
    batch_size = 8
    x = torch.randn(batch_size, 101, 56)  # 测试输入
    v_true = torch.randn(batch_size, 3)   # 真实速度
    
    # 前向传播
    y_e, v_e, latent, obs_e, mu, logvar = model(x)
    
    # 计算损失
    total_loss, loss_dict = model.loss_function(y_e, x, v_e, v_true, obs_e, mu, logvar)
    
    # 输出结果
    print(f"输入形状: {x.shape}")
    print(f"重建信号形状: {y_e.shape}")
    print(f"预测速度形状: {v_e.shape}")
    print(f"潜在向量形状: {latent.shape}")
    print(f"预测下一帧形状: {obs_e.shape}")
    print(f"损失值: {loss_dict}") 
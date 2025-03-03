# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.


import math
import numpy as np
import mujoco, mujoco_viewer
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R

from legged_gym import LEGGED_GYM_ROOT_DIR

# from legged_gym.envs.pai.pai_config import PaiRoughCfg
# from legged_gym.envs.hi_none_phase.hi_config_none_phase import HiNonePhaseCfg as cfg
from legged_gym.envs.hicl_hug.hicl_config_hug import HiclHugCfg as cfg

# from legged_gym.envs.pai.pai_config_demo import PaiDemoRoughCfg
from isaacgym.torch_utils import *

import torch
import onnxruntime as ort
import matplotlib.pyplot as plt
import csv
import pandas as pd
import threading
import queue
import pygame
import time
from legged_gym.scripts.test import pin_mj

base_path = "/home/hpx/HPX_Loco/hi_hug_wbc/legged_gym/resources/robots"
robot_patch = "/hi_cl_23_240925/urdf/hi_cl_23_240925_rl.urdf"
_pin = pin_mj(base_path + robot_patch)
hh = _pin.get_foot_pos(np.array([0] * (12), dtype=float))
import pandas as pd


class cmd:
    vx = 0.0
    vy = 0.0
    dyaw = 0.0


class env:
    obs = cfg.env.num_observations
    num_single_obs = obs
    frame_stack = cfg.env.num_obs_hist + 1
    obs_his = obs * cfg.env.num_obs_hist


def quat_rotate_inverse(q, v):
    q_w = q[-1]
    q_vec = q[:3]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c


def quat_rotate_inverse_ori(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = (
        q_vec
        * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1)
        * 2.0
    )
    return a - b + c


def get_obs(data):
    """Extracts an observation from the mujoco data structure"""
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor("orientation").data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
    omega = data.sensor("angular-velocity").data.astype(np.double)
    gvec = r.apply(np.array([0.0, 0.0, -1.0]), inverse=True).astype(np.double)
    return (q, dq, quat, v, omega, gvec)


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    # print("p:", (target_q - q) * kp )
    # print("d", (target_dq - dq) * kd)
    return (target_q - q) * kp + (target_dq - dq) * kd


def process_value(value, threshold=0.01):
    """Process the input value, setting it to zero if below a threshold."""
    return 0 if abs(value) < threshold else value


def run_mujoco(control_queue: queue.Queue, policy, cfg: cfg):
    """
    Run the Mujoco simulation using the provided policy and configuration.

    Args:
        policy: The policy used for controlling the simulation.
        cfg: The configuration object containing simulation settings.

    Returns:
        None
    """
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    viewer = mujoco_viewer.MujocoViewer(model, data)

    target_q = np.zeros((cfg.env.num_actions), dtype=np.double)
    action = np.zeros((cfg.env.num_actions), dtype=np.double)
    obs_history = torch.zeros(1, env.obs_his, dtype=torch.float)

    hist_obs = deque()
    for _ in range(env.frame_stack):
        hist_obs.append(np.zeros([1, env.num_single_obs], dtype=np.double))

    count_lowlevel = 0
    count_csv = 0
    clip = []
    for joint, vals in cfg.init_state.dof_pos_range_virtual.items():
        print(f"joint: {joint} vals: {vals}")
        clip.append(vals)
    cl = np.array(clip, dtype=np.float32)

    command = [0, 0, 0]  # vx, vy, dyaw
    phy_1 = 0
    phy_2 = 0
    h_in = np.zeros((1, 1, 64), dtype=np.float32)
    c_in = np.zeros((1, 1, 64), dtype=np.float32)
    joint_angles_all_frames = []
    joint_velocities_all_frames = []
    joint_acc_all_frames = []
    old_dq = [0] * 12
    old_q = [0] * 12
    for _ in tqdm(
        range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)),
        desc="Simulating...",
    ):
        # Obtain an observation
        q, dq, quat, v, omega, gvec = get_obs(data)
        q = q[-cfg.env.num_actions :]
        dq = dq[-cfg.env.num_actions :]

        for i in range(6):
            tmpq = q[i]
            q[i] = q[i + 6]
            q[i + 6] = tmpq

            tmpdq = dq[i]
            dq[i] = dq[i + 6]
            dq[i + 6] = tmpdq

        # 1000hz -> 100hz
        if count_lowlevel % cfg.sim_config.decimation == 0:
            obs = torch.zeros(1, env.obs, dtype=torch.float)
            _q = quat
            _v = np.array([0.0, 0.0, -1.0])
            projected_gravity = quat_rotate_inverse(_q, _v)
            if cfg.env.obs_with_base_lin_vel:
                add = 3
                obs[0, 3:6] = torch.tensor(v, dtype=torch.double)  # 3
            else:
                add = 0
            # base_ang_vel
            obs[0, :3] = torch.tensor(
                omega * cfg.normalization.obs_scales.ang_vel, dtype=torch.double
            )  # 3
            # projected_gravity
            obs[0, 3 + add : 6 + add] = torch.tensor(
                projected_gravity, dtype=torch.double
            )  # 3
            # commands

            try:
                axis_values = control_queue.get_nowait()
                # Process the values
                vx, vy, dyaw, level = [process_value(val) for val in axis_values]
                command = [
                    -vx * cfg.commands.ranges.lin_vel_x[1],
                    -vy * cfg.commands.ranges.lin_vel_y[1],
                    -dyaw * cfg.commands.ranges.ang_vel_yaw[1],
                ]
                # print(f"Received control input: vx={vx}, vy={vy}, dyaw={dyaw}")
            except queue.Empty:
                pass
            obs[0, 6 + add] = torch.tensor(
                command[0] * cfg.normalization.obs_scales.lin_vel, dtype=torch.double
            )
            obs[0, 7 + add] = torch.tensor(
                command[1] * cfg.normalization.obs_scales.lin_vel, dtype=torch.double
            )
            obs[0, 8 + add] = torch.tensor(
                command[2] * cfg.normalization.obs_scales.ang_vel, dtype=torch.double
            )
            # standing_command_mask
            if command[0] != 0 or command[1] != 0 or command[2] != 0:
                standing_command_mask = torch.tensor(0.0, dtype=torch.double)
                psi = 0.5
            else:
                # standing_command_mask = torch.tensor(0.0, dtype=torch.double)
                standing_command_mask = torch.tensor(1.0, dtype=torch.double)
                psi = 0

            obs[0, 9 + add] = standing_command_mask

            # dof_pos
            obs[0, 10 + add : 22 + add] = torch.tensor(
                q * cfg.normalization.obs_scales.dof_pos, dtype=torch.double
            )  # 12
            # dof_vel
            obs[0, 22 + add : 34 + add] = torch.tensor(
                dq * cfg.normalization.obs_scales.dof_vel, dtype=torch.double
            )  # 12
            # actions
            obs[0, 34 + add : 46 + add] = torch.tensor(action, dtype=torch.double)  # 12
            # f_t
            # print(level)
            f_t = (level + 1) * 0.1 * (
                cfg.commands.ranges.gait_frequency[1]
                - cfg.commands.ranges.gait_frequency[0]
            ) + cfg.commands.ranges.gait_frequency[0]
            # print(f_t)
            obs[0, 46 + add] = f_t
            # l_t
            obs[0, 47 + add] = 0.2
            # h_t
            # obs[0, 48 + add] = 0.
            # psi
            obs[0, 48 + add] = psi

            phy_1 += obs[0, 46 + add] * cfg.sim_config.dt * cfg.sim_config.decimation
            if phy_1 >= 1.0:
                phy_1 = 0.0
            # print(phy_1)
            phy_2 = phy_1 + obs[0, 48 + add]
            if phy_2 >= 1.0:
                phy_2 -= 1.0
            # print(phy_1,phy_2)
            if psi < 0.1:
                phy_1 = 0.25
                phy_2 = 0.25
            # clock_1
            obs[0, 49 + add] = math.sin(2 * math.pi * phy_1)
            # print(obs[0, 49 + add])
            # clock_2
            obs[0, 50 + add] = math.sin(2 * math.pi * phy_2)
            # print(obs[0, 50 + add])
            # print(type(q))
            l = _pin.get_foot_pos(q)
            # print(l)
            # feet_height_base_l
            obs[0, 51 + add] = l[2] + 0.59950981
            # feet_height_base_r
            obs[0, 52 + add] = l[2 + 3] + 0.59950981
            # feet_x_base_l
            obs[0, 53 + add] = l[0]
            # feet_x_base_r
            obs[0, 54 + add] = l[0 + 3]
            obs = torch.clip(
                obs,
                -cfg.normalization.clip_observations,
                cfg.normalization.clip_observations,
            )
            hist_obs.append(obs)
            hist_obs.popleft()

            policy_input = np.zeros(
                [1, int(env.frame_stack * env.num_single_obs)], dtype=np.float32
            )
            for i in range(env.frame_stack):
                policy_input[
                    0, i * env.num_single_obs : (i + 1) * env.num_single_obs
                ] = hist_obs[i][0, :]
            # print(policy_input.shape)

            _action = policy(torch.tensor(policy_input))
            action[:] = _action[0].detach().numpy()

            # inputs = {
            #     policy.get_inputs()[0].name: policy_input,
            #     policy.get_inputs()[1].name: h_in,
            #     policy.get_inputs()[2].name: c_in
            # }
            # outputs = policy.run(None, inputs)
            # _action, h_out, c_out = outputs
            # h_in = h_out
            # c_in = c_out
            # action = _action[0,:]

            # ort_inputs = {policy.get_inputs()[0].name: policy_input}
            # ort_outputs = policy.run(None, ort_inputs)
            # _action = ort_outputs[0]  # ONNX 输出已经是 numpy 数组
            # action[:] = _action[0]

            # obs_history长度为47*5 ，在给入网络之后再更新
            # action[:] = load_policy(logdir,obs,obs_history)[0].detach().numpy()
            # obs_history = torch.cat((obs_history[:,env.obs:], obs[:,:]), dim=-1)

            action = np.clip(
                action,
                -cfg.normalization.clip_actions,
                cfg.normalization.clip_actions,
            )
            target_q = action * cfg.control.action_scale

        target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)
        # print("==============================")
        # print(target_q)
        target_q = np.clip(target_q, cl[:, 0], cl[:, 1])
        # print(target_q)
        # Generate PD control
        tau = pd_control(
            target_q, q, cfg.robot_config.kps, target_dq, dq, cfg.robot_config.kds
        )  # Calc torques
        tau = np.clip(
            tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit
        )  # Clamp torques
        for i in range(6):
            tmptau = tau[i]
            tau[i] = tau[i + 6]
            tau[i + 6] = tmptau
        data.ctrl = tau
        # print(tau)
        joint_angles = [q[0], q[1], q[2], q[3], q[4], q[5]]  # 替换为你实际更新的数据
        joint_velocities = [
            ((dq[0]-q[0])/cfg.sim_config.dt)/dq[0],
            ((dq[1]-q[1])/cfg.sim_config.dt)/dq[1],
            ((dq[2]-q[2])/cfg.sim_config.dt)/dq[2],
            ((dq[3]-q[3])/cfg.sim_config.dt)/dq[3],
            ((dq[4]-q[4])/cfg.sim_config.dt)/dq[4],
            ((dq[5]-q[5])/cfg.sim_config.dt)/dq[5],
        ]  # 替换为你实际更新的数据
        # joint_velocities = [
        #     (dq[0]-q[0])/cfg.sim_config.dt,
        #     (dq[1]-q[1])/cfg.sim_config.dt,
        #     (dq[2]-q[2])/cfg.sim_config.dt,
        #     (dq[3]-q[3])/cfg.sim_config.dt,
        #     (dq[4]-q[4])/cfg.sim_config.dt,
        #     (dq[5]-q[5])/cfg.sim_config.dt,
        # ]  # 替换为你实际更新的数据
        joint_acc = [
            (old_dq[0] - dq[0])/cfg.sim_config.dt,
            (old_dq[1] - dq[1])/cfg.sim_config.dt,
            (old_dq[2] - dq[2])/cfg.sim_config.dt,
            (old_dq[3] - dq[3])/cfg.sim_config.dt,
            (old_dq[4] - dq[4])/cfg.sim_config.dt,
            (old_dq[5] - dq[5])/cfg.sim_config.dt,
        ]
        old_dq = dq
        # 将当前帧的数据追加到列表中
        joint_angles_all_frames.append(joint_angles)
        joint_velocities_all_frames.append(joint_velocities)
        joint_acc_all_frames.append(joint_acc)
        mujoco.mj_step(model, data)
        viewer.render()
        count_lowlevel += 1
    df = pd.DataFrame(
        {
            "Joint Angles Frame 1": [frame[0] for frame in joint_angles_all_frames],
            "Joint Angles Frame 2": [frame[1] for frame in joint_angles_all_frames],
            "Joint Angles Frame 3": [frame[2] for frame in joint_angles_all_frames],
            "Joint Angles Frame 4": [frame[3] for frame in joint_angles_all_frames],
            "Joint Angles Frame 5": [frame[4] for frame in joint_angles_all_frames],
            "Joint Angles Frame 6": [frame[5] for frame in joint_angles_all_frames],
            "Joint Velocities Frame 1": [
                frame[0] for frame in joint_velocities_all_frames
            ],
            "Joint Velocities Frame 2": [
                frame[1] for frame in joint_velocities_all_frames
            ],
            "Joint Velocities Frame 3": [
                frame[2] for frame in joint_velocities_all_frames
            ],
            "Joint Velocities Frame 4": [
                frame[3] for frame in joint_velocities_all_frames
            ],
            "Joint Velocities Frame 5": [
                frame[4] for frame in joint_velocities_all_frames
            ],
            "Joint Velocities Frame 6": [
                frame[5] for frame in joint_velocities_all_frames
            ],
            "Joint acc Frame 1": [
                frame[0] for frame in joint_acc_all_frames
            ],
            "Joint acc Frame 2": [
                frame[1] for frame in joint_acc_all_frames
            ],
            "Joint acc Frame 3": [
                frame[2] for frame in joint_acc_all_frames
            ],
            "Joint acc Frame 4": [
                frame[3] for frame in joint_acc_all_frames
            ],
            "Joint acc Frame 5": [
                frame[4] for frame in joint_acc_all_frames
            ],
            "Joint acc Frame 6": [
                frame[5] for frame in joint_acc_all_frames
            ],
        }
    )

    df.to_csv("multiple_frames_joint_data.csv", index=False)
    df = pd.read_csv("multiple_frames_joint_data.csv")

    # 获取数据
    time = df.index  # 假设每一行代表一个时间点

    # 获取角度、角速度和角加速度的数据
    joint_angles = df[["Joint Angles Frame 1", "Joint Angles Frame 2", "Joint Angles Frame 3",
                    "Joint Angles Frame 4", "Joint Angles Frame 5", "Joint Angles Frame 6"]]
    joint_velocities = df[["Joint Velocities Frame 1", "Joint Velocities Frame 2", "Joint Velocities Frame 3",
                        "Joint Velocities Frame 4", "Joint Velocities Frame 5", "Joint Velocities Frame 6"]]
    joint_acc = df[["Joint acc Frame 1", "Joint acc Frame 2", "Joint acc Frame 3",
                    "Joint acc Frame 4", "Joint acc Frame 5", "Joint acc Frame 6"]]
    # 创建三个窗口，每个窗口显示6个电机的图形
    fig1, axes1 = plt.subplots(3, 2, figsize=(10, 12))
    fig2, axes2 = plt.subplots(3, 2, figsize=(10, 12))
    fig3, axes3 = plt.subplots(3, 2, figsize=(10, 12))

    # 绘制角度图（第一个窗口）
    fig1.suptitle("Joint Angles Over Time")
    for i in range(6):
        row, col = divmod(i, 2)  # 确定该电机图表的行列位置
        axes1[row, col].plot(time, joint_angles.iloc[:, i], label=f"Motor {i+1}")
        axes1[row, col].set_title(f"Motor {i+1} - Joint Angle")
        axes1[row, col].set_xlabel("Time")
        axes1[row, col].set_ylabel("Angle (rad)")
        axes1[row, col].legend()

    # 绘制角速度图（第二个窗口）
    fig2.suptitle("Joint Velocities Over Time")
    for i in range(6):
        row, col = divmod(i, 2)
        axes2[row, col].plot(time, joint_velocities.iloc[:, i], label=f"Motor {i+1}")
        axes2[row, col].set_title(f"Motor {i+1} - Joint Velocity")
        axes2[row, col].set_xlabel("Time")
        axes2[row, col].set_ylabel("Velocity (rad/s)")
        axes2[row, col].legend()

    # 绘制角加速度图（第三个窗口）
    fig3.suptitle("Joint Accelerations Over Time")
    for i in range(6):
        row, col = divmod(i, 2)
        axes3[row, col].plot(time, joint_acc.iloc[:, i], label=f"Motor {i+1}")
        axes3[row, col].set_title(f"Motor {i+1} - Joint Acceleration")
        axes3[row, col].set_xlabel("Time")
        axes3[row, col].set_ylabel("Acceleration (rad/s²)")
        axes3[row, col].legend()

    # 显示所有的图形
    plt.tight_layout()
    fig1.subplots_adjust(top=0.9)  # Adjust the space for title
    fig2.subplots_adjust(top=0.9)
    fig3.subplots_adjust(top=0.9)
    plt.show()
    viewer.close()


class GamepadHandler:
    def __init__(self):
        # pygame.init()
        pygame.joystick.init()

        self.joystick_count = pygame.joystick.get_count()
        self.joysticks = []

        for i in range(self.joystick_count):
            joystick = pygame.joystick.Joystick(i)
            joystick.init()
            self.joysticks.append(joystick)
            print(f"Initialized Joystick {i}: {joystick.get_name()}")

    def process_events(self, command_queue):
        for event in pygame.event.get():
            if event.type == pygame.JOYAXISMOTION:
                axis_values = [
                    self.joysticks[0].get_axis(1),
                    self.joysticks[0].get_axis(0),
                    self.joysticks[0].get_axis(3),
                ]
                while not command_queue.empty():
                    command_queue.get_nowait()
                command_queue.put(axis_values)
            elif event.type == pygame.QUIT:
                return False
        return True

    def quit(self):
        pygame.quit()


def gamepad_input(control_queue):
    handler = GamepadHandler()
    running = True
    while running:
        running = handler.process_events(control_queue)
        time.sleep(0.00001)  # Adjust polling rate if necessary
    handler.quit()


class KeyboardHandler:
    def __init__(self):
        # 如果你想和手柄共用同一个 pygame 实例，请注释掉或者放到外面
        # 但要注意避免重复 init() 导致冲突。
        ...

    def process_events(self, command_queue):
        """
        轮询键盘事件，并将转换后的 [vx, vy, dyaw] 放到 command_queue。
        这里仅做示例，你可根据需要调整控制按键或速度方向。
        """
        # 获取所有事件
        for event in pygame.event.get():
            # 如果窗口关闭按钮被点击，或者其它事件，需要决定是否退出
            if event.type == pygame.QUIT:
                return False

        # 同时也可以直接用 pygame.key.get_pressed() 来检查哪些键被按下
        keys = pygame.key.get_pressed()

        # 下面给一个简单的 WSAD + QE 控制逻辑
        # 假设 vx, vy, dyaw 的取值范围在 [-1, 1] 之间
        vx = 0.0
        vy = 0.0
        dyaw = 0.0
        level = 0
        # W/S 控制前后
        # print(keys)
        for i in range(10):
            if keys[eval(f"pygame.K_{i}") if hasattr(pygame, f"K_{i}") else ...]:
                level = i
        if keys[pygame.K_w]:
            vx = -1.0
        if keys[pygame.K_s]:
            vx = 1.0

        # A/D 控制左右 (此处假设 A 为左, D 为右)
        if keys[pygame.K_a]:
            vy = 1.0
        if keys[pygame.K_d]:
            vy = -1.0

        # Q/E 控制 yaw
        if keys[pygame.K_q]:
            dyaw = -1.0
        if keys[pygame.K_e]:
            dyaw = 1.0

        # 如果需要给这个轴赋某种灵敏度或者死区，可以在这里做
        # 在放入队列之前，可以清空一下队列，确保只有最新命令
        # if abs(vx) > 1e-3 or abs(vy) > 1e-3 or abs(dyaw) > 1e-3:
        while not command_queue.empty():
            command_queue.get_nowait()
        command_queue.put([vx, vy, dyaw, level])

        return True

    def quit(self):
        pygame.quit()


def keyboard_input(control_queue):
    """
    该函数在单独的线程中运行，持续调用 KeyboardHandler 处理键盘事件。
    """
    handler = KeyboardHandler()
    running = True
    while running:
        running = handler.process_events(control_queue)
        time.sleep(0.01)  # 根据自己需要调整采样频率
    handler.quit()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Deployment script.")
    parser.add_argument(
        "--logdir",
        type=str,
        required=False,
        default=f"{LEGGED_GYM_ROOT_DIR}/logs/hicl_hug_ActorCriticRecurrent/exported/policies",
        help="Run to load from.",
    )
    parser.add_argument("--terrain", action="store_true", help="terrain or plane")
    args = parser.parse_args()

    class Sim2simCfg(cfg):
        class sim_config:
            # mujoco_model_path = f"{LEGGED_GYM_ROOT_DIR}/resources/robots/hi_cl_23_240925/mjcf/hi_12dof_release_v2.xml"  # 平地
            mujoco_model_path = f"{LEGGED_GYM_ROOT_DIR}/resources/robots/hi_cl_23_240925/mjcf/hi_12dof_release_v2.xml"  # 平地

            sim_duration = 20.0
            dt = 0.001
            decimation = 10

        class robot_config:
            kps_l = []
            kds_l = []
            for joint, vals in cfg.control.stiffness.items():
                print(f"joint: {joint} vals: {vals}")
                kps_l.append(vals)

            for joint, vals in cfg.control.damping.items():
                print(f"joint: {joint} vals: {vals}")
                kds_l.append(vals)
            kps = np.array(kps_l * (2), dtype=np.float32)
            kds = np.array(kds_l * (2), dtype=np.float32)
            # print("kps: ", kps)
            # print("kds: ", kds)
            # kps = np.array(
            #     [80,40,40,80,50,10]*(2),
            #     dtype=np.double
            # )  # v7

            # kds = np.array(
            #     [5,5,1,5,0.4,0.4]*(2),
            #     dtype=np.double,
            # )
            print("kps: ", kps)
            print("kds: ", kds)
            tau_limit = 40.0 * np.ones(12, dtype=np.double)

    a = args.logdir + "/policy.pt"
    policy = torch.jit.load(a)

    # b = args.logdir + "/policy.onnx"
    # policy = ort.InferenceSession(b)

    control_queue = queue.Queue()
    pygame.init()
    screen = pygame.display.set_mode((100, 100))
    # Thread for MuJoCo simulation
    thread_a = threading.Thread(
        target=run_mujoco, args=(control_queue, policy, Sim2simCfg())
    )
    thread_a.start()

    # Thread for gamepad input
    # thread_b = threading.Thread(target=gamepad_input, args=(control_queue,))
    # thread_b.start()

    thread_keyboard = threading.Thread(target=keyboard_input, args=(control_queue,))
    thread_keyboard.start()

    # Wait for threads to complete
    thread_a.join()
    # thread_b.join()
    thread_keyboard.join()

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
from legged_gym.envs.hi_hug.hi_config_hug import HiHugCfg as cfg

# from legged_gym.envs.pai.pai_config_demo import PaiDemoRoughCfg
from isaacgym.torch_utils import *

import torch

import csv
import pandas as pd
import threading
import queue
import pygame
import time


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
                vx, vy, dyaw = [process_value(val) for val in axis_values]
                command = [
                    -vx * cfg.commands.ranges.lin_vel_x[1],
                    -vy * cfg.commands.ranges.lin_vel_y[1],
                    -dyaw * cfg.commands.ranges.ang_vel_yaw[1],
                ]
                print(f"Received control input: vx={vx}, vy={vy}, dyaw={dyaw}")
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
            obs[0, 46 + add] = 0.2
            obs[0, 47 + add] = 0.2
            # obs[0, 48 + add] = 0.
            obs[0, 48 + add] = psi
            phy_1 += obs[0, 46 + add] * cfg.sim_config.dt
            if phy_1 >= 1.0:
                phy_1 = 0.0
            phy_2 = phy_1 + obs[0, 48 + add]
            if phy_2 >= 1.0:
                phy_2 -= 1.0
            if psi < 0.1:
                phy_1 = 0.25
                phy_2 = 0.25
            obs[0, 49 + add] = math.sin(
                    2
                    * math.pi
                    * phy_1
                )
            obs[0, 50 + add] = math.sin(
                    2
                    * math.pi
                    * phy_2
                )

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
            # _action,mean_vel,mean_latent = policy(torch.tensor(policy_input))
            # print("action:\n",_action)
            action[:] = _action[0].detach().numpy()
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

        mujoco.mj_step(model, data)
        viewer.render()
        count_lowlevel += 1

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
        # W/S 控制前后
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
        command_queue.put([vx, vy, dyaw])

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
        default=f"{LEGGED_GYM_ROOT_DIR}/logs/hi_hug_ActorCriticLSTM/exported/policies",
        help="Run to load from.",
    )
    parser.add_argument("--terrain", action="store_true", help="terrain or plane")
    args = parser.parse_args()

    class Sim2simCfg(cfg):
        class sim_config:
            # mujoco_model_path = f"{LEGGED_GYM_ROOT_DIR}/resources/robots/hi_cl_23_240925/mjcf/hi_12dof_release_v2.xml"  # 平地
            mujoco_model_path = f"{LEGGED_GYM_ROOT_DIR}/resources/robots/hi_12dof_250108_4/mjcf/hi_12dof_release_v2.xml"  # 平地

            sim_duration = 60.0
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

    a = args.logdir + "/lstm_1.pt"
    policy = torch.jit.load(a)
    # run_mujoco(policy, Sim2simCfg())

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

    thread_keyboard = threading.Thread(
        target=keyboard_input, 
        args=(control_queue,)
    )
    thread_keyboard.start()


    # Wait for threads to complete
    thread_a.join()
    # thread_b.join()
    thread_keyboard.join()

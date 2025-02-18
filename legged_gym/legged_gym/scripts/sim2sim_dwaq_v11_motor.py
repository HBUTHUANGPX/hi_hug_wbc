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
from legged_gym.envs.pai.pai_config_demo import PaiDemoRoughCfg
import torch

import csv
import pandas as pd

import matplotlib.pyplot as plt
import time
import cv2
import threading
import glfw
import matplotlib.animation as animation
from typing import List


class cmd:
    vx = 0.0
    vy = 0.0
    dyaw = 0.0


def quat_rotate_inverse(q, v):
    q_w = q[-1]
    q_vec = q[:3]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c


class env:
    obs = 47
    num_single_obs = 47
    frame_stack = 6
    obs_his = 47 * 5


class plot_line:
    ax: plt.Axes

    def __init__(self, ax, style="r-"):
        self.x = []
        self.y = []
        self.ax = ax
        (self.line,) = self.ax.plot(self.x, self.y, style)

    def update_xy(self, new_x, new_y):
        self.x.append(new_x)
        self.y.append(new_y)
        self.line.set_xdata(self.x)
        self.line.set_ydata(self.y)


class plot_one_figure:
    def __init__(self):
        self.init_figure()

    def init_figure(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel("time")
        self.ax.set_ylabel("value")
        self.ax.set_title("plot")
        self.ax.grid(True)
        self.true_v_ = plot_line(self.ax, "r-")
        self.net_v_ = plot_line(self.ax, "b-")
        self.derta_v_ = plot_line(self.ax, "g-")

    def update_xy(self, x, true_vel, net_vel):
        # self.true_v_.update_xy(x, true_vel)
        # self.net_v_.update_xy(x, net_vel)
        self.derta_v_.update_xy(x, (true_vel - net_vel) / 10)
        self.ax.relim()  # 重新计算坐标轴范围
        self.ax.autoscale_view()  # 自动缩放视图

    def update_xy2(self, x, pos, vel, tau):
        self.true_v_.update_xy(x, pos)
        self.net_v_.update_xy(x, vel)
        self.derta_v_.update_xy(x, tau)
        self.ax.relim()  # 重新计算坐标轴范围
        self.ax.autoscale_view()  # 自动缩放视图


class mujoco_visual:
    def __init__(self) -> None:
        self.count_lowlevel = 0
        self.close = 1
        self.mujoco_close = 1
        self.stop_event = threading.Event()
        self.vel = [0, 0, 0]
        self.w = [0, 0, 0]
        self.net_vel = [0, 0, 0]
        self.q = [0] * (12)
        self.tq = [0] * (12)
        self.dq = [0] * (12)
        self.plot_index = 4

    def quaternion_to_euler_array(self, quat):
        # Ensure quaternion is in the correct format [x, y, z, w]
        x, y, z, w = quat

        # Roll (x-axis rotation)
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = np.arctan2(t0, t1)

        # Pitch (y-axis rotation)
        t2 = +2.0 * (w * y - z * x)
        t2 = np.clip(t2, -1.0, 1.0)
        pitch_y = np.arcsin(t2)

        # Yaw (z-axis rotation)
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = np.arctan2(t3, t4)

        # Returns roll, pitch, yaw in a NumPy array in radians
        return np.array([roll_x, pitch_y, yaw_z])

    def get_obs(self, data):
        """Extracts an observation from the mujoco data structure"""
        q = data.qpos.astype(np.double)
        dq = data.qvel.astype(np.double)
        quat = data.sensor("orientation").data[[1, 2, 3, 0]].astype(np.double)
        r = R.from_quat(quat)
        v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
        self.vel = [v[0], v[1], v[2]]
        omega = data.sensor("angular-velocity").data.astype(np.double)
        self.w = omega
        gvec = r.apply(np.array([0.0, 0.0, -1.0]), inverse=True).astype(np.double)
        return (q, dq, quat, v, omega, gvec)

    def pd_control(self, target_q, q, kp, target_dq, dq, kd):
        """Calculates torques from position commands"""
        # print("p:", (target_q - q) * kp )
        # print("d", (target_dq - dq) * kd)
        return (target_q - q) * kp + (target_dq - dq) * kd

    def plot_thread(self):
        plt.ion()  # 打开交互模式
        pof = []
        num_figures = 3  # 假设需要三个画板
        for i in range(num_figures):
            aa = plot_one_figure()
            pof.append(aa)
        while not self.stop_event.is_set():
            for i, _pof in enumerate(pof):
                _pof.update_xy(
                    self.count_lowlevel * 0.001, self.vel[i], self.net_vel[i]
                )
            plt.draw()  # 绘制更新
            plt.pause(0.001)
        for i, _pof in enumerate(pof):
            _pof.fig.savefig(f"sine_wave_{i}.png")  # 保存为 PNG 格式
        plt.ioff()
        plt.close("all")

    def plot_joint_thread(self):
        plt.ion()  # 打开交互模式
        pof: List[plot_one_figure] = []
        num_figures = 1  # 假设需要三个画板
        for i in range(num_figures):
            aa = plot_one_figure()
            pof.append(aa)
        while not self.stop_event.is_set():
            for i, _pof in enumerate(pof):
                _pof.update_xy2(
                    self.count_lowlevel * 0.001,
                    self.q[self.plot_index],
                    self.tq[self.plot_index],
                    0,
                    # self.dq[self.plot_index],
                )

                # _pof.update_xy(
                #     self.count_lowlevel * 0.001, self.q[0], self.tq[0]
                # )
                # print("plot==========:\n")
                # print(self.q[4], self.tq[4], self.dq[4])
            plt.draw()  # 绘制更新
            plt.pause(0.001)
        for i, _pof in enumerate(pof):
            _pof.fig.savefig(f"sine_wave_{i}.png")  # 保存为 PNG 格式
        plt.ioff()
        plt.close("all")

    def run_mujoco(self, policy, cfg):
        model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
        model.opt.timestep = cfg.sim_config.dt
        data = mujoco.MjData(model)
        mujoco.mj_step(model, data)
        viewer = mujoco_viewer.MujocoViewer(model, data)
        self.window = viewer.window
        target_q = np.zeros((cfg.env.num_actions), dtype=np.double)
        action = np.zeros((cfg.env.num_actions), dtype=np.double)
        flip = 1
        hist_obs = deque()
        for _ in range(env.frame_stack):
            hist_obs.append(np.zeros([1, env.num_single_obs], dtype=np.double))

        for _ in tqdm(
            range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)),
            desc="Simulating...",
        ):
            if glfw.window_should_close(self.window):
                print("=============out mujoco==========")
                break
            # Obtain an observation
            q, dq, quat, v, omega, gvec = self.get_obs(data)
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
            self.q = q
            self.dq = dq

            if self.count_lowlevel % cfg.sim_config.decimation == 0:
                obs = torch.zeros(1, env.obs, dtype=torch.float)
                _q = quat
                _v = np.array([0.0, 0.0, -1.0])
                projected_gravity = quat_rotate_inverse(_q, _v)

                obs[0, :3] = torch.tensor(
                    omega * cfg.normalization.obs_scales.ang_vel, dtype=torch.double
                )  # 3
                obs[0, 3:6] = torch.tensor(projected_gravity, dtype=torch.double)  # 3
                obs[0, 6] = torch.tensor(
                    cmd.vx * cfg.normalization.obs_scales.lin_vel, dtype=torch.double
                )
                obs[0, 7] = torch.tensor(
                    cmd.vy * cfg.normalization.obs_scales.lin_vel, dtype=torch.double
                )
                obs[0, 8] = torch.tensor(
                    cmd.dyaw * cfg.normalization.obs_scales.ang_vel, dtype=torch.double
                )
                obs[0, 9:21] = torch.tensor(
                    q * cfg.normalization.obs_scales.dof_pos, dtype=torch.double
                )  # 12
                obs[0, 21:33] = torch.tensor(
                    dq * cfg.normalization.obs_scales.dof_vel, dtype=torch.double
                )  # 12
                obs[0, 33:45] = torch.tensor(action, dtype=torch.double)  # 12
                obs[0, 45] = math.sin(
                    2
                    * math.pi
                    * self.count_lowlevel
                    * cfg.sim_config.dt
                    / cfg.rewards.cycle_time
                )
                obs[0, 46] = math.cos(
                    2
                    * math.pi
                    * self.count_lowlevel
                    * cfg.sim_config.dt
                    / cfg.rewards.cycle_time
                )

                obs = np.clip(
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
                _action = policy(torch.tensor(policy_input))
                # _action, _mean_vel = policy(torch.tensor(policy_input))
                # self.net_vel = _mean_vel[0].detach().numpy()
                action[:] = _action[0].detach().numpy()
                action = np.clip(
                    action,
                    -cfg.normalization.clip_actions,
                    cfg.normalization.clip_actions,
                )
                target_q = action * cfg.control.action_scale
            target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)
            target_q = np.zeros((cfg.env.num_actions), dtype=np.double)
            if self.count_lowlevel % (2 * 1000) == 0:
                print("==========" + str(self.count_lowlevel) + "===========")
                flip = -flip
            target_q[self.plot_index] = flip * 0.3
            self.tq = target_q
            # Generate PD control
            tau = self.pd_control(
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

            mujoco.mj_step(model, data)
            viewer.render()
            self.count_lowlevel += 1
        self.stop_event.set()
        viewer.close()


if __name__ == "__main__":
    import argparse

    print(LEGGED_GYM_ROOT_DIR)
    parser = argparse.ArgumentParser(description="Deployment script.")
    parser.add_argument(
        "--load_model",
        type=str,
        required=False,
        help="Run to load from.",
        default=f"{LEGGED_GYM_ROOT_DIR}/logs/pai_demo/exported/policies",
    )
    parser.add_argument("--terrain", action="store_true", help="terrain or plane")
    args = parser.parse_args()

    class Sim2simCfg(PaiDemoRoughCfg):

        class sim_config:
            mujoco_model_path = f"{LEGGED_GYM_ROOT_DIR}/resources/robots/pi_12dof_release_v1/mjcf/pi_12dof_release_v2_fixedbase_2.xml"
            sim_duration = 60.0
            dt = 0.001
            decimation = 10

        class robot_config:
            kps_l = []
            kds_l = []
            for joint, vals in PaiDemoRoughCfg.control.stiffness.items():
                print(f"joint: {joint} vals: {vals}")
                kps_l.append(vals)

            for joint, vals in PaiDemoRoughCfg.control.damping.items():
                print(f"joint: {joint} vals: {vals}")
                kds_l.append(vals)
            kps = np.array([60.0, 40.0, 20.0, 60.0, 30.0, 10.0] * (2), dtype=np.float32)
            kds = np.array([5.2, 3.4, 0.8, 3.2, 0.8, 0.3] * (2), dtype=np.float32)
            tau_limit = 40.0 * np.ones(12, dtype=np.double)

    policy = torch.jit.load(args.load_model + "/combined_model_dwaq.pt")
    a = mujoco_visual()
    matplotlib_thread = threading.Thread(target=a.plot_joint_thread)
    mujoco_thread = threading.Thread(target=a.run_mujoco, args=(policy, Sim2simCfg()))
    matplotlib_thread.start()
    mujoco_thread.start()
    matplotlib_thread.join()
    mujoco_thread.join()
    print("Both threads have finished. Main thread will now terminate.")

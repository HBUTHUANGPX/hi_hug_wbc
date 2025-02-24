# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger
from legged_gym.utils.helpers import (
    export_policy_as_jit_actor,
    export_policy_as_jit_encoder,
    class_to_dict,
    export_lstm_model,
)
from legged_gym.envs.hicl_hug.hicl_env_hug import HiclHugEnv as rb_env
import numpy as np
import torch
import pickle

# import matplotlib
# matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import threading
import queue
import time
from multiprocessing import Process, Value

data_queue = queue.Queue()
plot_num = 4


def plot_data(data_queue):
    global plot_num
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


def play(args):  # dwaq
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    class_to_dict(env_cfg)
    class_to_dict(train_cfg)

    with open("env_cfg.pkl", "wb") as f:
        pickle.dump(class_to_dict(env_cfg), f)
    with open("train_cfg.pkl", "wb") as f:
        pickle.dump(train_cfg, f)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_robots = True
    # env_cfg.asset.fix_base_link = False#True
    # env_cfg.asset.fix_base_link = True
    env_cfg.init_state.pos = [0.0, 0.0, 0.648]
    env_cfg.sim.physx.num_threads = 12
    # env_cfg.control.stiffness ={
    #         "hip_pitch_joint": 4000.0,
    #         "hip_roll_joint": 2000.0,
    #         "thigh_joint": 2000.0,
    #         "calf_joint": 4000.0,
    #         "ankle_pitch_joint": 2000,
    #         "ankle_roll_joint": 2000,
    #     }
    # env_cfg.control.damping = {
    #         "hip_pitch_joint": 50,
    #         "hip_roll_joint": 80,
    #         "thigh_joint": 80,
    #         "calf_joint": 50,
    #         "ankle_pitch_joint": 50,
    #         "ankle_roll_joint": 50,
    #     }
    # prepare environment
    env: rb_env
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs, obs_hist = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = ppo_runner.get_inference_policy(device=env.device)

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(
            LEGGED_GYM_ROOT_DIR,
            "logs",
            train_cfg.runner.experiment_name,
            "exported",
            "policies",
        )
        print("Exported policy as jit script to: ", path)
        print("policy ", policy)
        # export_lstm_model(ppo_runner.alg.actor_critic,path,48,0)

    # 启动子线程进行绘图
    plot_thread = threading.Thread(target=plot_data, args=(data_queue,))
    plot_thread.daemon = True
    plot_thread.start()
    global plot_num
    for i in range(10 * int(env.max_episode_length)):
        actions = policy(obs.detach())

        # actions = policy(obs.detach())
        # actions = policy(obs.detach())
        # actions = policy(obs.detach(), obs_hist.detach())
        # actions[:,0:1] -= 1*lt
        # actions[:,3:4] += 2*lt
        # actions[:,4:5] -= 1*lt
        # actions[:,6:7] -= 4*lt
        # actions[:,9:10] += 8*lt
        # actions[:,10:11] -= 4*lt
        obs, _, _, obs_hist, rews, dones, infos = env.step(actions.detach())

        # fh,lt,err,cdf,nmerr,rew=env._reward_foot_swing_track(play=True)
        # merged_tensor = torch.cat([
        #     fh,
        #     lt,
        #     err.unsqueeze(1),
        #     nmerr.unsqueeze(1),
        #     rew.unsqueeze(1),
        #     ], dim=1)[0,:]
        # plot_num = 5

        # scm,cnt,cf,fat,atr,r = env._reward_feet_airtime(play=True)
        # merged_tensor = torch.cat([
        #     scm.unsqueeze(1),
        #     cnt[:,0:1],
        #     cf[:,0:1],
        #     fat[:,0:1],
        #     atr.unsqueeze(1),
        #     r.unsqueeze(1)
        #     ], dim=1)[0,:]
        merged_tensor = torch.cat([obs[:, 46:47], obs[:, 48:49], obs[:, 49:50],env.phy_1_bar], dim=1)[
            0, :
        ]
        data_queue.put(merged_tensor)


if __name__ == "__main__":
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)

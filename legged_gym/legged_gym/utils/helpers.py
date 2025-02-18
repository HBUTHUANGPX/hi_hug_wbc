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

import os
import copy
import torch
import numpy as np
import random
from isaacgym import gymapi
from isaacgym import gymutil

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR


def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


def update_class_from_dict(obj, dict):
    for key, val in dict.items():
        attr = getattr(obj, key, None)
        if isinstance(attr, type):
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)
    return


def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_sim_params(args, cfg):
    # code from Isaac Gym Preview 2
    # initialize sim params
    sim_params = gymapi.SimParams()

    # set some values from args
    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params


def get_load_path(root, load_run=-1, checkpoint=-1):
    try:
        runs = os.listdir(root)
        # TODO sort by date to handle change of month
        runs.sort()
        if "exported" in runs:
            runs.remove("exported")
        last_run = os.path.join(root, runs[-1])
    except:
        raise ValueError("No runs in this directory: " + root)
    if load_run == -1:
        load_run = last_run
    else:
        load_run = os.path.join(root, load_run)

    if checkpoint == -1:
        models = [file for file in os.listdir(load_run) if "model" in file]
        models.sort(key=lambda m: "{0:0>15}".format(m))
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint)

    load_path = os.path.join(load_run, model)
    return load_path


def update_cfg_from_args(env_cfg, cfg_train, args):
    # seed
    if env_cfg is not None:
        # num envs
        if args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs
    if cfg_train is not None:
        if args.seed is not None:
            cfg_train.seed = args.seed
        # alg runner parameters
        if args.max_iterations is not None:
            cfg_train.runner.max_iterations = args.max_iterations
        if args.resume:
            cfg_train.runner.resume = args.resume
        if args.experiment_name is not None:
            cfg_train.runner.experiment_name = args.experiment_name
        if args.run_name is not None:
            cfg_train.runner.run_name = args.run_name
        if args.load_run is not None:
            cfg_train.runner.load_run = args.load_run
        if args.checkpoint is not None:
            cfg_train.runner.checkpoint = args.checkpoint

    return env_cfg, cfg_train


def get_args():
    custom_parameters = [
        {
            "name": "--task",
            "type": str,
            "default": "a1",
            "help": "Resume training or start testing from a checkpoint. Overrides config file if provided.",
        },
        {
            "name": "--resume",
            "action": "store_true",
            "default": False,
            "help": "Resume training from a checkpoint",
        },
        {
            "name": "--experiment_name",
            "type": str,
            "help": "Name of the experiment to run or load. Overrides config file if provided.",
        },
        {
            "name": "--run_name",
            "type": str,
            "help": "Name of the run. Overrides config file if provided.",
        },
        {
            "name": "--load_run",
            "type": str,
            "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided.",
        },
        {
            "name": "--checkpoint",
            "type": int,
            "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided.",
        },
        {
            "name": "--headless",
            "action": "store_true",
            "default": False,
            "help": "Force display off at all times",
        },
        {
            "name": "--horovod",
            "action": "store_true",
            "default": False,
            "help": "Use horovod for multi-gpu training",
        },
        {
            "name": "--rl_device",
            "type": str,
            "default": "cuda:0",
            "help": "Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)",
        },
        {
            "name": "--num_envs",
            "type": int,
            "help": "Number of environments to create. Overrides config file if provided.",
        },
        {
            "name": "--seed",
            "type": int,
            "help": "Random seed. Overrides config file if provided.",
        },
        {
            "name": "--max_iterations",
            "type": int,
            "help": "Maximum number of training iterations. Overrides config file if provided.",
        },
    ]
    # parse arguments
    args = gymutil.parse_arguments(
        description="RL Policy", custom_parameters=custom_parameters
    )

    # name allignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.sim_device_id}"
    return args


def export_policy_as_jit(actor_critic, path):
    if hasattr(actor_critic, "memory_a"):
        # assumes LSTM: TODO add GRU
        exporter = PolicyExporterLSTM(actor_critic)
        exporter.export(path)
    else:
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, "policy_1.pt")
        model = copy.deepcopy(actor_critic.actor).to("cpu")
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(path)


def export_policy_as_jit_actor(actor_critic, path):
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, "actor_dwaq.pt")
    model = copy.deepcopy(actor_critic.actor).to("cpu")
    print("policy model", model)
    traced_script_module = torch.jit.script(model)
    traced_script_module.save(path)


def export_policy_as_jit_encoder(actor_critic, path):
    os.makedirs(path, exist_ok=True)
    path1 = os.path.join(path, "encoder_dwaq.pt")
    model = copy.deepcopy(actor_critic.encoder).to("cpu")
    print("encoder model", model)
    traced_script_module = torch.jit.script(model)
    traced_script_module.save(path1)

    path2 = os.path.join(path, "latent_mu_dwaq.pt")
    model = copy.deepcopy(actor_critic.encode_mean_latent).to("cpu")
    print("latent mu model", model)
    traced_script_module = torch.jit.script(model)
    traced_script_module.save(path2)

    path3 = os.path.join(path, "latent_var_dwaq.pt")
    model = copy.deepcopy(actor_critic.encode_logvar_latent).to("cpu")
    print("latent var model", model)
    traced_script_module = torch.jit.script(model)
    traced_script_module.save(path3)

    path4 = os.path.join(path, "vel_mu_dwaq.pt")
    model = copy.deepcopy(actor_critic.encode_mean_vel).to("cpu")
    print("vel mu model", model)
    traced_script_module = torch.jit.script(model)
    traced_script_module.save(path4)

    path5 = os.path.join(path, "vel_var_dwaq.pt")
    model = copy.deepcopy(actor_critic.encode_logvar_vel).to("cpu")
    print("vel var model", model)
    traced_script_module = torch.jit.script(model)
    traced_script_module.save(path5)

def export_policy_as_jit(actor_critic, path):
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, "policy_1.pt")
    model = copy.deepcopy(actor_critic.actor).to("cpu")
    traced_script_module = torch.jit.script(model)
    traced_script_module.save(path)
class ACModel(torch.nn.Module):
    def __init__(self, actor_critic,num_observations = 48):
        super().__init__()
        self.actor = actor_critic.actor
        self.num_observations = num_observations
    def forward(self, input):
        obs = input
        action = self.actor(obs)

        return action#,mean_vel,mean_latent


import torch
import torch.nn as nn
import torch.jit
import os

class PolicyExporterLSTM(nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        
        # 复制 Actor 网络的 LSTM 和 MLP 部分
        self.actor_lstm = copy.deepcopy(actor_critic.actor_rnn.lstm)
        self.actor_mlp = copy.deepcopy(actor_critic.actor_mlp)
        
        # 注册 LSTM 的隐藏状态和细胞状态为 buffer
        self.register_buffer(
            "hidden_state",
            torch.zeros(self.actor_lstm.num_layers, 1, self.actor_lstm.hidden_size),
        )
        self.register_buffer(
            "cell_state",
            torch.zeros(self.actor_lstm.num_layers, 1, self.actor_lstm.hidden_size),
        )

    def forward(self, x):
        # 处理输入序列
        out, (h, c) = self.actor_lstm(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        # 更新隐藏状态和细胞状态
        self.hidden_state[:] = h
        self.cell_state[:] = c
        # 通过 MLP 生成动作
        return self.actor_mlp(out.squeeze(0))

    @torch.jit.export
    def reset_memory(self):
        # 重置隐藏状态和细胞状态
        self.hidden_state[:] = 0.0
        self.cell_state[:] = 0.0

    def export(self, path):
        # 确保路径存在
        os.makedirs(path, exist_ok=True)
        # 保存模型
        path = os.path.join(path, "policy_lstm_1.pt")
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)

# 示例使用
# actor_critic = ActorCriticLSTM(num_actor_obs=..., num_critic_obs=..., num_actions=...)
# exporter = PolicyExporterLSTM(actor_critic)
# exporter.export("path_to_save_model")

def export_lstm_model(
    actor_critic: torch.nn.Module, path, num_observations=48, num_obs_hist=15
):
    os.makedirs(path, exist_ok=True)
    import copy

    ac = copy.deepcopy(actor_critic)
    model = PolicyExporterLSTM(ac).to("cpu")
    traced_script_module = torch.jit.script(model)
    traced_script_module.save(os.path.join(path, "lstm_1.pt"))
    # 假定模型的输入尺寸，这需要根据您的模型输入进行调整
    # 例如：假设模型输入是一个大小为(1, input_size)的张量
    dummy_input = torch.randn(
        1, (num_obs_hist + 1) * num_observations
    )  # 请根据您的模型调整 input_size

    # 导出为 ONNX
    onnx_path = os.path.join(path, "lstm_1.onnx")
    torch.onnx.export(
        model,  # 要导出的模型
        dummy_input,  # 模型输入的虚拟张量
        onnx_path,  # 导出的 ONNX 文件路径
        export_params=True,  # 是否导出模型参数
        opset_version=11,  # ONNX opset 版本
        do_constant_folding=True,  # 是否执行常量折叠优化
        input_names=["input"],  # 输入节点的名称
        output_names=["output"],  # 输出节点的名称
    )
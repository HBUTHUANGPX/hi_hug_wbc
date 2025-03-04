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

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
from .actor_critic import ActorCritic, get_activation
from rsl_rl.utils import unpad_trajectories
class ActorCriticRecurrentPAE(ActorCritic):
    is_recurrent = True
    def __init__(self,  num_actor_obs,#当前帧数据
                        num_critic_obs,
                        num_actions,
                        paenet_in_dim,#历史帧累积数据
                        paenet_out_dim=19,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        rnn_type='lstm',
                        rnn_hidden_size=256,
                        rnn_num_layers=1,
                        init_noise_std=1.0,
                        **kwargs):
        if kwargs:
            print("ActorCriticRecurrent.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),)

        super().__init__(num_actor_obs=rnn_hidden_size,
                         num_critic_obs=rnn_hidden_size,
                         num_actions=num_actions,
                         actor_hidden_dims=actor_hidden_dims,
                         critic_hidden_dims=critic_hidden_dims,
                         activation=activation,
                         init_noise_std=init_noise_std)

        activation = get_activation(activation)

        self.memory_a = Memory(num_actor_obs + 19, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)
        self.memory_c = Memory(num_critic_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)

        self.encoder = nn.Sequential(
            nn.Linear(paenet_in_dim, 128),
            activation,
            nn.Linear(128, 64),
            activation,
        )
        self.encode_mean_latent = nn.Linear(64, paenet_out_dim - 3)
        self.encode_logvar_latent = nn.Linear(64, paenet_out_dim - 3)
        self.encode_mean_vel = nn.Linear(64, 3)
        self.encode_logvar_vel = nn.Linear(64, 3)
        self.decoder = nn.Sequential(
            nn.Linear(paenet_out_dim, 64),
            activation,
            nn.Linear(64, 128),
            activation,
            nn.Linear(128, num_actor_obs),  # (128,45)
        )
        print(f"Actor RNN: {self.memory_a}")
        print(f"Critic RNN: {self.memory_c}")
        print(f"encoder: {self.encoder}")
        print(f"decoder: {self.decoder}")

    def reset(self, dones=None):
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    def reparameterise(self, mean, logvar):
        var = torch.exp(logvar * 0.5)
        code_temp = torch.randn_like(var)
        code = mean + var * code_temp
        return code
    
    def cenet_forward(self, obs_history):
        # print(obs_history.size())
        distribution = self.encoder(obs_history)
        mean_latent = self.encode_mean_latent(distribution)
        logvar_latent = self.encode_logvar_latent(distribution)
        code_latent = self.reparameterise(mean_latent, logvar_latent)
        
        mean_vel = self.encode_mean_vel(distribution)
        logvar_vel = self.encode_logvar_vel(distribution)
        code_vel = self.reparameterise(mean_vel, logvar_vel)
        
        code = torch.cat((code_vel, code_latent), dim=-1)
        decode = self.decoder(code)
        return code, code_vel, decode, mean_vel, logvar_vel, mean_latent, logvar_latent

    def act(self, observations,obs_history, masks=None, hidden_states=None):
        code, _, decode, _, _, _, _ = self.cenet_forward(obs_history)
        _observations = torch.cat((code, observations), dim=-1)
        input_a = self.memory_a(_observations, masks, hidden_states)
        return super().act(input_a.squeeze(0))

    def act_inference(self, observations,obs_history):
        code, _, decode, _, _, _, _ = self.cenet_forward(obs_history)
        _observations = torch.cat((code, observations), dim=-1)
        input_a = self.memory_a(_observations)
        return super().act_inference(input_a.squeeze(0))

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        input_c = self.memory_c(critic_observations, masks, hidden_states)
        return super().evaluate(input_c.squeeze(0))
    
    def get_hidden_states(self):
        return self.memory_a.hidden_states, self.memory_c.hidden_states


class Memory(torch.nn.Module):
    def __init__(self, input_size, type='lstm', num_layers=1, hidden_size=256):
        super().__init__()
        # RNN
        rnn_cls = nn.GRU if type.lower() == 'gru' else nn.LSTM
        self.rnn = rnn_cls(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.hidden_states = None
    
    def forward(self, input, masks=None, hidden_states=None):
        batch_mode = masks is not None
        if batch_mode:
            # batch mode (policy update): need saved hidden states
            if hidden_states is None:
                raise ValueError("Hidden states not passed to memory module during policy update")
            out, _ = self.rnn(input, hidden_states)
            out = unpad_trajectories(out, masks)
        else:
            # inference mode (collection): use hidden states of last step
            out, self.hidden_states = self.rnn(input.unsqueeze(0), self.hidden_states)
        return out

    def reset(self, dones=None):
        # When the RNN is an LSTM, self.hidden_states_a is a list with hidden_state and cell_state
        for hidden_state in self.hidden_states:
            hidden_state[..., dones, :] = 0.0
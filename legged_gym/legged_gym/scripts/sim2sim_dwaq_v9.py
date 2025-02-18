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
from legged_gym.envs.pai.pai_config import PaiRoughCfg
from isaacgym.torch_utils import *

import torch

import csv
import pandas as pd


class cmd:
    vx = 1.
    vy = 0.0
    dyaw = 0.0

class env:
    obs=47
    obs_his=47*5

def quat_rotate_inverse(q, v):
    q_w = q[-1]
    q_vec = q[:3]
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c


def quat_rotate_inverse_ori(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c

def get_obs(data):
    '''Extracts an observation from the mujoco data structure
    '''
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
    omega = data.sensor('angular-velocity').data.astype(np.double)
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    return (q, dq, quat, v, omega, gvec)


def pd_control(target_q, q, kp, target_dq, dq, kd):
    '''Calculates torques from position commands
    '''
    # print("p:", (target_q - q) * kp )
    # print("d", (target_dq - dq) * kd)
    return (target_q - q) * kp + (target_dq - dq) * kd

def run_mujoco(logdir, cfg):
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

   

    count_lowlevel = 0
    
    count_csv = 0
    with open('sim2sim_robot_states.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # csvwriter.writerow([f'q_{i}' for i in range(19)])
        csvwriter.writerow([
            "sim2sim_base_euler_roll", "sim2sim_base_euler_pitch", "sim2sim_base_euler_yaw",
            # "sim2sim_base_quat_x", "sim2sim_base_quat_y", "sim2sim_base_quat_z", "sim2sim_base_quat_w",
            "sim2sim_dof_pos_0", "sim2sim_dof_pos_1", "sim2sim_dof_pos_2", "sim2sim_dof_pos_3",
            "sim2sim_dof_pos_4", "sim2sim_dof_pos_5", "sim2sim_dof_pos_6", "sim2sim_dof_pos_7",
            "sim2sim_dof_pos_8", "sim2sim_dof_pos_9", "sim2sim_dof_pos_10", "sim2sim_dof_pos_11",
            "sim2sim_target_dof_pos_0", "sim2sim_target_dof_pos_1", "sim2sim_target_dof_pos_2", "sim2sim_target_dof_pos_3",
            "sim2sim_target_dof_pos_4", "sim2sim_target_dof_pos_5", "sim2sim_target_dof_pos_6", "sim2sim_target_dof_pos_7",
            "sim2sim_target_dof_pos_8", "sim2sim_target_dof_pos_9", "sim2sim_target_dof_pos_10", "sim2sim_target_dof_pos_11",
        ])

        for _ in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating..."):
            # Obtain an observation
            q, dq, quat, v, omega, gvec = get_obs(data)
            q = q[-cfg.env.num_actions:]
            dq = dq[-cfg.env.num_actions:]

            for i in range(6):
                tmpq = q[i]
                q[i] = q[i+6]
                q[i+6] = tmpq

                tmpdq = dq[i]
                dq[i] = dq[i+6]
                dq[i+6] = tmpdq

            # 1000hz -> 100hz
            if count_lowlevel % cfg.sim_config.decimation == 0:
                obs = torch.zeros(1, env.obs, dtype=torch.float)

                projected_gravity = quat_rotate_inverse(quat, np.array([0., 0., -1.]))
                

                obs[0, :3] = torch.tensor(omega* cfg.normalization.obs_scales.ang_vel, dtype=torch.double)  #3
                obs[0, 3:6] =  torch.tensor(projected_gravity, dtype=torch.double)  #3
                obs[0, 6] =  torch.tensor(cmd.vx * cfg.normalization.obs_scales.lin_vel , dtype=torch.double)  
                obs[0, 7] =  torch.tensor(cmd.vy * cfg.normalization.obs_scales.lin_vel , dtype=torch.double)  
                obs[0, 8] =   torch.tensor(cmd.dyaw * cfg.normalization.obs_scales.ang_vel, dtype=torch.double)  
                obs[0, 9:21] = torch.tensor( q * cfg.normalization.obs_scales.dof_pos , dtype=torch.double)  #12
                obs[0, 21:33] = torch.tensor( dq * cfg.normalization.obs_scales.dof_vel , dtype=torch.double)  #12
                obs[0, 33:45] =   torch.tensor(action, dtype=torch.double)  #12
                obs[0, 45] = math.sin(2 * math.pi * count_lowlevel * cfg.sim_config.dt  / 0.64)
                obs[0, 46] = math.cos(2 * math.pi * count_lowlevel * cfg.sim_config.dt  / 0.64)

               
                obs = torch.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)

                action[:] = load_policy(logdir,obs,obs_history)[0].detach().numpy()
                obs_history = torch.cat((obs_history[:,env.obs:], obs[:,:]), dim=-1)

                action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)
                target_q = action * cfg.control.action_scale
                
                

            target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)
            
            # Generate PD control
            tau = pd_control(target_q, q, cfg.robot_config.kps,
                            target_dq, dq, cfg.robot_config.kds)  # Calc torques
            tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)  # Clamp torques
            for i in range(6):
                tmptau = tau[i]
                tau[i] = tau[i+6]
                tau[i+6] = tmptau
            data.ctrl = tau
            # print(tau)

            mujoco.mj_step(model, data)
            viewer.render()
            count_lowlevel += 1

    viewer.close()

def reparameterise(mean,logvar):
    var = torch.exp(logvar*0.5)
    code_temp = torch.randn_like(var)
    code = mean + var*code_temp
    return code

def load_policy(logdir,obs,obs_history):
    actor=torch.jit.load(logdir + '/actor_dwaq.pt')
    encoder=torch.jit.load(logdir + "/encoder_dwaq.pt")
    latent_mu = torch.jit.load(logdir + '/latent_mu_dwaq.pt')
    latent_var = torch.jit.load(logdir + '/latent_var_dwaq.pt')
    vel_mu = torch.jit.load(logdir + '/vel_mu_dwaq.pt')
    vel_var = torch.jit.load(logdir + '/vel_var_dwaq.pt')
     
    
    distribution= encoder(obs_history)
    mean_latent = latent_mu(distribution)
    logvar_latent = latent_var(distribution)
    
    mean_vel = vel_mu(distribution)
    logvar_vel = vel_var(distribution)
    code_latent = reparameterise(mean_latent,logvar_latent)
    code_vel = reparameterise(mean_vel,logvar_vel)
    code = torch.cat((code_vel,code_latent),dim=-1)

    # action = actor(torch.cat((obs, code), dim=-1))
    action = actor(torch.cat((code, obs), dim=-1))

    return action


def cenet_forward(self,obs_history):
        distribution = self.encoder(obs_history)
        mean_latent = self.encode_mean_latent(distribution)
        logvar_latent = self.encode_logvar_latent(distribution)
        # var = torch.exp(logvar_latent*0.5)
        # code_temp = torch.randn_like(var)
        # code = mean_latent + var*code_temp
        # print("latent : ",code[0])
        mean_vel = self.encode_mean_vel(distribution)
        logvar_vel = self.encode_mean_vel(distribution)
        code_latent = self.reparameterise(mean_latent,logvar_latent)
        code_vel = self.reparameterise(mean_vel,logvar_vel)
        code = torch.cat((code_vel,code_latent),dim=-1)
        decode = self.decoder(code)
        return code,code_vel,decode,mean_vel,logvar_vel,mean_latent,logvar_latent

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--logdir', type=str, required=True,
                        help='Run to load from.')
    parser.add_argument('--terrain', action='store_true', help='terrain or plane')
    args = parser.parse_args()

    class Sim2simCfg(PaiRoughCfg):

        class sim_config:
            # mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/pi_12dof_release_v1/mjcf/pi_12dof_release_v1.xml' #平地
            mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/pi_12dof_release_v1/mjcf/pi_12dof_release_v1_hfield_l1.xml' #hfield
            # mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/pi_12dof_release_v1/mjcf/pi_12dof_release_v1_hfield.xml' #hfield

            # mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/pi_12dof_release_v1/mjcf/pi_12dof_release_v1_slope.xml' #hfield


            sim_duration = 60.0
            dt = 0.001
            decimation = 10

        class robot_config:
            # kps = np.array([80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80], dtype=np.double) #v3
            # kps = np.array([80, 80, 80, 80, 80, 40, 80, 80, 80, 80, 80, 40], dtype=np.double) # v4
            kps = np.array([80, 40, 40, 80, 80, 20, 80, 40, 40, 80, 80, 20], dtype=np.double) # v7


            kds = np.array([0.5, 0.5, 0.5, 0.5, 0.4, 0.2, 0.5, 0.5, 0.5, 0.5, 0.4, 0.2], dtype=np.double)
            # kds = np.array([3,3,3,3, 3, 3, 3,3,3,3, 3, 3], dtype=np.double)

            tau_limit = 10. * np.ones(12, dtype=np.double)

    run_mujoco(args.logdir, Sim2simCfg())

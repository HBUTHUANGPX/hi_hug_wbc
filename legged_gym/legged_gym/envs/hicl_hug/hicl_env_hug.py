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


from .hicl_config_hug import HiclHugCfg as _cfg
from .hicl_utils_hug import (
    FootStepGeometry,
    SimpleLineGeometry,
    VelCommandGeometry,
    smart_sort,
)
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.math import (
    quat_apply_yaw,
    wrap_to_pi,
    torch_rand_sqrt_float,
    get_scale_shift,
    get_scale_shift,
    euler_from_quat,
)

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os
from torch.distributions import Normal


# from humanoid.utils.terrain import  HumanoidTerrain
# from collections import deque
def get_euler_xyz_tensor(quat):
    r, p, w = get_euler_xyz(quat)
    # stack r, p, w in dim1
    euler_xyz = torch.stack((r, p, w), dim=1)
    euler_xyz[euler_xyz > np.pi] -= 2 * np.pi
    return euler_xyz


class HiclHugEnv(LeggedRobot):
    """
    PaiFreeEnv is a class that represents a custom environment for a legged robot.

    Args:
        cfg (LeggedRobotCfg): Configuration object for the legged robot.
        sim_params: Parameters for the simulation.
        physics_engine: Physics engine used in the simulation.
        sim_device: Device used for the simulation.
        headless: Flag indicating whether the simulation should be run in headless mode.

    Attributes:
        last_feet_z (float): The z-coordinate of the last feet position.
        feet_height (torch.Tensor): Tensor representing the height of the feet.
        sim (gymtorch.GymSim): The simulation object.
        terrain (HumanoidTerrain): The terrain object.
        up_axis_idx (int): The index representing the up axis.
        command_input (torch.Tensor): Tensor representing the command input.
        privileged_obs_buf (torch.Tensor): Tensor representing the privileged observations buffer.
        obs_buf (torch.Tensor): Tensor representing the observations buffer.
        obs_history (collections.deque): Deque containing the history of observations.
        critic_history (collections.deque): Deque containing the history of critic observations.

    Methods:
        _push_robots(): Randomly pushes the robots by setting a randomized base velocity.
        _get_phase(): Calculates the phase of the gait cycle.
        _get_gait_phase(): Calculates the gait phase.
        compute_ref_state(): Computes the reference state.
        create_sim(): Creates the simulation, terrain, and environments.
        _get_noise_scale_vec(cfg): Sets a vector used to scale the noise added to the observations.
        step(actions): Performs a simulation step with the given actions.
        compute_observations(): Computes the observations.
        reset_idx(env_ids): Resets the environment for the specified environment IDs.
    """

    cfg: _cfg

    def __init__(self, cfg: _cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.last_feet_z = 0.0
        self.feet_height = torch.zeros((self.num_envs, 2), device=self.device)

        self.cmd_level = 0
        self.cmd_level_old = 0
        self.random_half_phase = torch.pi * torch.randint(
            0, 2, (1, self.num_envs), device=self.device, dtype=torch.long
        )
        self.disturbance = torch.zeros(
            self.num_envs,
            self.num_bodies,
            3,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.disturbance_norm = torch.zeros(
            self.num_envs,
            self.num_bodies,
            3,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        self.add_vel_norm = torch.zeros(
            self.num_envs,
            3,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.dist_norm = torch.zeros(
            self.num_envs,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        # self.reset_idx(torch.tensor(range(self.num_envs), device=self.device))
        # self.compute_observations()

    def _create_envs(self):
        """Creates environments:
        1. loads the robot URDF/MJCF asset,
        2. For each environment
           2.1 creates the environment,
           2.2 calls DOF and Rigid shape properties callbacks,
           2.3 create actor with these properties and add them to the env
        3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = (
            self.cfg.asset.replace_cylinder_with_capsule
        )
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options
        )
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        knee_names = [s for s in body_names if self.cfg.asset.knee_name in s]

        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        print("Check Body Names: ", body_names)
        print("aseet file", self.cfg.asset.file)

        base_init_state_list = (
            self.cfg.init_state.pos
            + self.cfg.init_state.rot
            + self.cfg.init_state.lin_vel
            + self.cfg.init_state.ang_vel
        )
        self.base_init_state = to_torch(
            base_init_state_list, device=self.device, requires_grad=False
        )
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])
        self.default_friction = rigid_shape_props_asset[1].friction

        self._get_env_origins()
        self._init_custom_buffers__()
        self._randomize_rigid_body_props(
            torch.arange(self.num_envs, device=self.device), self.cfg
        )

        env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
        env_upper = gymapi.Vec3(0.0, 0.0, 0.0)
        self.actor_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(
                self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs))
            )
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1.0, 1.0, (2, 1), device=self.device).squeeze(
                1
            )
            start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props = self._process_rigid_shape_props(
                rigid_shape_props_asset, i
            )
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(
                env_handle,
                robot_asset,
                start_pose,
                self.cfg.asset.name,
                i,
                self.cfg.asset.self_collisions,
                0,
            )
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(
                env_handle, actor_handle
            )
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(
                env_handle, actor_handle, body_props, recomputeInertia=True
            )
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_indices = torch.zeros(
            len(feet_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], feet_names[i]
            )

        self.knee_indices = torch.zeros(
            len(knee_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], knee_names[i]
            )

        self.penalised_contact_indices = torch.zeros(
            len(penalized_contact_names),
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], penalized_contact_names[i]
            )

        self.termination_contact_indices = torch.zeros(
            len(termination_contact_names),
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], termination_contact_names[i]
            )

    def post_physics_step(self):
        """check terminations, compute observations and rewards
        calls self._post_physics_step_callback() for common computations
        calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 7:10]
        )
        self.base_ang_vel[:] = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 10:13]
        )
        self.projected_gravity[:] = quat_rotate_inverse(
            self.base_quat, self.gravity_vec
        )
        self.foot_positions[:] = self.rigid_body_state.view(
            self.num_envs, self.num_bodies, 13
        )[:, self.feet_indices, 0:3]
        self.foot_orient[:] = self.rigid_body_state.view(
            self.num_envs, self.num_bodies, 13
        )[:, self.feet_indices, 3:7]
        self.foot_velocities[:] = self.rigid_body_state.view(
            self.num_envs, self.num_bodies, 13
        )[:, self.feet_indices, 7:10]
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)

        self._post_physics_step_callback()
        self.update_behavior_command()
        self.first_rc_filter()
        swing_mask = 1 - self._get_gait_phase()
        self.swing_mask = swing_mask * (1 - self.standing_command_mask.unsqueeze(1))
        self.stance_mask = 1 - self.swing_mask

        self.swing_mask_l = self.swing_mask[:, 0]
        self.swing_mask_r = self.swing_mask[:, 1]
        
        foot_pos_world = self.rigid_state[:, self.feet_indices, 0:3]
        foot_pos_base_left = quat_rotate_inverse(
            self.base_quat, foot_pos_world[:, 0, :] - self.base_pos
        )
        foot_pos_base_right = quat_rotate_inverse(
            self.base_quat, foot_pos_world[:, 1, :] - self.base_pos
        )
        # print(foot_pos_base_left[0,1])
        self.feet_height_base_l = foot_pos_base_left[:,2:3] + 0.4294
        self.feet_height_base_r = foot_pos_base_right[:,2:3] + 0.4294
        
        self.feet_x_base_l = foot_pos_base_left[:,0:1]
        self.feet_x_base_r = foot_pos_base_right[:,0:1]
        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations()  # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.slast_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_last_joint_pos_target[:] = self.last_joint_pos_target[:]
        self.last_joint_pos_target[:] = self.joint_pos_target[:]

        self.last_dof_vel[:] = self.dof_vel[:]
        self.root_acc[:] = self.root_states[:, 7:10] - self.last_root_vel
        self.root_acc_base = quat_rotate_inverse(self.base_quat, self.root_acc)
        ra = self.root_acc
        ra[:, 2] -= 9.81
        self.root_acc_base_with_g = self.root_acc_base + quat_rotate_inverse(
            self.base_quat, ra
        )
        self.last_root_vel[:] = self.root_states[:, 7:10]
        self.last_last_actions[:] = torch.clone(self.last_actions[:])
        self.base_orn_rp[:] = self.get_body_orientation()

        # print(torch.mean(torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1),dim=-1))

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

        # if not self.headless:
        #     self._visualization()

    def check_termination(self):
        """Check if environments need to be reset"""
        self.reset_buf = torch.any(
            torch.norm(
                self.contact_forces[:, self.termination_contact_indices, :], dim=-1
            )
            > 1.0,
            dim=1,
        )
        self.time_out_buf = (
            self.episode_length_buf > self.max_episode_length
        )  # no terminal reward for time-outs
        self.reset_buf |= torch.any(
            torch.abs(self.base_euler_xyz[:, 1:2]) > 45 / 180 * torch.pi, dim=1
        )
        self.reset_buf |= torch.any(
            torch.abs(self.base_euler_xyz[:, 0:1]) > 45 / 180 * torch.pi, dim=1
        )
        self.reset_buf |= self.time_out_buf

    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        # if self.cfg.domain_rand.randomize_base_mass:
        #     rng = self.cfg.domain_rand.added_mass_range
        #     props[0].mass += np.random.uniform(rng[0], rng[1])
        # return props

        self.default_body_mass = props[0].mass
        props[0].mass = self.default_body_mass + self.payloads[env_id]
        props[0].com = gymapi.Vec3(
            self.com_displacements[env_id, 0],
            self.com_displacements[env_id, 1],
            self.com_displacements[env_id, 2],
        )

        return props

    def _randomize_rigid_body_props(self, env_ids, cfg: _cfg):

        if cfg.domain_rand.randomize_base_mass:
            min_payload, max_payload = cfg.domain_rand.added_mass_range
            # self.payloads[env_ids] = -1.0
            self.payloads[env_ids] = (
                torch.rand(
                    len(env_ids),
                    dtype=torch.float,
                    device=self.device,
                    requires_grad=False,
                )
                * (max_payload - min_payload)
                + min_payload
            )
        if cfg.domain_rand.randomize_com_displacement:
            min_com_displacement, max_com_displacement = (
                cfg.domain_rand.com_displacement_range
            )
            self.com_displacements[env_ids, :] = (
                torch.rand(
                    len(env_ids),
                    3,
                    dtype=torch.float,
                    device=self.device,
                    requires_grad=False,
                )
                * (max_com_displacement - min_com_displacement)
                + min_com_displacement
            )

        if cfg.domain_rand.randomize_friction:
            min_friction, max_friction = cfg.domain_rand.friction_range
            self.friction_coeffs[env_ids, :] = (
                torch.rand(
                    len(env_ids),
                    1,
                    dtype=torch.float,
                    device=self.device,
                    requires_grad=False,
                )
                * (max_friction - min_friction)
                + min_friction
            )

    def torch_rand_float_with_min(self, low, high, size, device, min_abs=0.05):
        rand: torch.Tensor = low + (high - low) * torch.rand(size, device=device)
        # 生成绝对值小于 min_abs 且不为0的掩码
        mask = (rand.abs() < min_abs) & (rand != 0)
        # 将满足条件的值调整为 min_abs，保留符号
        rand[mask] = min_abs * torch.sign(rand[mask])
        return rand

    def init_behavior_command(self):
        self.gait_f_min = self.cfg.commands.ranges.gait_frequency[0]
        self.gait_f_max = self.cfg.commands.ranges.gait_frequency[1]
        self.f_t = (
            torch.rand(
                self.num_envs,
                1,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
            * (self.gait_f_max - self.gait_f_min)
            + self.gait_f_min
        )

        self.swing_height_min = self.cfg.commands.ranges.swing_height[0]
        self.swing_height_max = self.cfg.commands.ranges.swing_height[1]
        self.l_t = (
            torch.rand(
                self.num_envs,
                1,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
            * (self.swing_height_max - self.swing_height_min)
            + self.swing_height_min
        )

        self.base_height_min = self.cfg.commands.ranges.base_height[0]
        self.base_height_max = self.cfg.commands.ranges.base_height[1]
        self.h_t = (
            torch.rand(
                self.num_envs,
                1,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
            * (self.base_height_max - self.base_height_min)
            + self.base_height_min
        )

        self.psi = torch.zeros(
            self.num_envs,
            1,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.phy_1 = torch.zeros(
            self.num_envs,
            1,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.phy_2 = torch.zeros(
            self.num_envs,
            1,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.phy_1_bar = torch.zeros(
            self.num_envs,
            1,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.phy_2_bar = torch.zeros(
            self.num_envs,
            1,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.clock_1 = torch.zeros(
            self.num_envs,
            1,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.clock_2 = torch.zeros(
            self.num_envs,
            1,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.l_t_1 = torch.zeros(
            self.num_envs,
            1,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.l_t_2 = torch.zeros(
            self.num_envs,
            1,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.l_t_all = torch.zeros(
            self.num_envs,
            2,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.phi_stance = 0.5 * torch.ones(
            self.num_envs,
            1,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.zeros_n = torch.zeros(
            self.num_envs,
            1,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.ones_n = torch.ones(
            self.num_envs,
            1,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.psz = torch.zeros(
            self.num_envs,
            1,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.pez = torch.zeros(
            self.num_envs,
            1,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.phi_0_5 = self.normalized_phase(
            self.ones_n * 0.5, self.phi_stance
        )  # 随机生成一个n×1的phi_0_5
        self.phi_0_75 = self.normalized_phase(
            self.ones_n * 0.75, self.phi_stance
        )  # 随机生成一个n×1的phi_0_75
        self.phi_1_0 = self.normalized_phase(self.ones_n, self.phi_stance)  # 固定为1
        self.CDF_1 = self.C_fun(self.phy_1, 0.05)
        self.CDF_2 = self.C_fun(self.phy_2, 0.05)

    def update_behavior_command(self):
        """
        ψ：phase offset
        ϕ0,1: periodic phase (left,right leg,ϕi ∈ [0, 1))
        ϕstance : duty cycle(∈ [0, 1] )
        """
        # ψ：phase offset
        # walking gaits ψ = 0.5
        # standing gait ϕi = 0.25，ψ = 0
        standing_mask = self.standing_command_mask == 1
        walking_mask = self.standing_command_mask != 1
        self.psi[walking_mask] = 0.5
        self.psi[standing_mask] = 0

        # ϕ0,1: periodic phase (left,right leg,ϕi ∈ [0, 1))
        self.phy_1 += self.f_t * self.dt
        self.phy_2 = self.phy_1 + self.psi
        self.phy_1[self.phy_1 >= 1.0] = 0.0
        self.phy_2[self.phy_2 >= 1.0] -= 1.0
        self.phy_1[standing_mask] = 0.25 * torch.ones_like(self.phy_1[standing_mask])
        self.phy_2[standing_mask] = 0.25 * torch.ones_like(self.phy_2[standing_mask])

        self.phy_1_bar = self.normalized_phase(self.phy_1, self.phi_stance)
        self.phy_2_bar = self.normalized_phase(self.phy_2, self.phi_stance)

        self.clock_1 = torch.sin(self.phy_1_bar * 2 * torch.pi)
        self.clock_2 = torch.sin(self.phy_2_bar * 2 * torch.pi)

        self.l_t_1 = self.l_t_cal(
            self.phy_1, self.phy_1_bar, self.psz, self.pez, self.l_t, self.zeros_n
        )
        self.l_t_2 = self.l_t_cal(
            self.phy_2, self.phy_2_bar, self.psz, self.pez, self.l_t, self.zeros_n
        )
        self.l_t_all = torch.cat([self.l_t_1, self.l_t_2], dim=1)

        self.CDF_1 = self.C_fun(self.phy_1, 0.05)
        self.CDF_2 = self.C_fun(self.phy_2, 0.05)

    def resample_behavior_command(self, env_ids):
        """
        f_t : gait frequency
        l_t : maximum foot swing height
        h_t : body height
        p_t : pitch angle
        """
        # f_t : gait frequency 随机 1.5-3
        self.f_t[env_ids] = (
            torch.rand(
                self.num_envs,
                1,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
            * (self.gait_f_max - self.gait_f_min)
            + self.gait_f_min
        )[env_ids]
        # l_t : maximum foot swing height
        self.l_t[env_ids] = (
            torch.rand(
                self.num_envs,
                1,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
            * (self.swing_height_max - self.swing_height_min)
            + self.swing_height_min
        )[env_ids]
        self.h_t[env_ids] = (
            torch.rand(
                self.num_envs,
                1,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
            * (self.base_height_max - self.base_height_min)
            + self.base_height_min
        )[env_ids]

    def l_t_cal(self, phi, nm_phi, p_s_z, p_e_z, l_t, zero_n):
        A1 = self.return_A(self.phi_0_5, self.phi_0_75, self.ones_n, self.zeros_n)
        A2 = self.return_A(self.phi_0_75, self.phi_1_0, self.ones_n, self.zeros_n)
        # 定义常数项矩阵
        b1 = torch.stack([p_s_z, l_t, zero_n, zero_n, zero_n, zero_n], dim=1)
        b2 = torch.stack([l_t, p_e_z, zero_n, zero_n, zero_n, zero_n], dim=1)
        a1 = torch.linalg.solve(A1, b1)
        a2 = torch.linalg.solve(A2, b2)

        l_t_target = torch.where(
            phi < 0.75,
            a1[:, 0] * nm_phi**5
            + a1[:, 1] * nm_phi**4
            + a1[:, 2] * nm_phi**3
            + a1[:, 3] * nm_phi**2
            + a1[:, 4] * nm_phi
            + a1[:, 5],
            a2[:, 0] * nm_phi**5
            + a2[:, 1] * nm_phi**4
            + a2[:, 2] * nm_phi**3
            + a2[:, 3] * nm_phi**2
            + a2[:, 4] * nm_phi
            + a2[:, 5],
        )
        l_t_target[phi < 0.5] = 0
        return l_t_target  # num_envs * num_envs * 1

    def return_A(self, s_phi, e_phi, ones_n, zeros_n):
        # 定义多项式的系数矩阵
        A1_1 = torch.cat(
            [s_phi**5, s_phi**4, s_phi**3, s_phi**2, s_phi, ones_n], dim=1
        )  # n×6 矩阵
        A1_2 = torch.cat(
            [e_phi**5, e_phi**4, e_phi**3, e_phi**2, e_phi, ones_n], dim=1
        )  # n×6 矩阵
        A1_3 = torch.cat(
            [5 * s_phi**4, 4 * s_phi**3, 3 * s_phi**2, 2 * s_phi, ones_n, zeros_n],
            dim=1,
        )  # n×6 矩阵
        A1_4 = torch.cat(
            [5 * e_phi**4, 4 * e_phi**3, 3 * e_phi**2, 2 * e_phi, ones_n, zeros_n],
            dim=1,
        )  # n×6 矩阵
        A1_5 = torch.cat(
            [20 * s_phi**3, 12 * s_phi**2, 6 * s_phi, 2 * ones_n, zeros_n, zeros_n],
            dim=1,
        )  # n×6 矩阵
        A1_6 = torch.cat(
            [20 * e_phi**3, 12 * e_phi**2, 6 * e_phi, 2 * ones_n, zeros_n, zeros_n],
            dim=1,
        )  # n×6 矩阵
        return torch.stack([A1_1, A1_2, A1_3, A1_4, A1_5, A1_6], dim=1)  # n×6×6 矩阵

    def normalized_phase(self, phi_i, phi_stance):
        phy_i_bar = torch.zeros_like(phi_i)
        mask_sma = phi_i < phi_stance
        if torch.any(mask_sma):
            phy_i_bar[mask_sma] = (0.5 * (phi_i / phi_stance))[mask_sma]
        mask_big = phi_i >= phi_stance
        if torch.any(mask_big):
            phy_i_bar[mask_big] = (
                0.5 + 0.5 * ((phi_i - phi_stance) / (1 - phi_stance))
            )[mask_big]
        return phy_i_bar

    def first_rc_filter(self):
        
        ...
        
    def _resample_commands(self, env_ids):
        """Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.command_catrgories[env_ids] = torch.randint(
            0,
            5,
            (len(env_ids), 1),
            device=self.device,
            requires_grad=False,
        ).squeeze(1)
        # 0 is standing，vel x is 0, vel y is 0, ang vel yaw is 0
        # 1 is walking in sagittal, vel y is 0, ang vel yaw is 0
        # 2 is walking laterally, vel x is 0, ang vel yaw is 0
        # 3 is rotating in place, vel x is 0, vel y is 0
        # 4 is omnidirectional walking, all commands are random
        self.commands[env_ids, 0] = self.torch_rand_float_with_min(  # vel x
            self.command_ranges["lin_vel_x"][0],
            self.command_ranges["lin_vel_x"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        self.commands[env_ids, 1] = self.torch_rand_float_with_min(  # vel y
            self.command_ranges["lin_vel_y"][0],
            self.command_ranges["lin_vel_y"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        self.commands[env_ids, 2] = self.torch_rand_float_with_min(  # ang vel yaw
            self.command_ranges["ang_vel_yaw"][0],
            self.command_ranges["ang_vel_yaw"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        if torch.any(self.command_catrgories == 1):
            self.commands[self.command_catrgories == 1, 1] = 0
            self.commands[self.command_catrgories == 1, 2] = 0
            self.standing_command_mask[self.command_catrgories == 1] = 0
        if torch.any(self.command_catrgories == 2):
            self.commands[self.command_catrgories == 2, 0] = 0
            self.commands[self.command_catrgories == 2, 2] = 0
            self.standing_command_mask[self.command_catrgories == 2] = 0
        if torch.any(self.command_catrgories == 3):
            self.commands[self.command_catrgories == 3, 0] = 0
            self.commands[self.command_catrgories == 3, 1] = 0
            self.standing_command_mask[self.command_catrgories == 3] = 0
        if torch.any(self.command_catrgories == 0):
            self.commands[self.command_catrgories == 0, 0] = 0
            self.commands[self.command_catrgories == 0, 1] = 0
            self.commands[self.command_catrgories == 0, 2] = 0
            self.standing_command_mask[self.command_catrgories == 0] = 1
        # print("=============standing_command_mask===============")
        # print(self.standing_command_mask)
        # print(self.commands)

        # set small commands to zero
        self.commands[env_ids, :2] *= (
            torch.norm(self.commands[env_ids, :2], dim=1) > 0.1
        ).unsqueeze(1)
        self.command_height = 0.05 * (
            torch.rand(
                self.num_envs,
                1,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
            - 0.5
        )  # base height
        self.resample_behavior_command(env_ids)

    def step(self, actions):
        """Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        # self.obs_hist_buf.append(self.obs_buf) # append the latest obs into hist
        self.disturbance_force = self.disturbance_force.to(self.device)
        self.obs_hist_buf = self.obs_hist_buf[:, self.cfg.env.num_observations :]
        self.obs_hist_buf = torch.cat((self.obs_hist_buf, self.obs_buf), dim=-1)
        # print("###########obs_hist_buf=====",self.obs_hist_buf)
        self.prev_privileged_obs_buf = self.privileged_obs_buf
        self.prev_foot_velocities = self.foot_velocities

        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # print("#######actions : ", self.actions[0])
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(self.torques)
            )
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        # print("#######obs_buf=====",self.obs_buf[0])
        # self.obs_hist_buf = self.obs_hist_buf[:,45:]
        # self.obs_hist_buf = torch.cat((self.obs_hist_buf,self.obs_buf),dim = -1)

        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(
                self.privileged_obs_buf, -clip_obs, clip_obs
            )
        # self._reward_joint_pos()

        self.step_pos[:, :, :2] = self.rigid_state[:, self.feet_indices, :2]
        self.step_pos[:, :, 2] = self.rigid_state[:, self.feet_indices, 5]
        # 左上
        self.quadrilateral_point[:, 0, 0] = self.step_pos[:, 0, 0] + 0.095
        self.quadrilateral_point[:, 0, 1] = self.step_pos[:, 0, 1] + 0.04
        # 左下
        self.quadrilateral_point[:, 1, 0] = self.step_pos[:, 0, 0] - 0.06
        self.quadrilateral_point[:, 1, 1] = self.step_pos[:, 0, 1] + 0.04
        # 右下
        self.quadrilateral_point[:, 2, 0] = self.step_pos[:, 1, 0] - 0.06
        self.quadrilateral_point[:, 2, 1] = self.step_pos[:, 1, 1] - 0.04
        # 右上
        self.quadrilateral_point[:, 3, 0] = self.step_pos[:, 1, 0] + 0.095
        self.quadrilateral_point[:, 3, 1] = self.step_pos[:, 1, 1] - 0.04
        # print(self.quadrilateral_point[0, ...])
        return (
            self.obs_buf,
            self.privileged_obs_buf,
            self.prev_privileged_obs_buf,
            self.obs_hist_buf,
            self.rew_buf,
            self.reset_buf,
            self.extras,
        )  ## do we need to return obs history buffer??

    def _init_buffers(self):
        """Initialize torch tensors which will contain simulation states and processed quantities"""
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        dof_force = self.gym.acquire_dof_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.forces = gymtorch.wrap_tensor(dof_force)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.disturbance_force = self.forces.view(self.num_envs, self.num_dof)
        self.base_quat = self.root_states[:, 3:7]
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)[
            : self.num_envs * self.num_bodies, :
        ]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(
            self.num_envs, -1, 3
        )  # shape: num_envs, num_bodies, xyz axis
        self.rigid_state = gymtorch.wrap_tensor(rigid_body_state).view(
            self.num_envs, 13, 13
        )

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(
            get_axis_params(-1.0, self.up_axis_idx), device=self.device
        ).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1.0, 0.0, 0.0], device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.torques = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.p_gains = torch.zeros(
            self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.d_gains = torch.zeros(
            self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.actions = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.slast_actions = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        self.last_actions = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_last_actions = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        self.joint_pos_target = torch.zeros(
            self.num_envs,
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_joint_pos_target = torch.zeros(
            self.num_envs,
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_last_joint_pos_target = torch.zeros(
            self.num_envs,
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:10])
        self.root_acc = torch.zeros_like(self.root_states[:, 7:10])
        self.root_acc_base = torch.zeros_like(self.root_acc)
        self.root_acc_base_with_g = torch.zeros_like(self.root_acc)
        self.commands = torch.zeros(
            self.num_envs,
            self.cfg.commands.num_commands,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )  # x vel, y vel, yaw vel, heading
        self.command_height = torch.zeros(
            self.num_envs,
            1,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )  # base height
        self.command_base_pitch = torch.zeros(
            self.num_envs,
            1,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )  # base pitch
        self.command_swing_height = torch.zeros(
            self.num_envs,
            1,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )  # swing foot height

        self.commands_scale = torch.tensor(
            [self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel],
            device=self.device,
            requires_grad=False,
        )  # TODO change this
        self.feet_air_time = torch.zeros(
            self.num_envs,
            self.feet_indices.shape[0],
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_contacts = torch.zeros(
            self.num_envs,
            len(self.feet_indices),
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )
        self.base_lin_vel = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 7:10]
        )
        self.base_ang_vel = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 10:13]
        )
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.foot_positions = self.rigid_body_state.view(
            self.num_envs, self.num_bodies, 13
        )[:, self.feet_indices, 0:3]
        self.foot_orient = self.rigid_body_state.view(
            self.num_envs, self.num_bodies, 13
        )[:, self.feet_indices, 3:7]
        self.foot_velocities = self.rigid_body_state.view(
            self.num_envs, self.num_bodies, 13
        )[:, self.feet_indices, 7:10]
        self.prev_foot_velocities = self.rigid_body_state.view(
            self.num_envs, self.num_bodies, 13
        )[:, self.feet_indices, 7:10]
        self.base_pos = self.root_states[:, 0:3]
        self.base_orn_rp = self.get_body_orientation()  # [r, p]
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(
            self.num_dof, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.default_joint_pd_target = self.default_dof_pos.clone()
        self.ref_dof_pos = torch.zeros_like(self.dof_pos)

        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.0
                self.d_gains[i] = 0.0
                if self.cfg.control.control_type in ["P", "V"]:
                    print(
                        f"PD gain of joint {name} were not defined, setting them to zero"
                    )
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        self.contact_queue = torch.zeros(
            self.num_envs,
            (int(0.35 / self.dt) + 1),
            2,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.standing_command_mask = torch.zeros(
            self.num_envs, dtype=torch.int64, device=self.device, requires_grad=False
        )
        self.command_catrgories = torch.randint(
            0,
            5,
            (self.num_envs, 1),
            device=self.device,
            requires_grad=False,
        ).squeeze(1)
        self.vel_x_command_mask = torch.ones(
            self.num_envs, dtype=torch.int64, device=self.device, requires_grad=False
        )
        self.vel_y_command_mask = torch.ones(
            self.num_envs, dtype=torch.int64, device=self.device, requires_grad=False
        )
        self.ang_vel_yaw_command_mask = torch.ones(
            self.num_envs, dtype=torch.int64, device=self.device, requires_grad=False
        )
        self.step_pos = torch.zeros(
            self.num_envs,
            len(self.feet_indices),
            3,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.quadrilateral_point = torch.zeros(
            self.num_envs,
            4,
            2,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.init_behavior_command()
        self._init_fft()
        self.feet_height_base_l = torch.zeros(
            self.num_envs,
            1,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.feet_height_base_r = torch.zeros(
            self.num_envs,
            1,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.feet_x_base_l = torch.zeros(
            self.num_envs,
            1,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.feet_x_base_r = torch.zeros(
            self.num_envs,
            1,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.feet_y_base_l = torch.zeros(
            self.num_envs,
            1,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.feet_y_base_r = torch.zeros(
            self.num_envs,
            1,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

    def get_body_orientation(self, return_yaw=False):
        r, p, y = euler_from_quat(self.base_quat)
        if return_yaw:
            return torch.stack([r, p, y], dim=-1)
        else:
            return torch.stack([r, p], dim=-1)

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        super().reset_idx(env_ids)
        self.last_last_actions[env_ids] = 0.0
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)
        self.random_half_phase[0, env_ids] = torch.pi * torch.randint(
            0, 2, (len(env_ids),), device=self.device
        )
        self.phy_1[env_ids] = 0.0

    def _get_phase(self):
        cycle_time = self.cfg.rewards.cycle_time
        phase = (
            self.episode_length_buf * self.dt / cycle_time + self.random_half_phase[0]
        )
        return phase

    def _get_gait_phase(self):
        # return float mask 1 is stance, 0 is swing
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)

        # Add double support phase
        stance_mask = torch.zeros((self.num_envs, 2), device=self.device)
        # left foot stance
        stance_mask[:, 0] = sin_pos >= 0
        # right foot stance
        stance_mask[:, 1] = sin_pos < 0
        # Double support phase
        stance_mask[torch.abs(sin_pos) < self.cfg.rewards.bias] = 1

        return stance_mask

    def compute_ref_state(self):
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        eps = self.cfg.rewards.eps
        sin_pos_l = sin_pos.clone() + self.cfg.rewards.bias
        sin_pos_l = (
            sin_pos_l / torch.sqrt(sin_pos_l**2.0 + eps**2.0) - self.cfg.rewards.y_bias
        )
        sin_pos_l[sin_pos_l > 0] = 0
        # sin_pos_l += 0.05

        sin_pos_r = sin_pos.clone() - self.cfg.rewards.bias
        sin_pos_r = (
            sin_pos_r / torch.sqrt(sin_pos_r**2.0 + eps**2.0) + self.cfg.rewards.y_bias
        )
        sin_pos_r[sin_pos_r < 0] = 0
        # sin_pos_r -= 0.05

        self.ref_dof_pos = torch.zeros_like(self.dof_pos)
        scale_1 = self.cfg.rewards.target_joint_pos_scale
        scale_2 = 2 * scale_1
        # left foot stance phase set to default joint pos
        self.ref_dof_pos[:, 0] = sin_pos_l * scale_1
        self.ref_dof_pos[:, 3] = -sin_pos_l * scale_2
        self.ref_dof_pos[:, 4] = sin_pos_l * scale_1
        # right foot stance phase set to default joint pos
        self.ref_dof_pos[:, 6] = -sin_pos_r * scale_1
        self.ref_dof_pos[:, 9] = sin_pos_r * scale_2
        self.ref_dof_pos[:, 10] = -sin_pos_r * scale_1
        # Double support phase
        # self.ref_dof_pos[torch.abs(sin_pos) < 0.5] = 0

    def update_command_curriculum(self, env_ids):
        """Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        ...
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if (
            torch.mean(self.episode_sums["tracking_lin_vel"][env_ids])
            / self.max_episode_length
            > 1.72 * self.reward_scales["tracking_lin_vel"]
        ):
            mm = (
                self.cfg.commands.ranges.lin_vel_x[1]
                - self.cfg.commands.ranges.lin_vel_x[0]
            )
            self.command_ranges["lin_vel_x"][0] = np.clip(
                self.command_ranges["lin_vel_x"][0] - 0.01 * mm,
                -self.cfg.commands.max_curriculum_x[0],
                0.0,
            )
            self.command_ranges["lin_vel_x"][1] = np.clip(
                self.command_ranges["lin_vel_x"][1] + 0.01 * mm,
                0.0,
                self.cfg.commands.max_curriculum_x[1],
            )
            self.cmd_level += 1
            self.cmd_level_old = self.cmd_level

        if (
            torch.mean(self.episode_sums["tracking_lin_vel"][env_ids])
            / self.max_episode_length
            < 1.7 * self.reward_scales["tracking_lin_vel"]
        ) and self.cmd_level_old > 0:
            mm = (
                self.cfg.commands.ranges.lin_vel_x[1]
                - self.cfg.commands.ranges.lin_vel_x[0]
            )
            self.command_ranges["lin_vel_x"][0] = np.clip(
                self.command_ranges["lin_vel_x"][0] + 0.01 * mm,
                -self.cfg.commands.max_curriculum_x[0],
                0.0,
            )
            self.command_ranges["lin_vel_x"][1] = np.clip(
                self.command_ranges["lin_vel_x"][1] - 0.01 * mm,
                0.0,
                self.cfg.commands.max_curriculum_x[1],
            )
            self.cmd_level -= 1
            self.cmd_level_old = self.cmd_level

    def _push_robots(self):
        """Random pushes the robots. Emulates an impulse by setting a randomized base velocity."""
        # print("Pushing robots")
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        add_vel = torch_rand_float(
            -max_vel, max_vel, (self.num_envs, 3), device=self.device
        )  # lin vel x/y
        add_vel[2] = 0
        self.add_vel_norm = add_vel / (2 * max_vel)
        self.dist_norm = torch.norm(self.add_vel_norm, dim=1)
        self.root_states[:, 7:10] += add_vel

        # self.disturbance_force = torch.Tensor(self.root_states[:,7:9]).to(self.device)
        self.gym.set_actor_root_state_tensor(
            self.sim, gymtorch.unwrap_tensor(self.root_states)
        )
        # print(self.add_vel_norm.size())

    def _disturbance_robots(self):
        """Random add disturbance force to the robots."""
        disturbance = torch_rand_float(
            self.cfg.domain_rand.disturbance_range[0],
            self.cfg.domain_rand.disturbance_range[1],
            (self.num_envs, 3),
            device=self.device,
        )
        self.disturbance_norm[:, 0, :] = disturbance / (
            self.cfg.domain_rand.disturbance_range[1]
            - self.cfg.domain_rand.disturbance_range[0]
        )
        self.disturbance[:, 0, :] = disturbance * 0
        # print(self.disturbance[0, 0, :])
        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            forceTensor=gymtorch.unwrap_tensor(self.disturbance),
            space=gymapi.CoordinateSpace.LOCAL_SPACE,
        )

    def compute_observations(self):
        """Computes observations"""
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1)

        stance_mask = self._get_gait_phase()
        contact_mask = self.contact_forces[:, self.feet_indices, 2] > 1.0
        self.compute_ref_state()
        diff = self.dof_pos - self.ref_dof_pos
        heights = (
            torch.clip(
                self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
                -1,
                1.0,
            )
            * self.obs_scales.height_measurements
        )
        contact_forces_scale, contact_forces_shift = get_scale_shift(
            self.cfg.normalization.contact_force_range
        )

        self.obs_buf = torch.cat(
            (
                self.base_ang_vel * self.obs_scales.ang_vel,  # 3
                self.projected_gravity,  # 3
                self.commands[:, :3] * self.commands_scale,  # 3
                # self.command_height,
                self.standing_command_mask.unsqueeze(1),  # 1
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,  # 12
                self.dof_vel * self.obs_scales.dof_vel,  # 12
                self.actions,  # 12
                # self.root_acc_base_with_g, # 3
                self.f_t,  # 1 46
                self.l_t,  # 1 47
                # self.h_t,  # 1
                self.psi,  # 1 48
                self.clock_1,  # 1 49
                self.clock_2,  # 1 50
                self.feet_height_base_l,
                self.feet_height_base_r,
                self.feet_x_base_l,
                self.feet_x_base_r,
            ),
            dim=-1,
        )  # 48
        """
        if torch.isnan(self.dof_vel).any():
            for i, row in enumerate(self.dof_vel):
                if torch.isnan(row).any():
                    print(f"{i} {row}")
            raise ValueError("self.dof_vel contains NaN")
        """

        self.privileged_obs_buf = torch.cat(
            (
                self.base_ang_vel * self.obs_scales.ang_vel,  # 3
                self.projected_gravity,  # 3
                self.commands[:, :3] * self.commands_scale,  # 3
                # self.command_height,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,  # 12
                self.dof_vel * self.obs_scales.dof_vel,  # 12
                self.actions,  # 12
                self.base_lin_vel * self.obs_scales.lin_vel,  # 3
                # self.root_acc_base_with_g, # 3
                (self.contact_forces.view(self.num_envs, -1) - contact_forces_shift)
                * contact_forces_scale,  # 13*3 = 39
                heights,  # 17*11 = 187
                diff,  # 12
                self.stance_mask,  # 2
                contact_mask,  # 2
                self.standing_command_mask.unsqueeze(1),  # 1
                self.add_vel_norm,  # 3
                self.f_t,  # 1
                self.l_t,  # 1
                # self.h_t,  # 1
                self.psi,  # 1
                self.clock_1,  # 1
                self.clock_2,  # 1
                self.feet_height_base_l,
                self.feet_height_base_r,
                self.feet_x_base_l,
                self.feet_x_base_r,
            ),
            dim=-1,
        )
        # print(self.privileged_obs_buf.size())

        ## privileged_obs_buffer shape 96,268
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (
                2 * torch.rand_like(self.obs_buf) - 1
            ) * self.noise_scale_vec

    def _get_noise_scale_vec(self, cfg):
        """Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel

        if self.cfg.env.obs_with_base_lin_vel:
            noise_vec[3:6] = (
                noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
            )
            noise_vec[6:9] = noise_scales.gravity * noise_level
            noise_vec[12:24] = (
                noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
            )
            noise_vec[24:36] = (
                noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
            )
        else:
            noise_vec[3:6] = noise_scales.gravity * noise_level
            # noise_vec[6:9] = 0. # commands
            noise_vec[9:21] = (
                noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
            )
            noise_vec[21:33] = (
                noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
            )
            # noise_vec[33:45] = 0. # previous actions

        # if self.cfg.terrain.measure_heights:
        #     noise_vec[48:235] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
        return noise_vec

    # ================HugWBC========================

    def _reward_ankle_roll_posture_roll(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.0
        feet_eular_0 = get_euler_xyz_tensor(
            self.rigid_state[:, self.feet_indices[0], 3:7]
        )[:, 0:1]
        feet_eular_1 = get_euler_xyz_tensor(
            self.rigid_state[:, self.feet_indices[1], 3:7]
        )[:, 0:1]
        # print(feet_eular_1.size())
        rew = torch.exp(
            -(
                torch.norm(feet_eular_0 * contact[:, 0], dim=1)
                + torch.norm(feet_eular_1 * contact[:, 1], dim=1)
            )
        )
        # print(nn.size())
        return rew

    def _reward_ankle_roll_posture_pitch(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 0.1

        feet_eular_0 = get_euler_xyz_tensor(
            self.rigid_state[:, self.feet_indices[0], 3:7]
        )[:, 1:2]
        feet_eular_1 = get_euler_xyz_tensor(
            self.rigid_state[:, self.feet_indices[1], 3:7]
        )[:, 1:2]
        # print(feet_eular_1.size())
        rew = torch.exp(
            -(
                torch.norm(feet_eular_0 * contact[:, 0], dim=1)
                + torch.norm(feet_eular_1 * contact[:, 1], dim=1)
            )
        )
        # print(nn.size())
        return rew

    def _init_fft(self):
        window_size = int(100 * 4.0)
        self.buffer_left = torch.zeros(self.num_envs, window_size, device=self.device)
        self.buffer_right = torch.zeros(self.num_envs, window_size, device=self.device)
        self.alpha = 1.0
        self.beta = 1.0
        self.tau_samples = int(100 * 1 / 0.27)

    def _reward_style_similar(self, play=False):
        feet_height = self.rigid_state[:, self.feet_indices - 1, 2] - 0.048

        self.buffer_left = torch.roll(self.buffer_left, shifts=-1, dims=1)
        self.buffer_left[:, -1] = feet_height[:, 0]
        self.buffer_right = torch.roll(self.buffer_right, shifts=-1, dims=1)
        self.buffer_right[:, -1] = feet_height[:, 1]

        # 计算FFT
        X_l = torch.fft.fft(self.buffer_left, dim=1)
        X_r = torch.fft.fft(self.buffer_right, dim=1)
        cross_power = X_l * torch.conj(X_r)
        R = torch.fft.ifft(cross_power, dim=1)
        correlation_value = R[:, self.tau_samples].real

        # 计算幅度谱差
        magnitude_spectrum_l = torch.abs(X_l)
        magnitude_spectrum_r = torch.abs(X_r)
        diff_magnitude = torch.mean(
            (magnitude_spectrum_l - magnitude_spectrum_r) ** 2, dim=1
        )

        # 计算频域奖励
        reward_frequency = self.alpha * correlation_value - self.beta * diff_magnitude
        if play:
            return correlation_value, diff_magnitude, reward_frequency
        else:
            return reward_frequency


    # ========task reward=========
    def _reward_lin_vel_track(self):
        error = self.commands[:, :2] - self.base_lin_vel[:, :2]
        _rew = torch.exp(-torch.norm(error, p=2, dim=1) / 0.2)
        return _rew

    def _reward_ang_vel_track(self):
        error = self.commands[:, 2:3] - self.base_ang_vel[:, 2:3]
        _rew = torch.exp(-torch.norm(error, p=2, dim=1) / 0.2)
        return _rew

    # ========behavior reward=====
    def _reward_body_height_track(self):
        error = self.root_states[:, 2:3] - (
            self.cfg.rewards.base_height_target + self.h_t
        )
        _rew = torch.exp(-torch.norm(error, p=2, dim=1)/0.01)
        return _rew

    def _reward_foot_swing_track(self, play=False):
        feet_height = self.rigid_state[:, self.feet_indices - 1, 2] - 0.048
        lt1 = self.l_t_1.squeeze()
        lt2 = self.l_t_2.squeeze()
        err_1 = feet_height[:, 0] - lt1
        err_2 = feet_height[:, 1] - lt2
        # print(m_cdf.size())
        # print((torch.norm(err_1, p=2,dim=1)).size())
        eep_1 = self._negsqrd_exp(torch.abs(err_1), 0.03)
        eep_2 = self._negsqrd_exp(torch.abs(err_2), 0.03)
        # _rew_1 = (1 - self.CDF_1).squeeze(1) * (eep_1)
        # _rew_2 = (1 - self.CDF_2).squeeze(1) * (eep_2)
        _rew_1 = eep_1
        _rew_1[lt1<0.005] = 0.3*torch.ones_like(_rew_1)[lt1<0.005]
        _rew_2 = eep_2
        _rew_2[lt2<0.005] = 0.3*torch.ones_like(_rew_2)[lt2<0.005]
        
        _rew = (_rew_1 + _rew_2) / 2
        if play:
            return (
                feet_height[:, 1:2],
                self.l_t_2,
                torch.abs(err_2),
                (1 - self.CDF_2).squeeze(1),
                eep_2,
                _rew_2,
            )
        else:
            return _rew

    def _reward_contact_swing_track(self, play=False):
        foot_contact_force = self.contact_forces[:, self.feet_indices, :]
        foot_velocity = self.rigid_state[:, self.feet_indices, 7:9]
        _r_fcf_1 = (1 - self.CDF_1) * (
            1 - torch.exp(torch.norm(foot_contact_force[:, 0:1, :], p=2, dim=2) / 100)
        )
        _r_fcf_2 = (1 - self.CDF_2) * (
            1 - torch.exp(torch.norm(foot_contact_force[:, 1:2, :], p=2, dim=2) / 100)
        )
        _r_fw_1 = self.CDF_1 * (
            1 - torch.exp(torch.norm(foot_velocity[:, 0:1, :], p=2, dim=2) / 20)
        )
        _r_fw_2 = self.CDF_2 * (
            1 - torch.exp(torch.norm(foot_velocity[:, 1:2, :], p=2, dim=2) / 20)
        )
        _rew = (-_r_fcf_1 * 0 - _r_fcf_2 * 0 + _r_fw_1 + _r_fw_2).squeeze()
        if play:
            return (
                torch.norm(foot_contact_force[:, 0:1, :], p=2, dim=2) / 500,
                1
                - torch.exp(
                    torch.norm(foot_contact_force[:, 0:1, :], p=2, dim=2) / 500
                ),
                torch.norm(foot_velocity[:, 0:1, :], p=2, dim=2) / 50,
                1 - torch.exp(torch.norm(foot_velocity[:, 1:2, :], p=2, dim=2) / 50),
            )
        else:
            return _rew

    # =======regularization reward=======
    def _reward_rp_ang_vel(self):
        _rew = torch.norm(self.base_ang_vel[:, :2], p=2, dim=1)
        return _rew

    def _reward_vertical_body_movement(self):
        _rew = torch.norm(self.base_lin_vel[:, 2:3], p=2, dim=1)
        return _rew

    def _reward_feet_slip(self):
        foot_velocity = self.rigid_state[:, self.feet_indices, 7:9]
        _r_fs_1 = torch.exp(-torch.norm(foot_velocity[:, 0, :], p=2, dim=1))
        _r_fs_2 = torch.exp(-torch.norm(foot_velocity[:, 1, :], p=2, dim=1))
        _rew = 1 - (_r_fs_1 + _r_fs_2)
        return _rew

    def _reward_action_rate(self):
        err = self.last_actions - self.actions
        _rew = torch.norm(err, p=2, dim=1)
        return _rew


    def _reward_joint_torque(self):
        _rew = torch.norm(self.torques, p=2, dim=1)
        return _rew

    def _reward_joint_acc(self):
        _rew = torch.norm((self.last_dof_vel - self.dof_vel) / self.dt, p=2, dim=1)
        return _rew

    def _reward_hip_joint_deviation(self):
        diff = self.dof_pos - self.ref_dof_pos
        _scale_l = [0.1, 2, 2, 0.2, 0.2, 2] * (2)
        _scale = torch.tensor(
            _scale_l, dtype=torch.float, device=self.device, requires_grad=False
        )
        _rew = torch.exp(-6 * torch.norm(diff[:, :] * _scale, dim=1))
        return _rew

    def _reward_feet_symmetry(self):
        foot_pos = self.rigid_state[:, self.feet_indices, :3]
        xz_indices = [0, 2]
        foot_dist = torch.norm(
            foot_pos[:, 0, xz_indices] - foot_pos[:, 1, xz_indices], dim=1
        )
        standing_mask = self.standing_command_mask == 1
        _rew = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        _rew[standing_mask] = foot_dist[standing_mask]
        return _rew

    def C_fun(self, phi_t_i, sigma):
        normal_dist = Normal(loc=0, scale=1)

        # Calculate the components of the function
        term1 = normal_dist.cdf(phi_t_i / sigma) * (
            1 - normal_dist.cdf((phi_t_i - 0.5) / sigma)
        )
        term2 = normal_dist.cdf((phi_t_i - 1) / sigma) * (
            1 - normal_dist.cdf((phi_t_i - 1.5) / sigma)
        )
        return term1 + term2

    # ================Standing & Walking===========================
    # ===base velocity=========
    def _reward_tracking_lin_vel(self):
        # Reward tracking linear velocity command in world frame
        error = self.commands[:, :2] - self.base_lin_vel[:, :2]
        error *= 1.0 / (1.0 + torch.abs(self.commands[:, :2]))
        return self._negsqrd_exp(error, a=1.0).sum(dim=1)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / 0.25)

    def _reward_base_z_vel(self):
        # Reward tracking linear velocity command in world frame
        error = self.base_lin_vel[:, 2]
        return self._negsqrd_exp(error, a=1.0)

    def _reward_low_speed(self):
        """
        Rewards or penalizes the robot based on its speed relative to the commanded speed.
        This function checks if the robot is moving too slow, too fast, or at the desired speed,
        and if the movement direction matches the command.
        """
        # Calculate the absolute value of speed and command for comparison
        absolute_speed = torch.abs(self.base_lin_vel[:, 0])
        absolute_command = torch.abs(self.commands[:, 0])

        # Define speed criteria for desired range
        speed_too_low = absolute_speed < 0.5 * absolute_command
        speed_too_high = absolute_speed > 1.2 * absolute_command
        speed_desired = ~(speed_too_low | speed_too_high)

        # Check if the speed and command directions are mismatched
        sign_mismatch = torch.sign(self.base_lin_vel[:, 0]) != torch.sign(
            self.commands[:, 0]
        )

        # Initialize reward tensor
        reward = torch.zeros_like(self.base_lin_vel[:, 0])

        # Assign rewards based on conditions
        # Speed too low
        reward[speed_too_low] = -1.0
        # Speed too high
        reward[speed_too_high] = 0.0
        # Speed within desired range
        reward[speed_desired] = 1.2
        # Sign mismatch has the highest priority
        reward[sign_mismatch] = -2.0
        return reward * (self.commands[:, 0].abs() > 0.1)

    # ===base pose=============
    def _reward_roll_pitch_orient(self):
        #
        _rew = torch.exp(-8 * torch.norm(self.base_euler_xyz[:, :2], dim=1))
        return _rew * (1 - self.dist_norm)

    def _reward_base_height(self):
        # Penalize base height
        aa = self.root_states[:, 2]
        bb = self.cfg.rewards.base_height_target
        # bb =(self.cfg.rewards.base_height_target+self.command_height).squeeze(-1)
        # print(aa.size())
        # print(bb.size())
        return torch.exp(-50 * torch.square(aa / bb - 1)) * (1 - self.dist_norm)

    # ===base acc==============
    def _reward_base_acc(self):
        rew = torch.exp(-torch.sum(torch.abs(self.root_acc_base), dim=1))
        return rew * (1 - self.dist_norm)

    # ===action smooth=========
    def _reward_action_smoothness(self):
        """
        Encourages smoothness in the robot's actions by penalizing large differences between consecutive actions.
        This is important for achieving fluid motion and reducing mechanical stress.
        """
        term_1 = torch.sum(torch.square(self.last_actions - self.actions), dim=1)
        term_2 = torch.sum(
            torch.square(self.actions + self.last_last_actions - 2 * self.last_actions),
            dim=1,
        )
        term_3 = torch.sum(torch.abs(self.actions), dim=1)
        return (0.2 * term_1 +120 * term_2 + 0.05 * term_3)

    def _reward_torques(self):
        """
        Penalizes the use of high torques in the robot's joints. Encourages efficient movement by minimizing
        the necessary force exerted by the motors.
        """
        return (torch.sum(torch.square(self.torques), dim=1)) * (1 - self.dist_norm)

    def _reward_dof_vel(self):
        """
        Penalizes high velocities at the degrees of freedom (DOF) of the robot. This encourages smoother and
        more controlled movements.
        """
        return (torch.sum(torch.square(self.dof_vel), dim=1)) * (1 - self.dist_norm)

    def _reward_dof_acc(self):
        """
        Penalizes high accelerations at the robot's degrees of freedom (DOF). This is important for ensuring
        smooth and stable motion, reducing wear on the robot's mechanical parts.
        """
        return (
            torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
        ) * (1 - self.dist_norm)

    # ===feet motion===========
    def _reward_feet_contact(self, play=False):
        # Penalize feet contact
        _rew = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        contact = self.contact_forces[:, self.feet_indices, 2] > 0
        # self.contact_queue存放了最近200次(0.2s)的contact
        self.contact_queue = torch.roll(self.contact_queue, 1, dims=0)  # 右滚
        self.contact_queue[:, 0, :] = contact
        sum_contact = torch.sum(self.contact_queue, dim=2)

        # has_zero_contact = sum_contact == 0
        # has_true_in_row_zero = torch.any(has_zero_contact, dim=1)

        has_single_contact = sum_contact == 1
        has_true_in_row = torch.any(has_single_contact, dim=1)

        has_double_contact = sum_contact == 2
        has_true_in_row_double = torch.any(has_double_contact, dim=1)

        pitch_mask = (torch.abs(self.base_euler_xyz[:, 1])) > (5 / 180 * torch.pi)
        # print(pitch_mask.size())
        # print(dist[0],self.disturbance[0, 0, :])

        # if not standing command
        mask_walk_single_contact = (self.standing_command_mask != 1) & (
            has_true_in_row == 1
        )
        _rew[mask_walk_single_contact] = 1
        # if standing command
        # mask_stand_zero_contact = (self.standing_command_mask == 1) & (
        #     has_true_in_row_zero == 1
        # )
        mask_stand_double_contact = (self.standing_command_mask == 1) & (
            has_true_in_row_double == 1
        )
        mask_stand_single_contact = (self.standing_command_mask == 1) & (
            has_true_in_row == 1
        )
        _rew[mask_stand_single_contact] = (
            0.2 * self.dist_norm[mask_stand_single_contact]
            + 0.1 * pitch_mask[mask_stand_single_contact]
        )
        _rew[mask_stand_double_contact] = 1
        # _rew[mask_stand_zero_contact] = -1

        if play:
            return (
                _rew,
                contact,
                has_true_in_row,
                has_single_contact[:, 0],
                has_true_in_row_double,
            )
        else:
            return _rew

    def _reward_feet_airtime(self, play=False):
        """
        通过在每次脚着地时施加-0.4的惩罚来规范踏步频率，
        这可以通过一个积极的奖励成分来抵消，即足部腾空后的秒数(腾空时间)。
        如果没有这个组件，学习到的控制器倾向于采用步进频率在风格上太大的步态，
        这可能是由于这些频率对应于可能的局部最小值。这个分量在站立时是恒定的。
        """
        _rew = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        
        # else
        contact = self.contact_forces[:, self.feet_indices, 2] > 0
        self.contact_filt = torch.logical_and(
            contact, (self.feet_air_time > 0)
        )  # 是否第一次接触，如果接触就为1，不接触就为0
        # print(self.contact_filt[0,:],(self.feet_air_time * self.contact_filt)[0,:])
        self.feet_air_time += self.dt
        T = (1/self.f_t).repeat_interleave(2,dim=1)
        air_time_reward: torch.Tensor =(T-self.feet_air_time) * self.contact_filt
        air_time_reward = air_time_reward.sum(dim=1)/2
        self.feet_air_time *= ~self.last_contacts  # 不接触的话就持续计数，接触就清零
        self.last_contacts = contact  # 更新上一帧的接触情况
        _rew = torch.exp(air_time_reward) - 1
        # print(_rew)
        # if standing command
        if torch.any(self.standing_command_mask == 1):
            _rew[self.standing_command_mask == 1] = 1
        if play:
            return (
                self.standing_command_mask,
                contact,
                self.contact_filt,
                self.feet_air_time,
                air_time_reward,
                _rew,
            )
        else:
            return _rew

    def _reward_feet_contact_forces(self):
        return torch.sum(
            (
                torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
                - self.cfg.rewards.max_contact_force
            ).clip(0, 400),
            dim=1,
        )

    def _reward_feet_swing_height(self):
        _rew = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        # if standing command
        if torch.any(self.standing_command_mask == 1):
            _rew[self.standing_command_mask == 1] = 1
        # else
        contact = self.contact_forces[:, self.feet_indices, 2] > 0
        self.contact_filt = torch.logical_and(
            contact, (self.feet_air_time > 0)
        )  # 是否第一次接触，如果接触就为1，不接触就为0
        self.feet_air_time += self.dt
        self.feet_air_time *= ~contact  # 不接触的话就持续计数，接触就清零
        feet_height = self.rigid_state[:, self.feet_indices, 2] - 0.031
        feet_height_clip = torch.clip(feet_height - self.l_t_all, max=0)
        # print(self.feet_air_time.size(),feet_height.size())
        # print(feet_height[0,:])
        feet_height_rew = torch.exp(10 * feet_height_clip * (self.feet_air_time > 0))

        # print(feet_height_rew[0,:])
        if torch.any(self.standing_command_mask == 0):
            _rew[self.standing_command_mask == 0] = feet_height_rew[
                self.standing_command_mask == 0
            ].sum(dim=1)

        return _rew

    def _reward_foot_slip(self):
        """
        Calculates the reward for minimizing foot slip. The reward is based on the contact forces
        and the speed of the feet. A contact threshold is used to determine if the foot is in contact
        with the ground. The speed of the foot is calculated and scaled by the contact condition.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.0
        foot_speed_norm = torch.norm(self.rigid_state[:, self.feet_indices, 7:9], dim=2)
        rew = torch.sqrt(foot_speed_norm)
        rew *= contact
        return torch.sum(rew, dim=1)

    def _reward_boundary(self):
        _rew = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        mask = ~self.is_point_in_quadrilateral(
            self.quadrilateral_point, self.root_states[:, :2]
        )
        # print(mask[0])
        return mask

    # ===pos===================
    def _reward_joint_pos(self):
        diff = self.dof_pos - self.ref_dof_pos
        _scale_l = [0., 1, 1, 0.0, 0.0, 1] * (2)
        _scale = torch.tensor(
            _scale_l, dtype=torch.float, device=self.device, requires_grad=False
        )
        r = torch.exp(-torch.sum(torch.abs(diff[:, :] * _scale),dim=1))
        return r

    def _reward_default_joint_pos(self):
        """
        Calculates the reward for keeping joint positions close to default positions, with a focus
        on penalizing deviation in yaw and roll directions. Excludes yaw and roll from the main penalty.
        """
        joint_diff = self.dof_pos - self.default_joint_pd_target
        left_yaw_roll = joint_diff[:, 1:3]
        right_yaw_roll = joint_diff[:, 7:9]
        yaw_roll = torch.norm(left_yaw_roll, dim=1) + torch.norm(right_yaw_roll, dim=1)
        yaw_roll = torch.clamp(yaw_roll - 0.1, 0, 50)
        return torch.exp(-yaw_roll * 50) - 0.01 * torch.norm(joint_diff, dim=1)

    def _reward_feet_distance(self):
        """
        Calculates the reward based on the distance between the feet. Penilize feet get close to each other or too far away.
        """

        foot_pos_world = self.rigid_state[:, self.feet_indices, 0:3]
        foot_pos_base_left = quat_rotate_inverse(
            self.base_quat, foot_pos_world[:, 0, :] - self.base_pos
        )
        foot_pos_base_right = quat_rotate_inverse(
            self.base_quat, foot_pos_world[:, 1, :] - self.base_pos
        )
        foot_dist = torch.norm(
            foot_pos_base_left[:, 1:2] - foot_pos_base_right[:, 1:2], dim=1
        )

        # print(foot_dist)
        fd = self.cfg.rewards.min_dist_fe
        max_df = self.cfg.rewards.max_dist_fe
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.0)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        return (
            torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)
        ) / 2

    def _reward_feet_distance_2(self):
        """
        Calculates the reward based on the distance between the feet. Penilize feet get close to each other or too far away.
        """

        foot_pos_world = self.rigid_state[:, self.feet_indices - 1, 0:3]
        foot_pos_base_left = quat_rotate_inverse(
            self.base_quat, foot_pos_world[:, 0, :] - self.base_pos
        )
        foot_pos_base_right = quat_rotate_inverse(
            self.base_quat, foot_pos_world[:, 1, :] - self.base_pos
        )
        foot_dist = torch.norm(
            foot_pos_base_left[:, 1:2] - foot_pos_base_right[:, 1:2], dim=1
        )

        # print(foot_dist)
        fd = self.cfg.rewards.min_dist_fe
        max_df = self.cfg.rewards.max_dist_fe
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.0)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        return (
            torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)
        ) / 2

    def _reward_feet_distance_x(self):
        foot_pos_world = self.rigid_state[:, self.feet_indices - 1, 0:3]
        foot_pos_base_left = quat_rotate_inverse(
            self.base_quat, foot_pos_world[:, 0, :] - self.base_pos
        )
        foot_pos_base_right = quat_rotate_inverse(
            self.base_quat, foot_pos_world[:, 1, :] - self.base_pos
        )
        foot_dist = torch.abs(
            foot_pos_base_left[:, 0] - foot_pos_base_right[:, 0]
        )
        rew = self._negsqrd_exp(foot_dist,0.05)
        return rew
            
    def _reward_knee_distance(self):
        """
        Calculates the reward based on the distance between the knee of the humanoid.
        """
        foot_pos_world = self.rigid_state[:, self.knee_indices, 0:3]
        foot_pos_base_left = quat_rotate_inverse(
            self.base_quat, foot_pos_world[:, 0, :] - self.base_pos
        )
        foot_pos_base_right = quat_rotate_inverse(
            self.base_quat, foot_pos_world[:, 1, :] - self.base_pos
        )
        foot_dist = torch.norm(
            foot_pos_base_left[:, 1:2] - foot_pos_base_right[:, 1:2], dim=1
        )

        fd = self.cfg.rewards.min_dist_kn
        max_df = self.cfg.rewards.max_dist_kn
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.0)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        return (
            torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)
        ) / 2

    def _reward_feet_orient(self):
        _rew = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        # print(self.foot_orient.size())
        feet_orient_l = get_euler_xyz_tensor(self.foot_orient[:, 0, :])
        feet_orient_r = get_euler_xyz_tensor(self.foot_orient[:, 1, :])
        # print(feet_orient_l.size())
        feet_orient = torch.stack((feet_orient_l, feet_orient_r), dim=2)
        # print(feet_orient.size())
        # base_orient = self.base_euler_xyz.unsqueeze(-1)
        # print(base_orient.size())
        diff_3 = torch.abs(feet_orient)
        diff_2 = torch.abs(feet_orient)[:, :2, :]
        # print(diff_3.size())
        # print(diff_2.size())
        cm_wz_mask = torch.abs(self.commands[:, 2]) > 0

        _rew = torch.exp(-3 * torch.sum(diff_3, dim=(1, 2)))

        # _rew[cm_wz_mask] = torch.exp(-3 * torch.sum(diff_2, dim=(1, 2)))[cm_wz_mask]

        # print(_rew.size())
        return _rew

    def _reward_feet_position(self):
        _rew = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        base_pos = self.root_states[:, :3]
        feet_pos_l = self.rigid_state[:, self.feet_indices[0], :3]
        # print(self.rigid_state.size())
        feet_pos_r = self.rigid_state[:, self.feet_indices[1], :3]
        feet_pos = torch.stack((feet_pos_l, feet_pos_r), dim=2)
        # print(base_pos[0, ...])
        # print(feet_pos[0, ...])
        aa = torch.tensor(
            [
                [0.01105107, 0.01105107],
                [-0.09949592, 0.09949592],
                [0.50925862, 0.50925862],
            ],
            device="cuda:0",
        )  # scripts/test/test_mujoco_feet.py

        # print(aa)
        diff = base_pos.unsqueeze(-1) - feet_pos - aa
        # print(diff[0,...])
        if torch.any(self.standing_command_mask == 1):
            _rew[self.standing_command_mask == 1] = torch.exp(
                -0.1
                * torch.sum(
                    torch.norm(
                        diff,
                        dim=2,
                    ),
                    dim=1,
                )
            )[self.standing_command_mask == 1]
        # print(_rew)
        return _rew

    def is_point_in_quadrilateral(
        self, quadrilateral_point: torch.Tensor, root_states: torch.Tensor
    ) -> torch.Tensor:
        """
        判断每个环境中的root_states是否在对应的四边形内部。

        参数：
            quadrilateral_point: Tensor，形状 [num_envs, 4, 2]，
                                存放每个环境中四边形的四个顶点，
                                点的顺序依次为左上、左下、右下、右上。
            root_states: Tensor，形状 [num_envs, 2]，每个环境中待检测的点。

        返回：
            inside: Bool型Tensor，形状 [num_envs]，每个元素表示对应环境中
                    root_states 点是否在四边形内部（True表示在内部，False表示不在内部）。
        """
        # 将四边形各个顶点沿第1维滚动（即对每个环境，将每个点的“下一个”顶点计算出来）
        quadr_next = torch.roll(
            quadrilateral_point, shifts=-1, dims=1
        )  # 形状 [num_envs, 4, 2]

        # 计算每条边的向量：edge = next_vertex - current_vertex
        edge_vectors = quadr_next - quadrilateral_point  # 形状 [num_envs, 4, 2]

        # 计算从每个顶点到待判断点的向量：相对向量
        # 注意：root_states的形状 [num_envs,2]扩展成 [num_envs,1,2]，与四边形顶点相减
        rel_vectors = (
            root_states.unsqueeze(1) - quadrilateral_point
        )  # 形状 [num_envs, 4, 2]

        # 计算二维叉积：对于二维向量 (a, b) 和 (c, d)，叉积的标量为 a*d - b*c
        cross_products = (
            edge_vectors[..., 0] * rel_vectors[..., 1]
            - edge_vectors[..., 1] * rel_vectors[..., 0]
        )  # 形状 [num_envs, 4]

        # 判断：如果在某个环境中，所有边的叉积均大于等于0或均小于等于0，则点在内部
        inside = torch.all(cross_products >= 0, dim=1) | torch.all(
            cross_products <= 0, dim=1
        )
        return inside

    def _visualization(self):
        # print("_visualization")
        self.gym.clear_lines(self.viewer)
        self._draw_world_velocity_arrow_vis()
        self._draw_step_vis()

    def _draw_world_velocity_arrow_vis(self):
        """Draws linear / angular velocity arrow for humanoid
        Angular velocity is described by axis-angle representation"""
        origins = self.base_pos + quat_apply(
            self.base_quat,
            torch.tensor([0.0, 0.0, 0.5]).repeat(self.num_envs, 1).to(self.device),
        )
        lin_vel_command = (
            torch.cat(
                (
                    self.commands[:, :2],
                    torch.zeros((self.num_envs, 1), device=self.device),
                ),
                dim=1,
            )
            / 2
        )
        # ang_vel_command = quat_apply(self.base_quat, torch.cat((torch.zeros((self.num_envs,2), device=self.device), self.commands[:, 2:3]), dim=1)/5)
        for i in range(self.num_envs):
            lin_vel_arrow = VelCommandGeometry(
                origins[i], lin_vel_command[i], color=(0, 1, 0)
            )
            # ang_vel_arrow = VelCommandGeometry(origins[i], ang_vel_command[i], color=(0,1,0))
            gymutil.draw_lines(
                lin_vel_arrow, self.gym, self.viewer, self.envs[i], pose=None
            )
            # gymutil.draw_lines(ang_vel_arrow, self.gym, self.viewer, self.envs[i], pose=None)

    def _draw_step_vis(self):
        """Draws step command for humanoid"""
        for i in range(self.num_envs):
            right_step_command = FootStepGeometry(
                self.step_pos[i, 0, :2],
                self.step_pos[i, 0, 2],
                color=(1, 0, 0),
            )  # Right foot: Red
            left_step_command = FootStepGeometry(
                self.step_pos[i, 1, :2],
                self.step_pos[i, 1, 2],
                color=(0, 0, 1),
            )  # Left foot: Blue
            # gymutil.draw_lines(
            #     left_step_command, self.gym, self.viewer, self.envs[i], pose=None
            # )
            # gymutil.draw_lines(
            #     right_step_command, self.gym, self.viewer, self.envs[i], pose=None
            # )

            verts = np.empty((1, 2), dtype=gymapi.Vec3.dtype)
            verts[0][0] = (
                self.step_pos[i, 0, 0] + 0.095,
                self.step_pos[i, 0, 1] + 0.04,
                1e-4,
            )
            verts[0][1] = (
                self.step_pos[i, 1, 0] + 0.095,
                self.step_pos[i, 1, 1] - 0.04,
                1e-4,
            )

            colors = np.empty(1, dtype=gymapi.Vec3.dtype)
            colors[0] = (0, 1, 0)
            self.gym.add_lines(self.viewer, self.envs[i], 1, verts, colors)

            verts = np.empty((1, 2), dtype=gymapi.Vec3.dtype)
            verts[0][0] = (
                self.step_pos[i, 0, 0] - 0.06,
                self.step_pos[i, 0, 1] + 0.04,
                1e-4,
            )
            verts[0][1] = (
                self.step_pos[i, 1, 0] - 0.06,
                self.step_pos[i, 1, 1] - 0.04,
                1e-4,
            )
            colors = np.empty(1, dtype=gymapi.Vec3.dtype)
            colors[0] = (0, 1, 0)
            self.gym.add_lines(self.viewer, self.envs[i], 1, verts, colors)

    # ================================================ Rewards useless  ================================================== #
    def _reward_feet_clearance(self):
        # Compute feet contact mask
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0

        # Get the z-position of the feet and compute the change in z-position
        feet_z = self.rigid_state[:, self.feet_indices, 2] - 0.048
        delta_z = feet_z - self.last_feet_z
        self.feet_height += delta_z
        self.last_feet_z = feet_z

        # Compute swing mask
        swing_mask = 1 - self._get_gait_phase()

        # feet height should be closed to target feet height at the peak
        rew_pos = (
            torch.abs(self.feet_height - self.cfg.rewards.target_feet_height) < 0.01
        )
        rew_pos = torch.sum(rew_pos * swing_mask, dim=1)
        self.feet_height *= ~contact
        return rew_pos

    def _reward_feet_contact_number(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        stance_mask = self._get_gait_phase()
        reward = torch.where(contact == self.stance_mask, 1.0, -50.0)
        return torch.mean(reward, dim=1)

    def _reward_ankle_roll_angle_zero(self):
        ankle_roll_indices = [5, 11]
        ankle_roll = self.dof_pos[:, ankle_roll_indices]
        rew = torch.exp(-torch.sum(torch.abs(ankle_roll), dim=1))
        return rew

    def _reward_hip_roll_angle_zero(self):
        hip_roll_indices = [1, 7]
        hip_roll = self.dof_pos[:, hip_roll_indices]
        rew = torch.exp(-torch.sum(torch.abs(hip_roll), dim=1))
        return rew

    def _reward_termination(self):
        # Terminal reward / penalty
        return -(self.reset_buf * ~self.time_out_buf).float()

    # * ######################### HELPER FUNCTIONS ############################## * #

    def _neg_exp(self, x, a=1):
        """shorthand helper for negative exponential e^(-x/a)
        a: range of x
        """
        return torch.exp(-(x / a) / 0.25)

    def _negsqrd_exp(self, x, a=1):
        """shorthand helper for negative squared exponential e^(-(x/a)^2)
        a: range of x
        """
        return torch.exp(-torch.square(x / a) / 0.25)


import math


def calculate_C(alpha, beta, sigma, x):
    """
    计算 C_{\alpha,\beta,\sigma}(x) 的值。

    参数:
    alpha -- 公式中的 \alpha
    beta -- 公式中的 \beta
    sigma -- 公式中的 \sigma
    x -- 公式中的 x

    返回:
    C_{\alpha,\beta,\sigma}(x) 的计算结果
    """
    # 计算括号内的部分
    term = (x / sigma) ** (2 * beta)

    # 计算整个表达式的值
    C = alpha / (term + 1)

    return C

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


from .pai_config_none_phase import PaiNonePhaseCfg

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi

import torch
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.math import (
    quat_apply_yaw,
    wrap_to_pi,
    torch_rand_sqrt_float,
    get_scale_shift,
    get_scale_shift,
    euler_from_quat,
    random_sample,
)

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os


# from humanoid.utils.terrain import  HumanoidTerrain
# from collections import deque
def get_euler_xyz_tensor(quat):
    r, p, w = get_euler_xyz(quat)
    # stack r, p, w in dim1
    euler_xyz = torch.stack((r, p, w), dim=1)
    euler_xyz[euler_xyz > np.pi] -= 2 * np.pi
    return euler_xyz


class PaiNonePhaseEnv(LeggedRobot):
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

    cfg: PaiNonePhaseCfg

    def __init__(
        self, cfg: PaiNonePhaseCfg, sim_params, physics_engine, sim_device, headless
    ):
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

        swing_mask = 1 - self._get_gait_phase()
        self.swing_mask = swing_mask * (1 - self.standing_command_mask.unsqueeze(1))
        self.stance_mask = 1 - self.swing_mask

        self.swing_mask_l = self.swing_mask[:, 0]
        self.swing_mask_r = self.swing_mask[:, 1]

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids_none_reset = (self.reset_buf == 0).nonzero(as_tuple=False).flatten()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.update_command_curriculum(env_ids_none_reset)

        self.reset_idx(env_ids)
        # print("reset_idx")
        self.compute_observations()  # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.slast_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_last_joint_pos_target[:] = self.last_joint_pos_target[:]
        self.last_joint_pos_target[:] = self.joint_pos_target[:]

        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_last_actions[:] = torch.clone(self.last_actions[:])
        self.base_orn_rp[:] = self.get_body_orientation()

        # print(torch.mean(torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1),dim=-1))

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

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

    def _randomize_rigid_body_props(self, env_ids, cfg: PaiNonePhaseCfg):

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

    def _disturbance_robots(self):
        disturbance = torch_rand_float(
            self.command_ranges["disturbance_range"][0],
            self.command_ranges["disturbance_range"][1],
            (self.num_envs, 3),
            device=self.device,
        )
        self.disturbance[:, 0, :] = disturbance / (
            self.cfg.domain_rand.disturbance_range[1]
            - self.cfg.domain_rand.disturbance_range[0]
        )
        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            forceTensor=gymtorch.unwrap_tensor(self.disturbance),
            space=gymapi.CoordinateSpace.LOCAL_SPACE,
        )

        ...

    def _resample_commands(self, env_ids):
        """Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.command_catrgories[env_ids] = torch.randint(
            # 0,
            1,
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
        # self.commands[env_ids, :2] *= (
        #     torch.norm(self.commands[env_ids, :2], dim=1) > 0.1
        # ).unsqueeze(1)

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
        self._reward_joint_pos()
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
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(
            self.num_envs,
            self.cfg.commands.num_commands,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )  # x vel, y vel, yaw vel, heading
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
            int(0.2 / self.dt),
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
        self._reward_feet_position()

        if self.cfg.init_state.reset_mode == "reset_to_range":
            # self.dof_pos_range = torch.zeros(self.num_dof, 2,
            #                                 dtype=torch.float,
            #                                 device=self.device,
            #                                 requires_grad=False)
            # self.dof_vel_range = torch.zeros(self.num_dof, 2,
            #                                 dtype=torch.float,
            #                                 device=self.device,
            #                                 requires_grad=False)

            # for joint, vals in self.cfg.init_state.dof_pos_range.items():
            #     for i in range(self.num_dof):
            #         if joint in self.dof_names[i]:
            #             self.dof_pos_range[i, :] = to_torch(vals)

            # for joint, vals in self.cfg.init_state.dof_vel_range.items():
            #     for i in range(self.num_dof):
            #         if joint in self.dof_names[i]:
            #             self.dof_vel_range[i, :] = to_torch(vals)

            self.root_pos_range = torch.tensor(
                self.cfg.init_state.root_pos_range,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
            self.root_vel_range = torch.tensor(
                self.cfg.init_state.root_vel_range,
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
            # print("No env to reset")
            return
        # else:
        # print("Resetting envs: ", env_ids)
        super().reset_idx(env_ids)
        self.last_last_actions[env_ids] = 0.0
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)
        self.random_half_phase[0, env_ids] = torch.pi * torch.randint(
            0, 2, (len(env_ids),), device=self.device
        )

    def _reset_root_states(self, env_ids):
        """Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(
                -1.0, 1.0, (len(env_ids), 2), device=self.device
            )  # xy position within 1m of the center
        else:
            if self.cfg.init_state.reset_mode == "reset_to_basic":
                self.root_states[env_ids] = self.base_init_state
            elif self.cfg.init_state.reset_mode == "reset_to_range":
                # base states
                # print("reset_to_range")
                random_com_pos = random_sample(
                    env_ids,
                    self.root_pos_range[:, 0],
                    self.root_pos_range[:, 1],
                    device=self.device,
                )
                self.root_states[env_ids, 0:7] = torch.cat(
                    (
                        random_com_pos[:, 0:3],
                        quat_from_euler_xyz(
                            random_com_pos[:, 3],
                            random_com_pos[:, 4],
                            random_com_pos[:, 5],
                        ),
                    ),
                    1,
                )
                self.root_states[env_ids, 7:13] = random_sample(
                    env_ids,
                    self.root_vel_range[:, 0],
                    self.root_vel_range[:, 1],
                    device=self.device,
                )
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(
            -0.5, 0.5, (len(env_ids), 6), device=self.device
        )  # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

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
        # sin_pos_l[sin_pos_l > 0] = 0
        # sin_pos_l += 0.05

        sin_pos_r = sin_pos.clone() - self.cfg.rewards.bias
        sin_pos_r = (
            sin_pos_r / torch.sqrt(sin_pos_r**2.0 + eps**2.0) + self.cfg.rewards.y_bias
        )
        # sin_pos_r[sin_pos_r < 0] = 0
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
        # print(torch.mean(self.episode_sums["tracking_lin_vel"][env_ids])
        #     / self.max_episode_length)
        # print(self.reward_scales["tracking_lin_vel"])
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        # a = (
        #     torch.mean(self.episode_sums["tracking_lin_vel"][env_ids])
        #     / self.max_episode_length
        # )
        # b = (
        #     torch.mean(self.episode_sums["tracking_lin_vel"][...])
        #     / self.max_episode_length
        # )
        # c = a > 0.8 * self.reward_scales["tracking_lin_vel"]
        # print(a*100,b*100,c)
        if (
            torch.mean(self.episode_sums["tracking_lin_vel"])
            / self.max_episode_length
            > 0.5 * self.reward_scales["tracking_lin_vel"]
            # > 1.43 * self.reward_scales["tracking_lin_vel"]
        ):
            mm = (
                self.cfg.commands.ranges.lin_vel_x[1]
                - self.cfg.commands.ranges.lin_vel_x[0]
            )
            self.command_ranges["lin_vel_x"][0] = np.clip(
                self.command_ranges["lin_vel_x"][0] - 0.0008 * mm,
                -self.cfg.commands.max_curriculum_x[0],
                0.0,
            )
            self.command_ranges["lin_vel_x"][1] = np.clip(
                self.command_ranges["lin_vel_x"][1] + 0.0008 * mm,
                0.0,
                self.cfg.commands.max_curriculum_x[1],
            )
            mm_d = (
                self.cfg.commands.ranges.disturbance_range[1]
                - self.cfg.commands.ranges.disturbance_range[0]
            )
            self.command_ranges["disturbance_range"][0] = np.clip(
                self.command_ranges["disturbance_range"][0] - 0.005 * mm_d,
                -self.cfg.commands.max_disturbance_range[0],
                0.0,
            )
            self.command_ranges["disturbance_range"][1] = np.clip(
                self.command_ranges["disturbance_range"][1] + 0.005 * mm_d,
                0.0,
                self.cfg.commands.max_disturbance_range[1],
            )

            self.cmd_level += 1
            self.cmd_level_old = self.cmd_level

        if (
            torch.mean(self.episode_sums["tracking_lin_vel"])
            / self.max_episode_length
            < 0.4 * self.reward_scales["tracking_lin_vel"]
            # < 1.4 * self.reward_scales["tracking_lin_vel"]
        ) and self.cmd_level_old > 0:
            mm = (
                self.cfg.commands.ranges.lin_vel_x[1]
                - self.cfg.commands.ranges.lin_vel_x[0]
            )
            self.command_ranges["lin_vel_x"][0] = np.clip(
                self.command_ranges["lin_vel_x"][0] + 0.001 * mm,
                -self.cfg.commands.max_curriculum_x[0],
                0.0,
            )
            self.command_ranges["lin_vel_x"][1] = np.clip(
                self.command_ranges["lin_vel_x"][1] - 0.001 * mm,
                0.0,
                self.cfg.commands.max_curriculum_x[1],
            )
            mm_d = (
                self.cfg.commands.ranges.disturbance_range[1]
                - self.cfg.commands.ranges.disturbance_range[0]
            )
            self.command_ranges["disturbance_range"][0] = np.clip(
                self.command_ranges["disturbance_range"][0] + 0.001 * mm_d,
                -self.cfg.commands.max_disturbance_range[0],
                0.0,
            )
            self.command_ranges["disturbance_range"][1] = np.clip(
                self.command_ranges["disturbance_range"][1] - 0.001 * mm_d,
                0.0,
                self.cfg.commands.max_disturbance_range[1],
            )

            self.cmd_level -= 1
            self.cmd_level_old = self.cmd_level

    def compute_observations(self):
        """Computes observations"""
        phase = self._get_phase()
        # sin_pos = torch.sin(2 * 0 * phase).unsqueeze(1)
        # cos_pos = torch.cos(2 * 0 * phase).unsqueeze(1)
        sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        sin_pos_2 = sin_pos * sin_pos
        sin2_pos = torch.sin(4 * torch.pi * phase).unsqueeze(1)
        cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1)
        cos2_pos = torch.cos(4 * torch.pi * phase).unsqueeze(1)
        cos_pos_2 = cos_pos * cos_pos

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

        if self.cfg.env.obs_with_base_lin_vel:  # DGZ
            self.obs_buf = torch.cat(
                (
                    self.base_ang_vel * self.obs_scales.ang_vel,  # 3
                    self.base_lin_vel * self.obs_scales.lin_vel,  # 3 #DGZ
                    self.projected_gravity,  # 3
                    self.commands[:, :3] * self.commands_scale,  # 3
                    self.standing_command_mask.unsqueeze(1),  # 1
                    sin_pos,  # 1
                    cos_pos,  # 1
                    (self.dof_pos - self.default_dof_pos)
                    * self.obs_scales.dof_pos,  # 12
                    self.dof_vel * self.obs_scales.dof_vel,  # 12
                    self.actions,  # 12
                ),
                dim=-1,
            )  # 51
        else:
            self.obs_buf = torch.cat(
                (
                    self.base_ang_vel * self.obs_scales.ang_vel,  # 3
                    self.projected_gravity,  # 3
                    self.commands[:, :3] * self.commands_scale,  # 3
                    self.standing_command_mask.unsqueeze(1),  # 1
                    sin_pos,  # 1
                    cos_pos,  # 1
                    (self.dof_pos - self.default_dof_pos)
                    * self.obs_scales.dof_pos,  # 12
                    self.dof_vel * self.obs_scales.dof_vel,  # 12
                    self.actions,  # 12
                ),
                dim=-1,
            )  # 48
            if torch.isnan(self.dof_vel).any():
                for i, row in enumerate(self.dof_vel):
                    if torch.isnan(row).any():
                        print(f"{i} {row}")
                raise ValueError("self.dof_vel contains NaN")
        self.privileged_obs_buf = torch.cat(
            (
                self.base_ang_vel * self.obs_scales.ang_vel,  # 3
                self.projected_gravity,  # 3
                self.commands[:, :3] * self.commands_scale,  # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,  # 12
                self.dof_vel * self.obs_scales.dof_vel,  # 12
                self.actions,  # 12
                sin_pos,  # 1
                cos_pos,  # 1
                self.base_lin_vel * self.obs_scales.lin_vel,  # 3
                (self.contact_forces.view(self.num_envs, -1) - contact_forces_shift)
                * contact_forces_scale,  # 13*3
                heights,  # 17*11
                diff,
                self.stance_mask,  # 2
                contact_mask,  # 2
                self.standing_command_mask.unsqueeze(1),  # 1
                self.disturbance[:, 0, :],  # 3
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

    # ================================================ Rewards None phase ================================================== #
    def _reward_joint_pos(self):
        # TODO 该奖励需要重构造
        # 当机器人站立时，机器人的关节位置应该是0
        # 当机器人行走时，只需要跟踪摆动腿的关节位置，
        #   支撑腿的关节位置应该是0，或是以一个较弱的约束跟踪到0

        # diff = (self.dof_pos - self.ref_dof_pos)[:, selected_columns]
        diff = self.dof_pos - self.ref_dof_pos
        # 当机器人行走时，只需要跟踪摆动腿的关节位置,跟随规划的位置
        # diff[self.swing_mask_l==0,:6] *= 0.2
        # diff[self.swing_mask_r==0,-6:] *= 0.2
        # 当机器人站立时，支撑腿的关节位置应该是0
        diff[self.standing_command_mask == 1, :] = (
            self.dof_pos[self.standing_command_mask == 1, :] * 1.0
        )

        selected_columns = [0, 3, 4, 6, 9, 10]
        # r = torch.exp(-6 * torch.norm(diff[:, selected_columns], dim=1))
        r = torch.exp(-5 * torch.norm(diff[:, :], dim=1))
        # r = torch.exp(-6 * torch.norm(diff[:, :], dim=1))
        return r

    def _reward_feet_contact_number(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        stance_mask = self._get_gait_phase()
        reward = torch.where(contact == self.stance_mask, 1.0, -3)
        return torch.mean(reward, dim=1)

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

    def _reward_feet_clearance_2(self):
        # Compute feet contact mask
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0

        # Get the z-position of the feet and compute the change in z-position
        feet_z = self.rigid_state[:, self.feet_indices - 1, 2] - 0.048
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

    def _reward_xy_velocity(self):
        # if standing command is 1, reward is exp(-5*(v_xy-c_xy))
        # if standing command is 0, reward is exp(-5*(v_xy-c_xy)^2)
        _rew = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        # print(standing_command_mask.size)
        vel_diff = self.base_lin_vel[:, :2] - self.commands[:, :2]
        # Check if there are any indices where standing_command_mask == 1
        if torch.any(self.standing_command_mask == 1):
            _rew[self.standing_command_mask == 1] = torch.exp(
                -4 * torch.norm(vel_diff[self.standing_command_mask == 1], dim=1)
            )

        # Check if there are any indices where standing_command_mask == 0
        if torch.any(self.standing_command_mask == 0):
            _rew[self.standing_command_mask == 0] = torch.exp(
                -4
                * torch.square(
                    torch.norm(vel_diff[self.standing_command_mask == 0], dim=1)
                )
            )
        return _rew

    def _reward_tracking_lin_vel(self):
        # Reward tracking linear velocity command in world frame
        error = self.commands[:, :2] - self.base_lin_vel[:, :2]
        error *= 1.0 / (1.0 + torch.abs(self.commands[:, :2]))
        return self._negsqrd_exp(error, a=1.0).sum(dim=1)

    def _reward_base_z_vel(self):
        # Reward tracking linear velocity command in world frame
        error = self.base_lin_vel[:, 2]
        return self._negsqrd_exp(error, a=1.0)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error * 4)

    def _reward_yaw_velocity(self):
        # Penalize yaw axis base orientation
        _rew = torch.exp(-8 * torch.abs(self.base_ang_vel[:, 2] - self.commands[:, 2]))
        return _rew

    def _reward_roll_pitch_orient(self):
        #
        _rew = torch.exp(-30 * torch.norm(self.base_euler_xyz[:, :2], dim=1))
        return _rew

    def _reward_base_height(self):
        # Penalize base height
        return torch.exp(
            -500
            * torch.square(
                self.root_states[:, 2] / self.cfg.rewards.base_height_target - 1
            )
        )
    def _reward_base_height_2(self):
        """
        Calculates the reward based on the robot's base height. Penalizes deviation from a target base height.
        The reward is computed based on the height difference between the robot's base and the average height 
        of its feet when they are in contact with the ground.
        """
        stance_mask = self._get_gait_phase()
        measured_foot_heights = torch.sum(
            self.rigid_state[:, self.feet_indices, 2] * stance_mask, dim=1) / torch.sum(stance_mask, dim=1)
        base_height = self.root_states[:, 2] - (measured_foot_heights)
        return torch.exp(-torch.abs(base_height - self.cfg.rewards.base_height_target) * 100)
    def _reward_feet_contact(self):
        # Penalize feet contact
        _rew = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )

        contact = self.contact_forces[:, self.feet_indices, 2] > 0
        # self.contact_queue存放了最近200次(0.2s)的contact
        self.contact_queue = torch.roll(self.contact_queue, 1, dims=0)  # 右滚
        self.contact_queue[:, 0, :] = contact
        sum_contact = torch.sum(self.contact_queue, dim=2)
        has_single_contact = sum_contact == 1
        has_true_in_row = torch.any(has_single_contact, dim=1)
        # if not standing command
        if torch.any(has_true_in_row == 1):
            _rew[has_true_in_row == 1] = 1

        # if standing command
        if torch.any(self.standing_command_mask == 0):
            _rew[self.standing_command_mask == 0] = 1
        return _rew

    def _reward_feet_airtime(self):
        """
        通过在每次脚着地时施加-0.4的惩罚来规范踏步频率，
        这可以通过一个积极的奖励成分来抵消，即足部腾空后的秒数(腾空时间)。
        如果没有这个组件，学习到的控制器倾向于采用步进频率在风格上太大的步态，
        这可能是由于这些频率对应于可能的局部最小值。这个分量在站立时是恒定的。
        """
        _rew = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        # if standing command
        if torch.any(self.standing_command_mask == 1):
            _rew[self.standing_command_mask == 1] = 1
        # else
        contact = self.contact_forces[:, self.feet_indices, 2] > 0
        self.contact_filt = torch.logical_and(
            contact, ~self.last_contacts
        )  # 是否第一次接触，如果接触就为1，不接触就为0
        self.last_contacts = contact  # 更新上一帧的接触情况
        # print(self.contact_filt[0,:])
        self.feet_air_time += self.dt
        self.feet_air_time *= ~contact  # 不接触的话就持续计数，接触就清零
        air_time_reward: torch.Tensor = self.feet_air_time * self.contact_filt - 0.4
        # print(air_time_reward)
        if torch.any(self.standing_command_mask == 0):
            _rew[self.standing_command_mask == 0] = air_time_reward[
                self.standing_command_mask == 0
            ].sum(dim=1)
        # print(_rew)
        return _rew

    def _reward_feet_air_time(self):
        """
        Calculates the reward for feet air time, promoting longer steps. This is achieved by
        checking the first contact with the ground after being in the air. The air time is
        limited to a maximum value for reward calculation.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        stance_mask = self._get_gait_phase()
        self.contact_filt = torch.logical_or(torch.logical_or(contact, stance_mask), self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * self.contact_filt
        self.feet_air_time += self.dt
        air_time = self.feet_air_time.clamp(0, 0.5) * first_contact
        self.feet_air_time *= ~self.contact_filt
        return air_time.sum(dim=1)
    
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

        # _rew = torch.exp(-30 * torch.sum(diff_3, dim=(1, 2)))

        _rew = torch.exp(-6 * torch.sum(diff_2, dim=(1, 2)))

        # print(_rew.size())
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

    def _reward_feet_position(self):
        _rew = torch.ones(
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
            [[-0.055, -0.055], [-0.0801, 0.0801], [0.3453 - 0.0480, 0.3453 - 0.0480]],
            device="cuda:0",
        )
        # print(aa)
        diff = base_pos.unsqueeze(-1) - feet_pos - aa
        # print(diff[0,...])
        if torch.any(self.standing_command_mask == 1):
            _rew[self.standing_command_mask == 1] = torch.exp(
                -torch.sum(
                    torch.norm(
                        diff,
                        dim=2,
                    ),
                    dim=1,
                )
            )[self.standing_command_mask == 1]
        return _rew

    def _reward_action_smoothness(self):
        """
        Encourages smoothness in the robot's actions by penalizing large differences between consecutive actions.
        This is important for achieving fluid motion and reducing mechanical stress.
        """
        term_1 = torch.sum(torch.square(self.last_actions - self.actions), dim=1)
        term_2 = 10 * torch.sum(
            torch.square(self.actions + self.last_last_actions - 2 * self.last_actions),
            dim=1,
        )
        term_3 = 0.05 * torch.sum(torch.abs(self.actions), dim=1)
        return term_1 + term_2 + term_3

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
        foot_pos = self.rigid_state[:, self.feet_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.cfg.rewards.min_dist
        max_df = self.cfg.rewards.max_dist
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.0)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        return (
            torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)
        ) / 2

    def _reward_knee_distance(self):
        """
        Calculates the reward based on the distance between the knee of the humanoid.
        """
        foot_pos = self.rigid_state[:, self.knee_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.cfg.rewards.min_dist
        max_df = self.cfg.rewards.max_dist
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.0)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        return (
            torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)
        ) / 2

    def _reward_base_acc(self):
        root_acc = self.last_root_vel - self.root_states[:, 7:13]
        rew = torch.exp(-torch.sum(torch.abs(root_acc), dim=1))
        return rew

    def _reward_ankle_roll_angle_zero(self):
        ankle_roll_indices = [5, 11]
        ankle_roll = self.dof_pos[:, ankle_roll_indices]
        rew = torch.exp(-torch.sum(torch.abs(ankle_roll), dim=1))
        return rew

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

    def _reward_torques(self):
        """
        Penalizes the use of high torques in the robot's joints. Encourages efficient movement by minimizing
        the necessary force exerted by the motors.
        """
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        """
        Penalizes high velocities at the degrees of freedom (DOF) of the robot. This encourages smoother and
        more controlled movements.
        """
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        """
        Penalizes high accelerations at the robot's degrees of freedom (DOF). This is important for ensuring
        smooth and stable motion, reducing wear on the robot's mechanical parts.
        """
        return torch.sum(
            torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1
        )

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

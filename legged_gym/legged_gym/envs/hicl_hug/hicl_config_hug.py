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

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class HiclHugCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        obs_with_base_lin_vel = False
        # obs_with_base_lin_vel = True
        if obs_with_base_lin_vel:
            num_observations = 0
        else:
            num_observations = 55
        num_obs_hist = 0
        num_actor_obs = num_observations
        num_privileged_obs = 303
        num_critic_obs = num_privileged_obs

        num_actions = 12
        env_spacing = 3.0  # not used with heightfields/trimeshes
        send_timeouts = True  # send time out information to the algorithm
        episode_length_s = 11  # episode length in seconds

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.648]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            "r_hip_pitch_joint": 0.0,
            "r_hip_roll_joint": 0.0,
            "r_thigh_joint": 0.0,
            "r_calf_joint": 0.0,
            "r_ankle_pitch_joint": 0.0,
            "r_ankle_roll_joint": 0.0,
            "l_hip_pitch_joint": 0.0,
            "l_hip_roll_joint": 0.0,
            "l_thigh_joint": 0.0,
            "l_calf_joint": 0.0,
            "l_ankle_pitch_joint": 0.0,
            "l_ankle_roll_joint": 0.0,
        }
        dof_pos_range_virtual = {  # 虚拟限位
            "l_hip_yaw_joint": [-1.0, 1.8],
            "l_hip_roll_joint": [-0.12, 0.5],
            "l_thigh_joint": [-0.3, 0.6],
            "l_calf_joint": [-0.8, 1.5],
            "l_ankle_pitch_joint": [-0.45, 1.15],
            "l_ankle_roll_joint": [-0.15, 0.15],
            "r_hip_yaw_joint": [-1.0, 1.8],
            "r_hip_roll_joint": [-0.5, 0.12],
            "r_thigh_joint": [-0.6, 0.3],
            "r_calf_joint": [-0.8, 1.5],
            "r_ankle_pitch_joint": [-0.45, 1.15],
            "r_ankle_roll_joint": [-0.15, 0.15],
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = "P"
        stiffness = {
            "hip_pitch_joint": 40.0,
            "hip_roll_joint": 20.0,
            "thigh_joint": 20.0,
            "calf_joint": 40.0,
            "ankle_pitch_joint": 20,
            "ankle_roll_joint": 20,
        }
        damping = {
            "hip_pitch_joint": 0.5,
            "hip_roll_joint": 0.8,
            "thigh_joint": 0.8,
            "calf_joint": 0.5,
            "ankle_pitch_joint": 0.5,
            "ankle_roll_joint": 0.5,
        }
        '''
        stiffness = {
            "hip_pitch_joint": 100.0,
            "hip_roll_joint": 100.0,
            "thigh_joint": 100.0,
            "calf_joint": 150.0,
            "ankle_pitch_joint": 40,
            "ankle_roll_joint": 40,
        }
        damping = {
            "hip_pitch_joint": 2,
            "hip_roll_joint": 2,
            "thigh_joint": 2,
            "calf_joint": 4,
            "ankle_pitch_joint": 2,
            "ankle_roll_joint": 2,
        }'''
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/hi_cl_23_240925/urdf/hi_cl_23_240925_rl.urdf"
        name = "Hi"
        foot_name = "ankle_roll"
        knee_name = "calf"

        # penalize_contacts_on = []
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base_link","hip_pitch","hip_roll","thigh","calf"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
        # fix_base_link = True # fixe the base of the robot
        fix_base_link = False  # fixe the base of the robot

    class normalization:
        contact_force_range = [0.0, 200.0]

        class obs_scales:
            lin_vel = 2.0
            ang_vel = 1.0
            dof_pos = 1.0
            dof_vel = 0.05
            quat = 1.0
            height_measurements = 5.0

        clip_observations = 18.0
        clip_actions = 18.0

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.0  # scales other values

        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            dof_pos = 0.05
            dof_vel = 0.5 * 2
            ang_vel = 0.1
            lin_vel = 0.05
            quat = 0.03
            gravity = 0.5
            height_measurements = 0.1

    class sim(LeggedRobotCfg.sim):
        dt = 0.001  # 1000 Hz
        substeps = 1  # 2
        up_axis = 1  # 0 is y, 1 is z

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 32
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 4
            contact_offset = 0.01  # [m]
            rest_offset = 0.0  # [m]
            bounce_threshold_velocity = 0.1  # [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            contact_collection = 2

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.99, 1.0]
        # friction_range = [0.1, 0.45]

        randomize_base_mass = False
        added_mass_range = [-1.5, 1.5]

        disturbance = False
        # disturbance = True
        disturbance_range = [-200.0,200.0]
        disturbance_interval = 3

        push_robots = False
        push_interval_s = 3  # 每次推的间隔时间
        max_push_vel_xy = 0.1
        # max_push_ang_vel = 0.2
        dynamic_randomization = 0.02

        randomize_com_displacement = False
        com_displacement_range = [-0.1, 0.1]

        randomize_motor_strength = False
        motor_strength_range = [0.9, 1.1]

        randomize_Kp_factor = False
        Kp_factor_range = [0.9, 1.1]

        randomize_Kd_factor = False
        Kd_factor_range = [0.9, 1.1]

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"

    class commands(LeggedRobotCfg.commands):
        # Vers: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 3.0  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error
        curriculum = True
        max_curriculum_x = [-1.2, 1.2]
        max_disturbance_range = [-440.0, 440.0]

        class ranges:
            lin_vel_x = [-1.8, 1.8]  # min max [m/s]
            lin_vel_y = [-0.5, 0.5]  # min max [m/s]
            ang_vel_yaw = [-0.9, 0.9]  # min max [rad/s]
            heading = [-3.14, 3.14]
            disturbance_range = [-320.0, 320.0]
            
            gait_frequency = [1.8,1.81]
            swing_height = [0.03,0.031]
            base_height = [-0.0,0]

    class rewards:
        soft_dof_pos_limit = 0.98

        base_height_target = 0.648

        min_dist_fe = 0.205 * 0.95
        max_dist_fe = 0.205 * 1.05
        min_dist_kn = 0.1953 * 0.95
        max_dist_kn = 0.1953 * 1.05
        # ref
        target_feet_height = 0.07  # m  0.025
        cycle_time = 0.66  # sec
        bias = 0.3
        y_bias = 0.3
        eps = 0.2
        target_joint_pos_scale = 0.1  # rad
        # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards = True
        # tracking reward = exp(error*sigma)
        tracking_sigma = 5
        max_contact_force = 150  # forces above this value are penalized

        # use_ankle_pitch= True

        plus = 10

        class scales:
            termination = 1.0
            
            # ====vel track======
            lin_vel_track = 10 # 2 10
            ang_vel_track = 2
            
            # ====body pose track=====
            body_height_track = 10
            roll_pitch_orient = 10  
            
            # ====foot pose track=====
            foot_swing_track = 40 # 2 5
            feet_contact = 5
            feet_airtime = 2
            # contact_swing_track = -1
            # style_similar = 80
            knee_distance = 0.16
            feet_distance = 0.16
            feet_distance_2 = 0.16
            feet_distance_x = 1
            ankle_roll_posture_roll = 0.5
            ankle_roll_posture_pitch = 1.5 *10
            # ====slow=====
            action_smoothness = -2e-3 # -2e-4
            torques = -1e-6
            dof_vel = -5e-4
            dof_acc = -1e-7
            # ====joint regulation======
            joint_pos = 10.0
            
            
            # foot_swing_track = -30
            # contact_swing_track = -2
            
            # rp_ang_vel = -0.5
            # vertical_body_movement = -0.1
            # feet_slip = -0.2
            # action_rate = -0.01
            # action_smoothness = -0.01
            # joint_torque = -5e-6
            # joint_acc = -2.5e-7
            # hip_joint_deviation = -2
            # feet_symmetry = -1
            
            # # ===base velocity=========
            # tracking_lin_vel = 1.5
            # tracking_ang_vel = 1.5
            # base_z_vel = 0.1
            # low_speed = 0.2
            # # ===base pose=============
            # roll_pitch_orient = 20  
            # base_height = 10
            # # ===base acc==============
            # base_acc = 0.1
            # # ===action smooth=========
            
            # # ===feet motion===========
            # feet_contact_forces = -0.02
            # feet_swing_height =1
            foot_slip = -2
            # boundary = -10
            # # ===pos===================
            # joint_pos = 1.0
            # feet_distance = 0.16
            # feet_distance_2 = 0.16
            # knee_distance = 0.16
            # feet_orient = 0.9
            # feet_position = 0.9

            
class HiclHugCfgPPO(LeggedRobotCfgPPO):
    class policy:
        init_noise_std = 1.0
        # actor_hidden_dims = [512, 256, 128]
        # critic_hidden_dims = [512, 256, 128]

        actor_hidden_dims = [32]
        critic_hidden_dims = [32]
        rnn_hidden_size = 64
        rnn_num_layers = 1

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = "hicl_Hug_WBC"

        policy_class_name = "ActorCriticLSTM"
        # policy_class_name = 'ActorCritic'
        # policy_class_name = 'ActorCriticRecurrent'

        experiment_name = "hicl_hug_" + policy_class_name
        # load and resume
        # resume=True
        # load_run = 'Nov08_09-31-25_'# -1 = last run
        # checkpoint = '400'
        resume = False

        num_steps_per_env = 25  # per iteration
        max_iterations = 1501  # number of policy updates
        # logging
        save_interval = (
            25  # Please check for potential savings every `save_interval` iterations.
        )

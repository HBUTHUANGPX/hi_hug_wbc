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
my_pi = 3.1415926


class PaiNonePhaseCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        obs_with_base_lin_vel = False
        # obs_with_base_lin_vel = True # DGZ
        if obs_with_base_lin_vel: # DGZ
            num_observations = 51
        else:
            num_observations = 48
        # num_obs_hist = 5
        # num_obs_hist = 15
        num_obs_hist = 30
        # num_privileged_obs = 293# + 3
        num_privileged_obs = 293 + 3
        num_actions = 12
        env_spacing = 3.0  # not used with heightfields/trimeshes
        send_timeouts = True  # send time out information to the algorithm
        episode_length_s = 16  # episode length in seconds

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.3453]  # x,y,z [m]
        reset_mode = 'reset_to_range' # 'reset_to_basic'
        # reset_mode = "reset_to_basic"  # 'reset_to_basic'
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
            "l_hip_yaw_joint": [-0.8, 0.8],
            "l_hip_roll_joint": [-0.157, 0.5],
            "l_thigh_joint": [-1.45, 0.35],
            "l_calf_joint": [-0.2, 2.1],
            "l_ankle_pitch_joint": [-0.55, 0.6],
            "l_ankle_roll_joint": [-0.7, 0.25],
            "r_hip_yaw_joint": [-0.8, 0.8],
            "r_hip_roll_joint": [-0.5, 0.157],
            "r_thigh_joint": [-1.45, 0.35],
            "r_calf_joint": [-0.2, 2.1],
            "r_ankle_pitch_joint": [-0.55, 0.6],
            "r_ankle_roll_joint": [-0.25, 0.7],
        }

        root_pos_range = [
            [0.0, 0.0],  # x
            [0.0, 0.0],  # y
            [pos[2]*1.0, pos[2]*1.1],  # z
            [-my_pi / 36*0, my_pi / 36*0],  # roll
            [-my_pi / 36*0, my_pi / 36*0],  # pitch
            [-my_pi / 36*0, my_pi / 36*0],  # yaw
        ]
        # ranges for [v_x, v_y, v_z, w_x, w_y, w_z]
        root_vel_range = [
            [-0.3*0, 0.3*0],  # x
            [-0.01*0, 0.01*0],  # y
            [-0.01*0, 0.01*0],  # z
            [-0.01*0, 0.01*0],  # roll
            [-0.01*0, 0.01*0],  # pitch
            [-0.01*0, 0.01*0],  # yaw
        ]

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = "P"
        stiffness = {
            "hip_pitch_joint": 40.0,
            "hip_roll_joint": 40.0,
            "thigh_joint": 20.0,
            "calf_joint": 40.0,
            "ankle_pitch_joint": 30,
            "ankle_roll_joint": 10,
        }
        damping = {
            "hip_pitch_joint": 2.4,
            "hip_roll_joint": 0.8,
            "thigh_joint": 0.4,
            "calf_joint": 2.8,
            "ankle_pitch_joint": 1.6,
            "ankle_roll_joint": 0.3,
        }
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/pi_12dof_release_v1/urdf/pi_12dof_release_v1_rl.urdf"
        name = "Pai"
        foot_name = "ankle_roll"
        knee_name = "calf"

        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base_link"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        # fix_base_link = True # fixe the base of the robot
        fix_base_link = False # fixe the base of the robot

    class normalization:
        contact_force_range = [0.0, 50.0]

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
        noise_level = 1.  # scales other values

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
        friction_range = [0.1, 0.45]

        randomize_base_mass = False
        added_mass_range = [-0.5, 0.5]

        disturbance = False
        disturbance_range = [-20.0, 20.0]
        disturbance_range_max = [-440.0, 440.0]
        disturbance_interval = 1

        push_robots = True
        push_interval_s = 4 # 每次推的间隔时间
        max_push_vel_xy = 0.2
        # max_push_ang_vel = 0.2
        dynamic_randomization = 0.02

        randomize_com_displacement = False
        com_displacement_range = [-0.05, 0.05]

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
        resampling_time = 5.0  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error
        # curriculum = True #False
        curriculum = False
        max_curriculum_x = [-0.8, 0.8]
        max_disturbance_range = [-440.0, 440.0]
        cycle_time_adjust_rate = [1,1.6]
        class ranges:
            lin_vel_x = [-0.8, 0.8]  # min max [m/s]
            lin_vel_y = [-0.25, 0.25]  # min max [m/s]
            ang_vel_yaw = [-0.6, 0.6]  # min max [rad/s]
            heading = [-3.14, 3.14]
            disturbance_range = [-220.0, 220.0]


    class rewards:
        soft_dof_pos_limit = 0.98

        base_height_target = 0.3453
        min_dist = 0.15
        max_dist = 0.2
        
        # ref
        target_joint_pos_scale = 0.1  # rad
        target_feet_height = 0.025  # m  0.025
        cycle_time = 0.4  # sec
        bias = 0.3
        eps = 0.1
        y_bias = 0.2
        # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards = True
        # tracking reward = exp(error*sigma)
        tracking_sigma = 5
        max_contact_force = 400  # forces above this value are penalized
        # use_ankle_pitch= True
        plus = 10

        class scales:
            # xy_velocity = 0.5
            # yaw_velocity = 0.5
            tracking_lin_vel = 1.5
            tracking_ang_vel = 1.5

            roll_pitch_orient = 0.9
            # base_height = 0.8 
            base_height_2 = 2.8
            base_acc = 0.1
            # base_z_vel = 0.1

            action_smoothness = -0.02
            # default_joint_pos = 5
            joint_pos = 0.18 #0.18 0.155 0.13 0.08 1.5
            default_joint_pos = 1.5#0.5

            # foot_slip = -0.15
            feet_clearance = 0.8
            feet_clearance_2 = 0.8
            feet_contact_number = 0.4 # 0.4
            feet_distance = 0.16 
            knee_distance = 0.16 
            # feet_contact = 2.5 # 0.5
            # feet_airtime = -1
            feet_air_time = 5
            feet_orient = 1.3
            # feet_position = 0.1
            ankle_roll_angle_zero = 2
            low_speed =  0.2
            torques = -1e-6
            dof_vel = -5e-4
            dof_acc = -1e-7


class PaiNonePhaseCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        # value_loss_coef = 1.0
        # use_clipped_value_loss = True
        # clip_param = 0.2
        # entropy_coef = 0.01
        # num_learning_epochs = 2
        # num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        # learning_rate = 1.e-5 #5.e-4
        # schedule = 'adaptive' # could be adaptive, fixed
        # gamma = 0.994
        # lam = 0.9
        desired_kl = 0.01
        # max_grad_norm = 1.

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ""
        experiment_name = "pai_none_phase"
        # load and resume
        # resume=True
        # load_run = 'Nov08_09-31-25_'# -1 = last run
        # checkpoint = '400'
        resume = False

        num_steps_per_env = 30  # per iteration
        max_iterations = 1001  # number of policy updates
        # logging
        save_interval = (
            50  # Please check for potential savings every `save_interval` iterations.
        )

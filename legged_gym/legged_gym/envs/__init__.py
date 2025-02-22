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

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .base.legged_robot import LeggedRobot

import os

from legged_gym.utils.task_registry import task_registry

from.pai_none_phase.pai_env_none_phase import PaiNonePhaseEnv
from .pai_none_phase.pai_config_none_phase import PaiNonePhaseCfg, PaiNonePhaseCfgPPO
task_registry.register(
    "pai_none_phase", PaiNonePhaseEnv, PaiNonePhaseCfg(), PaiNonePhaseCfgPPO()
)

from.hi_none_phase.hi_env_none_phase import HiNonePhaseEnv
from .hi_none_phase.hi_config_none_phase import HiNonePhaseCfg, HiNonePhaseCfgPPO
task_registry.register(
    "hi_none_phase", HiNonePhaseEnv, HiNonePhaseCfg(), HiNonePhaseCfgPPO()
)


from.hi_hug.hi_env_hug import HiHugEnv
from .hi_hug.hi_config_hug import HiHugCfg, HiHugCfgPPO
task_registry.register(
    "hi_hug", HiHugEnv, HiHugCfg(), HiHugCfgPPO()
)

from.hicl_hug.hicl_env_hug import HiclHugEnv
from .hicl_hug.hicl_config_hug import HiclHugCfg, HiclHugCfgPPO
task_registry.register(
    "hicl_hug", HiclHugEnv, HiclHugCfg(), HiclHugCfgPPO()
)

from.mini_pi_hug.mpi_env_hug import MpiHugEnv
from .mini_pi_hug.mpi_config_hug import MpiHugCfg, MpiHugCfgPPO
task_registry.register(
    "Mpi_hug", MpiHugEnv, MpiHugCfg(), MpiHugCfgPPO()
)
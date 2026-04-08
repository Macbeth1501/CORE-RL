# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Core Rl Environment.

The core_rl environment is a simple test environment that echoes back messages.
"""

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Data models for the CORE-RL Environment.
This file forwards the FinOps models from the server directory to the package root.
"""

from .server.models import Action, Observation, Resource, Reward

# We keep the class names generic (Action, Observation) to match 
# the OpenEnv expectations while using our custom FinOps fields.
__all__ = ["Action", "Observation", "Resource", "Reward"]

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Data models for the CORE-RL Environment.
"""

from .server.models import Action, Observation, Resource, Reward

# ALIASES: These satisfy the old template code looking for 'CoreRl...' names
CoreRlAction = Action
CoreRlObservation = Observation

__all__ = [
    "Action", 
    "Observation", 
    "Resource", 
    "Reward", 
    "CoreRlAction", 
    "CoreRlObservation"
]
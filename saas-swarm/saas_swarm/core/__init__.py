"""
Core components for the SaaS-Swarm platform.

This module contains the fundamental building blocks:
- Agent: Individual AI agents with inference capabilities
- SwarmTopology: Network topologies for agent communication
- MessageBus: Asynchronous communication layer
- FeedbackLoop: Reward and adaptation system
"""

from .agent import Agent
from .swarm import SwarmTopology, MeshSwarm, StarSwarm, RingSwarm, HierarchicalSwarm
from .message_bus import MessageBus
from .feedback_loop import FeedbackLoop

__all__ = [
    "Agent",
    "SwarmTopology",
    "MeshSwarm", 
    "StarSwarm",
    "RingSwarm",
    "HierarchicalSwarm",
    "MessageBus",
    "FeedbackLoop"
] 
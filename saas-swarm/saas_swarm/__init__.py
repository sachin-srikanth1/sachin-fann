"""
SaaS-Swarm: Swarm-as-a-Service Platform

A modular and efficient Python platform for implementing lightweight, 
trainable AI agent swarms with flexible topologies and real-time collaboration.
"""

__version__ = "0.1.0"
__author__ = "SaaS-Swarm Team"

from .core.agent import Agent
from .core.swarm import SwarmTopology, MeshSwarm, StarSwarm, RingSwarm, HierarchicalSwarm
from .core.message_bus import MessageBus
from .core.feedback_loop import FeedbackLoop
from .tools.registry import ToolRegistry

__all__ = [
    "Agent",
    "SwarmTopology", 
    "MeshSwarm",
    "StarSwarm", 
    "RingSwarm",
    "HierarchicalSwarm",
    "MessageBus",
    "FeedbackLoop",
    "ToolRegistry"
] 
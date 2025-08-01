"""
Examples module for SaaS-Swarm platform.

This module contains sample swarms and configurations
for demonstration and testing purposes.
"""

from .email_writer import create_email_writer_swarm
from .route_optimizer import create_route_optimizer_swarm
from .code_review import create_code_review_swarm

__all__ = [
    "create_email_writer_swarm",
    "create_route_optimizer_swarm", 
    "create_code_review_swarm"
] 
"""
Swarm module for SaaS-Swarm platform.

Defines swarm topologies and execution patterns for agent collaboration.
"""

import asyncio
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from abc import ABC, abstractmethod
import uuid

from .agent import Agent, AgentConfig
from .message_bus import MessageBus, MessageBusConfig


@dataclass
class SwarmConfig:
    """Configuration for a swarm."""
    swarm_id: str
    name: str
    topology_type: str = "mesh"  # mesh, star, ring, hierarchical
    max_execution_time: float = 30.0
    enable_feedback: bool = True
    message_bus_config: Optional[MessageBusConfig] = None


class SwarmTopology(ABC):
    """
    Abstract base class for swarm topologies.
    
    Defines the interface for different swarm patterns:
    - Mesh: All agents connected to all others
    - Star: Central hub with spoke connections
    - Ring: Circular connection pattern
    - Hierarchical: Tree-like structure
    """
    
    def __init__(self, config: SwarmConfig):
        self.config = config
        self.swarm_id = config.swarm_id
        self.name = config.name
        self.agents: Dict[str, Agent] = {}
        self.message_bus = MessageBus(config.message_bus_config or MessageBusConfig())
        self.is_running = False
        self.execution_history: List[Dict] = []
        
    @abstractmethod
    def add_agent(self, agent: Agent) -> bool:
        """Add an agent to the swarm topology."""
        pass
        
    @abstractmethod
    def remove_agent(self, agent_id: str) -> bool:
        """Remove an agent from the swarm topology."""
        pass
        
    @abstractmethod
    def get_agent_connections(self, agent_id: str) -> List[str]:
        """Get the list of agent IDs that the given agent is connected to."""
        pass
        
    async def start(self):
        """Start the swarm."""
        self.is_running = True
        await self.message_bus.start()
        
        # Start all agents
        for agent in self.agents.values():
            await agent.start()
            
        print(f"Swarm {self.name} started with {len(self.agents)} agents")
        
    async def stop(self):
        """Stop the swarm."""
        self.is_running = False
        
        # Stop all agents
        for agent in self.agents.values():
            await agent.stop()
            
        await self.message_bus.stop()
        print(f"Swarm {self.name} stopped")
        
    async def execute(self, input_data: Any, timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Execute the swarm with given input.
        
        Args:
            input_data: Input data for the swarm
            timeout: Execution timeout in seconds
            
        Returns:
            Dictionary containing execution results
        """
        if not self.is_running:
            raise RuntimeError("Swarm is not running")
            
        timeout = timeout or self.config.max_execution_time
        start_time = asyncio.get_event_loop().time()
        
        # Record execution start
        execution_id = str(uuid.uuid4())
        execution_record = {
            'execution_id': execution_id,
            'input_data': input_data,
            'start_time': start_time,
            'results': {}
        }
        
        try:
            # Execute the specific topology logic
            results = await self._execute_topology(input_data, timeout)
            execution_record['results'] = results
            execution_record['success'] = True
            
        except Exception as e:
            execution_record['error'] = str(e)
            execution_record['success'] = False
            raise
            
        finally:
            execution_record['end_time'] = asyncio.get_event_loop().time()
            execution_record['duration'] = execution_record['end_time'] - start_time
            self.execution_history.append(execution_record)
            
        return execution_record
        
    @abstractmethod
    async def _execute_topology(self, input_data: Any, timeout: float) -> Dict[str, Any]:
        """Execute the specific topology logic."""
        pass
        
    def get_agent_info(self) -> Dict[str, Dict]:
        """Get information about all agents in the swarm."""
        return {
            agent_id: {
                'name': agent.name,
                'connections': self.get_agent_connections(agent_id),
                'state': agent.get_state()
            }
            for agent_id, agent in self.agents.items()
        }
        
    def get_execution_history(self, limit: int = 10) -> List[Dict]:
        """Get recent execution history."""
        return self.execution_history[-limit:]


class MeshSwarm(SwarmTopology):
    """
    Mesh topology where all agents are connected to all others.
    
    Each agent can communicate directly with every other agent.
    """
    
    def add_agent(self, agent: Agent) -> bool:
        """Add an agent to the mesh swarm."""
        if agent.agent_id in self.agents:
            return False
            
        self.agents[agent.agent_id] = agent
        self.message_bus.register_agent(agent)
        return True
        
    def remove_agent(self, agent_id: str) -> bool:
        """Remove an agent from the mesh swarm."""
        if agent_id not in self.agents:
            return False
            
        del self.agents[agent_id]
        self.message_bus.unregister_agent(agent_id)
        return True
        
    def get_agent_connections(self, agent_id: str) -> List[str]:
        """Get all other agents in the mesh."""
        if agent_id not in self.agents:
            return []
        return [aid for aid in self.agents.keys() if aid != agent_id]
        
    async def _execute_topology(self, input_data: Any, timeout: float) -> Dict[str, Any]:
        """Execute mesh topology - broadcast to all agents and collect responses."""
        if not self.agents:
            return {'error': 'No agents in swarm'}
            
        # Get the first agent as the initial processor
        first_agent_id = list(self.agents.keys())[0]
        first_agent = self.agents[first_agent_id]
        
        # Process with first agent
        initial_result = await first_agent.infer(input_data)
        
        # Broadcast result to all other agents
        other_agents = [aid for aid in self.agents.keys() if aid != first_agent_id]
        
        if other_agents:
            await self.message_bus.broadcast_message(
                first_agent_id, 
                initial_result,
                exclude_sender=True
            )
            
        return {
            'initial_agent': first_agent_id,
            'initial_result': initial_result,
            'total_agents': len(self.agents),
            'mesh_connections': len(self.agents) * (len(self.agents) - 1) // 2
        }


class StarSwarm(SwarmTopology):
    """
    Star topology with a central hub agent and spoke connections.
    
    All communication goes through the central agent.
    """
    
    def __init__(self, config: SwarmConfig):
        super().__init__(config)
        self.hub_agent_id: Optional[str] = None
        
    def set_hub_agent(self, agent_id: str) -> bool:
        """Set the central hub agent."""
        if agent_id not in self.agents:
            return False
        self.hub_agent_id = agent_id
        return True
        
    def add_agent(self, agent: Agent) -> bool:
        """Add an agent to the star swarm."""
        if agent.agent_id in self.agents:
            return False
            
        self.agents[agent.agent_id] = agent
        self.message_bus.register_agent(agent)
        
        # Set as hub if it's the first agent
        if self.hub_agent_id is None:
            self.hub_agent_id = agent.agent_id
            
        return True
        
    def remove_agent(self, agent_id: str) -> bool:
        """Remove an agent from the star swarm."""
        if agent_id not in self.agents:
            return False
            
        del self.agents[agent_id]
        self.message_bus.unregister_agent(agent_id)
        
        # If hub was removed, set new hub
        if agent_id == self.hub_agent_id and self.agents:
            self.hub_agent_id = list(self.agents.keys())[0]
            
        return True
        
    def get_agent_connections(self, agent_id: str) -> List[str]:
        """Get connections for star topology."""
        if agent_id == self.hub_agent_id:
            # Hub connects to all spokes
            return [aid for aid in self.agents.keys() if aid != agent_id]
        else:
            # Spokes only connect to hub
            return [self.hub_agent_id] if self.hub_agent_id else []
            
    async def _execute_topology(self, input_data: Any, timeout: float) -> Dict[str, Any]:
        """Execute star topology - hub processes and coordinates with spokes."""
        if not self.agents or self.hub_agent_id is None:
            return {'error': 'No hub agent or agents in swarm'}
            
        hub_agent = self.agents[self.hub_agent_id]
        spoke_agents = [aid for aid in self.agents.keys() if aid != self.hub_agent_id]
        
        # Hub processes initial input
        hub_result = await hub_agent.infer(input_data)
        
        # Send to spokes for further processing
        spoke_results = {}
        for spoke_id in spoke_agents:
            spoke_agent = self.agents[spoke_id]
            spoke_result = await spoke_agent.infer(hub_result)
            spoke_results[spoke_id] = spoke_result
            
        return {
            'hub_agent': self.hub_agent_id,
            'hub_result': hub_result,
            'spoke_results': spoke_results,
            'total_agents': len(self.agents)
        }


class RingSwarm(SwarmTopology):
    """
    Ring topology where agents are connected in a circular pattern.
    
    Each agent connects to exactly two others (next and previous).
    """
    
    def add_agent(self, agent: Agent) -> bool:
        """Add an agent to the ring swarm."""
        if agent.agent_id in self.agents:
            return False
            
        self.agents[agent.agent_id] = agent
        self.message_bus.register_agent(agent)
        return True
        
    def remove_agent(self, agent_id: str) -> bool:
        """Remove an agent from the ring swarm."""
        if agent_id not in self.agents:
            return False
            
        del self.agents[agent_id]
        self.message_bus.unregister_agent(agent_id)
        return True
        
    def get_agent_connections(self, agent_id: str) -> List[str]:
        """Get ring connections for the agent."""
        if agent_id not in self.agents:
            return []
            
        agent_ids = list(self.agents.keys())
        if len(agent_ids) < 2:
            return []
            
        try:
            idx = agent_ids.index(agent_id)
            prev_idx = (idx - 1) % len(agent_ids)
            next_idx = (idx + 1) % len(agent_ids)
            
            return [agent_ids[prev_idx], agent_ids[next_idx]]
        except ValueError:
            return []
            
    async def _execute_topology(self, input_data: Any, timeout: float) -> Dict[str, Any]:
        """Execute ring topology - pass data around the ring."""
        if not self.agents:
            return {'error': 'No agents in swarm'}
            
        agent_ids = list(self.agents.keys())
        results = {}
        current_data = input_data
        
        # Pass data through the ring
        for i, agent_id in enumerate(agent_ids):
            agent = self.agents[agent_id]
            result = await agent.infer(current_data)
            results[agent_id] = result
            current_data = result  # Pass result to next agent
            
        return {
            'ring_order': agent_ids,
            'results': results,
            'final_result': current_data,
            'total_agents': len(self.agents)
        }


class HierarchicalSwarm(SwarmTopology):
    """
    Hierarchical topology with tree-like structure.
    
    Supports multiple levels with parent-child relationships.
    """
    
    def __init__(self, config: SwarmConfig):
        super().__init__(config)
        self.hierarchy: Dict[str, List[str]] = {}  # parent -> children
        self.parents: Dict[str, str] = {}  # child -> parent
        
    def add_agent(self, agent: Agent, parent_id: Optional[str] = None) -> bool:
        """Add an agent to the hierarchical swarm."""
        if agent.agent_id in self.agents:
            return False
            
        self.agents[agent.agent_id] = agent
        self.message_bus.register_agent(agent)
        
        # Set up hierarchy
        if parent_id is None:
            # Root agent
            self.hierarchy[agent.agent_id] = []
        else:
            # Child agent
            if parent_id not in self.hierarchy:
                self.hierarchy[parent_id] = []
            self.hierarchy[parent_id].append(agent.agent_id)
            self.parents[agent.agent_id] = parent_id
            
        return True
        
    def remove_agent(self, agent_id: str) -> bool:
        """Remove an agent from the hierarchical swarm."""
        if agent_id not in self.agents:
            return False
            
        # Remove from hierarchy
        if agent_id in self.hierarchy:
            # Move children to parent
            children = self.hierarchy[agent_id]
            parent_id = self.parents.get(agent_id)
            
            for child_id in children:
                if parent_id:
                    self.parents[child_id] = parent_id
                    self.hierarchy[parent_id].append(child_id)
                else:
                    # Child becomes root
                    del self.parents[child_id]
                    
            del self.hierarchy[agent_id]
            
        if agent_id in self.parents:
            parent_id = self.parents[agent_id]
            if parent_id in self.hierarchy:
                self.hierarchy[parent_id].remove(agent_id)
            del self.parents[agent_id]
            
        del self.agents[agent_id]
        self.message_bus.unregister_agent(agent_id)
        return True
        
    def get_agent_connections(self, agent_id: str) -> List[str]:
        """Get hierarchical connections for the agent."""
        connections = []
        
        # Add children
        if agent_id in self.hierarchy:
            connections.extend(self.hierarchy[agent_id])
            
        # Add parent
        if agent_id in self.parents:
            connections.append(self.parents[agent_id])
            
        return connections
        
    async def _execute_topology(self, input_data: Any, timeout: float) -> Dict[str, Any]:
        """Execute hierarchical topology - process from root down."""
        if not self.agents:
            return {'error': 'No agents in swarm'}
            
        # Find root agents (those without parents)
        root_agents = [aid for aid in self.agents.keys() if aid not in self.parents]
        
        if not root_agents:
            return {'error': 'No root agents found'}
            
        results = {}
        
        # Process from each root
        for root_id in root_agents:
            root_result = await self._process_hierarchy_branch(root_id, input_data, results)
            results[root_id] = root_result
            
        return {
            'root_agents': root_agents,
            'results': results,
            'hierarchy': self.hierarchy,
            'total_agents': len(self.agents)
        }
        
    async def _process_hierarchy_branch(self, agent_id: str, input_data: Any, 
                                      results: Dict[str, Any]) -> Any:
        """Process a branch of the hierarchy."""
        agent = self.agents[agent_id]
        result = await agent.infer(input_data)
        results[agent_id] = result
        
        # Process children
        if agent_id in self.hierarchy:
            for child_id in self.hierarchy[agent_id]:
                await self._process_hierarchy_branch(child_id, result, results)
                
        return result 
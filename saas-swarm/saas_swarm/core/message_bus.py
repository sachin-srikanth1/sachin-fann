"""
MessageBus module for SaaS-Swarm platform.

Provides asynchronous communication layer for agent messaging
with support for different backend implementations.
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import uuid

from .agent import Message


@dataclass
class MessageBusConfig:
    """Configuration for MessageBus."""
    backend: str = "memory"  # memory, redis, zeromq
    max_queue_size: int = 1000
    enable_persistence: bool = False


class MessageBus:
    """
    Asynchronous communication layer for agent messaging.
    
    Supports different backends:
    - In-memory queues (default)
    - Redis (future)
    - ZeroMQ (future)
    """
    
    def __init__(self, config: MessageBusConfig = None):
        self.config = config or MessageBusConfig()
        self.agents: Dict[str, 'Agent'] = {}
        self.message_handlers: Dict[str, List[Callable]] = {}
        self.is_running = False
        
    def register_agent(self, agent: 'Agent'):
        """Register an agent with the message bus."""
        self.agents[agent.agent_id] = agent
        print(f"Registered agent: {agent.agent_id} ({agent.name})")
        
    def unregister_agent(self, agent_id: str):
        """Unregister an agent from the message bus."""
        if agent_id in self.agents:
            del self.agents[agent_id]
            print(f"Unregistered agent: {agent_id}")
            
    async def start(self):
        """Start the message bus."""
        self.is_running = True
        print("MessageBus started")
        
    async def stop(self):
        """Stop the message bus."""
        self.is_running = False
        print("MessageBus stopped")
        
    async def send_message(self, message: Message) -> bool:
        """
        Send a message to a specific agent.
        
        Args:
            message: Message to send
            
        Returns:
            True if message was sent successfully
        """
        if not self.is_running:
            return False
            
        receiver_id = message.receiver_id
        
        if receiver_id not in self.agents:
            print(f"Warning: Receiver agent {receiver_id} not found")
            return False
            
        try:
            # Add message to receiver's queue
            await self.agents[receiver_id].message_queue.put(message)
            return True
        except Exception as e:
            print(f"Error sending message: {e}")
            return False
            
    async def broadcast_message(self, sender_id: str, content: Any, 
                              exclude_sender: bool = True) -> int:
        """
        Broadcast a message to all registered agents.
        
        Args:
            sender_id: ID of the sending agent
            content: Message content
            exclude_sender: Whether to exclude sender from broadcast
            
        Returns:
            Number of agents that received the message
        """
        if not self.is_running:
            return 0
            
        sent_count = 0
        
        for agent_id, agent in self.agents.items():
            if exclude_sender and agent_id == sender_id:
                continue
                
            message = Message(
                sender_id=sender_id,
                receiver_id=agent_id,
                content=content,
                message_type="broadcast"
            )
            
            if await self.send_message(message):
                sent_count += 1
                
        return sent_count
        
    async def send_to_group(self, sender_id: str, group_ids: List[str], 
                           content: Any) -> int:
        """
        Send a message to a specific group of agents.
        
        Args:
            sender_id: ID of the sending agent
            group_ids: List of agent IDs to send to
            content: Message content
            
        Returns:
            Number of agents that received the message
        """
        if not self.is_running:
            return 0
            
        sent_count = 0
        
        for agent_id in group_ids:
            if agent_id in self.agents:
                message = Message(
                    sender_id=sender_id,
                    receiver_id=agent_id,
                    content=content,
                    message_type="group"
                )
                
                if await self.send_message(message):
                    sent_count += 1
                    
        return sent_count
        
    def register_message_handler(self, message_type: str, handler: Callable):
        """Register a message handler for a specific message type."""
        if message_type not in self.message_handlers:
            self.message_handlers[message_type] = []
        self.message_handlers[message_type].append(handler)
        
    async def process_message(self, message: Message):
        """Process a message using registered handlers."""
        message_type = message.message_type
        
        if message_type in self.message_handlers:
            for handler in self.message_handlers[message_type]:
                try:
                    await handler(message)
                except Exception as e:
                    print(f"Error in message handler: {e}")
                    
    def get_agent_count(self) -> int:
        """Get the number of registered agents."""
        return len(self.agents)
        
    def get_agent_ids(self) -> List[str]:
        """Get list of registered agent IDs."""
        return list(self.agents.keys())
        
    def get_agent_info(self) -> Dict[str, Dict]:
        """Get information about all registered agents."""
        return {
            agent_id: {
                'name': agent.name,
                'input_size': agent.input_size,
                'output_size': agent.output_size
            }
            for agent_id, agent in self.agents.items()
        }
        
    async def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the message bus."""
        return {
            'is_running': self.is_running,
            'agent_count': self.get_agent_count(),
            'registered_agents': self.get_agent_ids(),
            'backend': self.config.backend
        }


class InMemoryMessageBus(MessageBus):
    """In-memory implementation of MessageBus."""
    
    def __init__(self, config: MessageBusConfig = None):
        super().__init__(config)
        self.message_history: List[Message] = []
        self.max_history_size = 1000
        
    async def send_message(self, message: Message) -> bool:
        """Send message with history tracking."""
        success = await super().send_message(message)
        
        if success:
            # Add to history
            self.message_history.append(message)
            
            # Trim history if too large
            if len(self.message_history) > self.max_history_size:
                self.message_history = self.message_history[-self.max_history_size:]
                
        return success
        
    def get_message_history(self, limit: int = 100) -> List[Message]:
        """Get recent message history."""
        return self.message_history[-limit:]
        
    def clear_history(self):
        """Clear message history."""
        self.message_history.clear() 
"""
Agent module for SaaS-Swarm platform.

Defines the core Agent class that encapsulates individual AI agents
with neural networks, tool integration, and feedback mechanisms.
"""

import asyncio
import uuid
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field
import numpy as np
from pydantic import BaseModel

from ..models.neural_core import TinyNeuralNetwork
from ..tools.registry import ToolRegistry


@dataclass
class AgentConfig:
    """Configuration for an Agent."""
    agent_id: str
    name: str
    input_size: int
    output_size: int
    hidden_size: int = 64
    learning_rate: float = 0.01
    enable_online_learning: bool = True
    tool_registry: Optional[ToolRegistry] = None
    custom_inference_fn: Optional[Callable] = None


class Message(BaseModel):
    """Message structure for agent communication."""
    sender_id: str
    receiver_id: str
    content: Any
    message_type: str = "data"
    timestamp: float = field(default_factory=lambda: asyncio.get_event_loop().time())


class Agent:
    """
    Individual AI agent with neural network, tool integration, and feedback handling.
    
    Each agent has:
    - A neural network for inference
    - Optional tool integration via ToolRegistry
    - Feedback mechanisms for online learning
    - Message handling capabilities
    """
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.agent_id = config.agent_id
        self.name = config.name
        self.input_size = config.input_size
        self.output_size = config.output_size
        
        # Neural network
        self.neural_network = TinyNeuralNetwork(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            output_size=config.output_size,
            learning_rate=config.learning_rate
        )
        
        # Tool integration
        self.tool_registry = config.tool_registry or ToolRegistry()
        
        # Custom inference function (overrides neural network)
        self.custom_inference_fn = config.custom_inference_fn
        
        # Online learning
        self.enable_online_learning = config.enable_online_learning
        self.feedback_history: List[Dict] = []
        
        # Message handling
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.is_running = False
        
    async def start(self):
        """Start the agent's message processing loop."""
        self.is_running = True
        asyncio.create_task(self._message_loop())
        
    async def stop(self):
        """Stop the agent's message processing loop."""
        self.is_running = False
        
    async def _message_loop(self):
        """Process incoming messages asynchronously."""
        while self.is_running:
            try:
                message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                await self._process_message(message)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Error in message loop for agent {self.agent_id}: {e}")
                
    async def _process_message(self, message: Message):
        """Process a single message."""
        if message.receiver_id == self.agent_id:
            # Process the message content
            result = await self.infer(message.content)
            
            # Send response if needed
            if hasattr(message, 'response_queue'):
                await message.response_queue.put(result)
                
    async def infer(self, input_data: Union[np.ndarray, List, Dict, str]) -> Any:
        """
        Perform inference on input data.
        
        Args:
            input_data: Input data (can be numpy array, list, dict, or string)
            
        Returns:
            Inference result
        """
        # Convert input to numpy array if needed
        if isinstance(input_data, (list, dict, str)):
            input_array = self._preprocess_input(input_data)
        else:
            input_array = input_data
            
        # Use custom inference function if provided
        if self.custom_inference_fn:
            return await self.custom_inference_fn(input_array, self.tool_registry)
        
        # Use neural network for inference
        output = self.neural_network.forward(input_array)
        return self._postprocess_output(output)
    
    def _preprocess_input(self, input_data: Union[List, Dict, str]) -> np.ndarray:
        """Preprocess input data into numpy array."""
        if isinstance(input_data, str):
            # Simple string to array conversion (can be enhanced)
            chars = [ord(c) for c in input_data[:self.input_size]]
            # Pad with zeros if needed
            while len(chars) < self.input_size:
                chars.append(0)
            return np.array(chars, dtype=np.float32)
        elif isinstance(input_data, list):
            # Ensure all elements are numbers
            numeric_data = []
            for item in input_data[:self.input_size]:
                if isinstance(item, (int, float)):
                    numeric_data.append(float(item))
                else:
                    numeric_data.append(0.0)
            # Pad with zeros if needed
            while len(numeric_data) < self.input_size:
                numeric_data.append(0.0)
            return np.array(numeric_data, dtype=np.float32)
        elif isinstance(input_data, dict):
            # Convert dict to array (simple implementation)
            values = []
            for value in list(input_data.values())[:self.input_size]:
                if isinstance(value, (int, float)):
                    values.append(float(value))
                else:
                    values.append(0.0)
            # Pad with zeros if needed
            while len(values) < self.input_size:
                values.append(0.0)
            return np.array(values, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")
    
    def _postprocess_output(self, output: np.ndarray) -> Any:
        """Postprocess neural network output."""
        # Simple conversion back to usable format
        return output.tolist()
    
    async def receive_feedback(self, feedback: Dict[str, Any]):
        """
        Receive feedback and potentially update the agent.
        
        Args:
            feedback: Dictionary containing feedback data
                     - 'reward': scalar reward value
                     - 'target': target output for supervised learning
                     - 'metadata': additional feedback information
        """
        self.feedback_history.append(feedback)
        
        if not self.enable_online_learning:
            return
            
        # Extract feedback components
        reward = feedback.get('reward', 0.0)
        target = feedback.get('target')
        
        if target is not None and len(self.feedback_history) > 0:
            # Get the last input/output pair for training
            last_feedback = self.feedback_history[-1]
            if 'input' in last_feedback and 'output' in last_feedback:
                input_data = np.array(last_feedback['input'])
                target_data = np.array(target)
                
                # Perform online training
                self.neural_network.train_step(input_data, target_data)
                
    async def send_message(self, receiver_id: str, content: Any, message_type: str = "data") -> Message:
        """Send a message to another agent."""
        message = Message(
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            content=content,
            message_type=message_type
        )
        return message
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the agent."""
        return {
            'agent_id': self.agent_id,
            'name': self.name,
            'input_size': self.input_size,
            'output_size': self.output_size,
            'feedback_count': len(self.feedback_history),
            'neural_network_state': self.neural_network.get_state()
        }
    
    def save_model(self, filepath: str):
        """Save the agent's neural network to a file."""
        self.neural_network.save(filepath)
    
    def load_model(self, filepath: str):
        """Load the agent's neural network from a file."""
        self.neural_network.load(filepath) 
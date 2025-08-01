"""
Basic tests for SaaS-Swarm platform.

Tests core functionality to ensure the platform works correctly.
"""

import pytest
import asyncio
import numpy as np

from saas_swarm.core.agent import Agent, AgentConfig
from saas_swarm.core.swarm import SwarmConfig, MeshSwarm
from saas_swarm.core.feedback_loop import FeedbackLoop, FeedbackConfig
from saas_swarm.models.neural_core import TinyNeuralNetwork
from saas_swarm.tools.registry import create_default_tool_registry


class TestAgent:
    """Test Agent functionality."""
    
    def test_agent_creation(self):
        """Test agent creation with basic configuration."""
        config = AgentConfig(
            agent_id="test-agent",
            name="TestAgent",
            input_size=10,
            output_size=5,
            hidden_size=8,
            learning_rate=0.01
        )
        
        agent = Agent(config)
        
        assert agent.agent_id == "test-agent"
        assert agent.name == "TestAgent"
        assert agent.input_size == 10
        assert agent.output_size == 5
        assert agent.neural_network is not None
        
    def test_agent_inference(self):
        """Test agent inference functionality."""
        config = AgentConfig(
            agent_id="test-agent",
            name="TestAgent",
            input_size=5,
            output_size=3,
            hidden_size=4
        )
        
        agent = Agent(config)
        
        # Test with numpy array
        input_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = asyncio.run(agent.infer(input_data))
        
        assert isinstance(result, list)
        assert len(result) == 3
        
    def test_agent_feedback(self):
        """Test agent feedback handling."""
        config = AgentConfig(
            agent_id="test-agent",
            name="TestAgent",
            input_size=5,
            output_size=3,
            hidden_size=4,
            enable_online_learning=True
        )
        
        agent = Agent(config)
        
        feedback = {
            'reward': 0.8,
            'target': [0.1, 0.2, 0.3],
            'metadata': {'test': True}
        }
        
        asyncio.run(agent.receive_feedback(feedback))
        
        assert len(agent.feedback_history) == 1
        assert agent.feedback_history[0]['reward'] == 0.8


class TestSwarm:
    """Test Swarm functionality."""
    
    def test_mesh_swarm_creation(self):
        """Test mesh swarm creation."""
        config = SwarmConfig(
            swarm_id="test-swarm",
            name="TestSwarm",
            topology_type="mesh"
        )
        
        swarm = MeshSwarm(config)
        
        assert swarm.swarm_id == "test-swarm"
        assert swarm.name == "TestSwarm"
        assert len(swarm.agents) == 0
        
    def test_swarm_add_agent(self):
        """Test adding agents to swarm."""
        swarm_config = SwarmConfig(
            swarm_id="test-swarm",
            name="TestSwarm",
            topology_type="mesh"
        )
        
        agent_config = AgentConfig(
            agent_id="test-agent",
            name="TestAgent",
            input_size=5,
            output_size=3,
            hidden_size=4
        )
        
        swarm = MeshSwarm(swarm_config)
        agent = Agent(agent_config)
        
        success = swarm.add_agent(agent)
        assert success is True
        assert len(swarm.agents) == 1
        assert "test-agent" in swarm.agents
        
    @pytest.mark.asyncio
    async def test_swarm_execution(self):
        """Test swarm execution."""
        swarm_config = SwarmConfig(
            swarm_id="test-swarm",
            name="TestSwarm",
            topology_type="mesh"
        )
        
        agent_config = AgentConfig(
            agent_id="test-agent",
            name="TestAgent",
            input_size=5,
            output_size=3,
            hidden_size=4
        )
        
        swarm = MeshSwarm(swarm_config)
        agent = Agent(agent_config)
        swarm.add_agent(agent)
        
        await swarm.start()
        
        input_data = "test input"
        result = await swarm.execute(input_data)
        
        assert 'results' in result
        assert result['success'] is True
        
        await swarm.stop()


class TestNeuralNetwork:
    """Test neural network functionality."""
    
    def test_neural_network_creation(self):
        """Test neural network creation."""
        nn = TinyNeuralNetwork(
            input_size=10,
            hidden_size=8,
            output_size=5,
            learning_rate=0.01
        )
        
        assert nn.input_size == 10
        assert nn.hidden_size == 8
        assert nn.output_size == 5
        assert nn.learning_rate == 0.01
        
    def test_neural_network_forward(self):
        """Test neural network forward pass."""
        nn = TinyNeuralNetwork(
            input_size=5,
            hidden_size=4,
            output_size=3,
            learning_rate=0.01
        )
        
        input_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        output = nn.forward(input_data)
        
        assert output.shape == (1, 3)
        assert not np.isnan(output).any()
        
    def test_neural_network_training(self):
        """Test neural network training."""
        nn = TinyNeuralNetwork(
            input_size=5,
            hidden_size=4,
            output_size=3,
            learning_rate=0.01
        )
        
        input_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        target_data = np.array([0.1, 0.2, 0.3])
        
        loss = nn.train_step(input_data, target_data)
        
        assert isinstance(loss, float)
        assert not np.isnan(loss)


class TestToolRegistry:
    """Test tool registry functionality."""
    
    def test_tool_registry_creation(self):
        """Test tool registry creation."""
        registry = create_default_tool_registry()
        
        assert len(registry.tools) > 0
        assert "summarize" in registry.tools
        assert "classify" in registry.tools
        
    @pytest.mark.asyncio
    async def test_tool_execution(self):
        """Test tool execution."""
        registry = create_default_tool_registry()
        
        # Test sync tool
        result = await registry.call_tool("summarize", "This is a very long text that needs to be summarized to a shorter version", 20)
        assert isinstance(result, str)
        assert len(result) <= 20
        
        # Test async tool
        result = await registry.call_tool("web_search", "artificial intelligence")
        assert isinstance(result, dict)
        assert "query" in result
        assert "results" in result


class TestFeedbackLoop:
    """Test feedback loop functionality."""
    
    def test_feedback_loop_creation(self):
        """Test feedback loop creation."""
        config = FeedbackConfig()
        feedback_loop = FeedbackLoop(config)
        
        assert feedback_loop.config == config
        assert len(feedback_loop.agents) == 0
        
    @pytest.mark.asyncio
    async def test_feedback_loop_evaluation(self):
        """Test feedback loop evaluation."""
        config = FeedbackConfig()
        feedback_loop = FeedbackLoop(config)
        
        await feedback_loop.start()
        
        swarm_output = {
            'results': {'test': 'data'},
            'total_agents': 2
        }
        
        evaluation = await feedback_loop.evaluate_swarm_output(swarm_output)
        
        assert 'score' in evaluation
        assert 'recommendation' in evaluation
        assert isinstance(evaluation['score'], float)
        
        await feedback_loop.stop()


if __name__ == "__main__":
    pytest.main([__file__]) 
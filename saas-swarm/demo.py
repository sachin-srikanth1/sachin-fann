#!/usr/bin/env python3
"""
SaaS-Swarm Platform Demo

This script demonstrates the complete SaaS-Swarm platform
with all its features and capabilities.
"""

import asyncio
import sys
import os
import time

# Add the package to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from saas_swarm.core.agent import Agent, AgentConfig
from saas_swarm.core.swarm import SwarmConfig, MeshSwarm, StarSwarm, RingSwarm, HierarchicalSwarm
from saas_swarm.core.feedback_loop import FeedbackLoop, FeedbackConfig
from saas_swarm.models.neural_core import TinyNeuralNetwork
from saas_swarm.tools.registry import create_default_tool_registry
from saas_swarm.examples.email_writer import create_email_writer_swarm
from saas_swarm.examples.route_optimizer import create_route_optimizer_swarm
from saas_swarm.examples.code_review import create_code_review_swarm


async def demo_basic_agent():
    """Demo basic agent functionality."""
    print("\n" + "="*50)
    print("BASIC AGENT DEMO")
    print("="*50)
    
    # Create a simple agent
    config = AgentConfig(
        agent_id="demo-agent",
        name="DemoAgent",
        input_size=10,
        output_size=5,
        hidden_size=8,
        learning_rate=0.01,
        enable_online_learning=True
    )
    
    agent = Agent(config)
    
    # Test inference
    input_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    result = await agent.infer(input_data)
    
    print(f"Agent inference result: {result}")
    
    # Test feedback
    feedback = {
        'reward': 0.8,
        'target': [0.1, 0.2, 0.3, 0.4, 0.5],
        'metadata': {'demo': True}
    }
    
    await agent.receive_feedback(feedback)
    print(f"Agent received feedback, history length: {len(agent.feedback_history)}")
    
    return agent


async def demo_tool_registry():
    """Demo tool registry functionality."""
    print("\n" + "="*50)
    print("TOOL REGISTRY DEMO")
    print("="*50)
    
    registry = create_default_tool_registry()
    
    print(f"Available tools: {registry.list_tools()}")
    
    # Test text summarization
    long_text = "This is a very long text that contains many words and needs to be summarized to a shorter version for better readability and understanding."
    summary = await registry.call_tool("summarize", long_text, 30)
    print(f"Text summarization: {summary}")
    
    # Test text classification
    categories = ["technology", "science", "politics", "sports"]
    classification = await registry.call_tool("classify", "artificial intelligence and machine learning", categories)
    print(f"Text classification: {classification}")
    
    # Test data transformation
    transformed = await registry.call_tool("transform", "hello world", "uppercase")
    print(f"Data transformation: {transformed}")
    
    return registry


async def demo_neural_network():
    """Demo neural network functionality."""
    print("\n" + "="*50)
    print("NEURAL NETWORK DEMO")
    print("="*50)
    
    # Create neural network
    nn = TinyNeuralNetwork(
        input_size=5,
        hidden_size=8,
        output_size=3,
        learning_rate=0.01
    )
    
    print(f"Neural network parameters: {nn.get_parameter_count()}")
    print(f"Memory usage: {nn.get_memory_usage()} bytes")
    
    # Test forward pass
    input_data = [1.0, 2.0, 3.0, 4.0, 5.0]
    output = nn.forward(input_data)
    print(f"Forward pass output: {output}")
    
    # Test training
    target = [0.1, 0.2, 0.3]
    loss = nn.train_step(input_data, target)
    print(f"Training loss: {loss}")
    
    # Test state
    state = nn.get_state()
    print(f"Neural network state: {state}")
    
    return nn


async def demo_swarm_topologies():
    """Demo different swarm topologies."""
    print("\n" + "="*50)
    print("SWARM TOPOLOGIES DEMO")
    print("="*50)
    
    # Create agents for different topologies
    agents = []
    for i in range(3):
        config = AgentConfig(
            agent_id=f"agent-{i}",
            name=f"Agent{i}",
            input_size=5,
            output_size=3,
            hidden_size=4
        )
        agents.append(Agent(config))
    
    topologies = [
        ("Mesh", MeshSwarm),
        ("Star", StarSwarm),
        ("Ring", RingSwarm),
        ("Hierarchical", HierarchicalSwarm)
    ]
    
    for name, SwarmClass in topologies:
        print(f"\n--- {name} Topology ---")
        
        config = SwarmConfig(
            swarm_id=f"demo-{name.lower()}",
            name=f"Demo{name}",
            topology_type=name.lower(),
            max_execution_time=10.0
        )
        
        swarm = SwarmClass(config)
        
        # Add agents
        for agent in agents:
            if name == "Hierarchical":
                swarm.add_agent(agent, parent_id=None if len(swarm.agents) == 0 else list(swarm.agents.keys())[0])
            else:
                swarm.add_agent(agent)
        
        print(f"Added {len(agents)} agents to {name} swarm")
        
        # Start and execute
        await swarm.start()
        
        input_data = f"test input for {name} topology"
        result = await swarm.execute(input_data)
        
        print(f"Execution result: {result.get('success', False)}")
        
        await swarm.stop()


async def demo_feedback_loop():
    """Demo feedback loop functionality."""
    print("\n" + "="*50)
    print("FEEDBACK LOOP DEMO")
    print("="*50)
    
    # Create feedback loop
    config = FeedbackConfig()
    feedback_loop = FeedbackLoop(config)
    
    await feedback_loop.start()
    
    # Create test agent
    agent_config = AgentConfig(
        agent_id="feedback-agent",
        name="FeedbackAgent",
        input_size=5,
        output_size=3,
        hidden_size=4
    )
    
    agent = Agent(agent_config)
    feedback_loop.register_agent(agent)
    
    # Test evaluation
    swarm_output = {
        'results': {'test': 'data', 'score': 0.8},
        'total_agents': 1,
        'execution_time': 1.5
    }
    
    evaluation = await feedback_loop.evaluate_swarm_output(swarm_output)
    print(f"Evaluation result: {evaluation}")
    
    # Test feedback propagation
    propagation = await feedback_loop.propagate_feedback(evaluation)
    print(f"Feedback propagation: {propagation}")
    
    await feedback_loop.stop()


async def demo_example_swarms():
    """Demo the example swarms."""
    print("\n" + "="*50)
    print("EXAMPLE SWARMS DEMO")
    print("="*50)
    
    examples = [
        ("Email Writer", create_email_writer_swarm),
        ("Route Optimizer", create_route_optimizer_swarm),
        ("Code Review", create_code_review_swarm)
    ]
    
    for name, create_func in examples:
        print(f"\n--- {name} Swarm ---")
        
        try:
            swarm = create_func()
            await swarm.start()
            
            # Test with sample input
            if name == "Email Writer":
                test_input = "artificial intelligence trends"
            elif name == "Route Optimizer":
                test_input = {
                    'locations': ['A', 'B', 'C'],
                    'description': 'Simple route'
                }
            else:  # Code Review
                test_input = '''
def test_function():
    return "hello world"
'''
            
            result = await swarm.execute(test_input, timeout=15.0)
            print(f"Execution completed: {result.get('success', False)}")
            
            await swarm.stop()
            
        except Exception as e:
            print(f"Error in {name} demo: {e}")


async def run_complete_demo():
    """Run the complete platform demo."""
    print("🚀 SaaS-SWARM PLATFORM COMPLETE DEMO")
    print("="*60)
    
    start_time = time.time()
    
    try:
        # Run all demos
        await demo_basic_agent()
        await demo_tool_registry()
        await demo_neural_network()
        await demo_swarm_topologies()
        await demo_feedback_loop()
        await demo_example_swarms()
        
        elapsed_time = time.time() - start_time
        
        print("\n" + "="*60)
        print("🎉 DEMO COMPLETED SUCCESSFULLY!")
        print(f"⏱️  Total time: {elapsed_time:.2f} seconds")
        print("="*60)
        
        print("\n📚 Next Steps:")
        print("- Run 'swarm --help' for CLI commands")
        print("- Start API server: python -m saas_swarm.api.main")
        print("- Run examples: python run_examples.py")
        print("- Run tests: python -m pytest tests/")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        asyncio.run(run_complete_demo())
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\nDemo failed: {e}")
        sys.exit(1) 
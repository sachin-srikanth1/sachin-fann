"""
CLI main application for SaaS-Swarm platform.

Provides command-line interface for:
- Swarm creation and management
- Agent configuration
- Swarm deployment and execution
- Configuration management
"""

import asyncio
import json
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
import click

from ..core.agent import Agent, AgentConfig
from ..core.swarm import SwarmConfig, MeshSwarm, StarSwarm, RingSwarm, HierarchicalSwarm
from ..core.feedback_loop import FeedbackLoop, FeedbackConfig
from ..tools.registry import create_default_tool_registry


# Global state for CLI
swarms: Dict[str, Any] = {}
agents: Dict[str, Agent] = {}
feedback_loop = FeedbackLoop(FeedbackConfig())
tool_registry = create_default_tool_registry()


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """SaaS-Swarm CLI - Swarm-as-a-Service platform command line interface."""
    pass


@cli.group()
def swarm():
    """Swarm management commands."""
    pass


@swarm.command()
@click.argument('name')
@click.option('--topology', '-t', default='mesh', 
              type=click.Choice(['mesh', 'star', 'ring', 'hierarchical']),
              help='Swarm topology type')
@click.option('--timeout', default=30.0, help='Maximum execution time')
@click.option('--feedback/--no-feedback', default=True, help='Enable feedback loop')
def new(name: str, topology: str, timeout: float, feedback: bool):
    """Create a new swarm."""
    import uuid
    
    swarm_id = str(uuid.uuid4())
    
    config = SwarmConfig(
        swarm_id=swarm_id,
        name=name,
        topology_type=topology,
        max_execution_time=timeout,
        enable_feedback=feedback
    )
    
    # Create swarm based on topology type
    if topology == "mesh":
        swarm = MeshSwarm(config)
    elif topology == "star":
        swarm = StarSwarm(config)
    elif topology == "ring":
        swarm = RingSwarm(config)
    elif topology == "hierarchical":
        swarm = HierarchicalSwarm(config)
    
    swarms[swarm_id] = swarm
    
    click.echo(f"Created swarm '{name}' with ID: {swarm_id}")
    click.echo(f"Topology: {topology}")
    click.echo(f"Timeout: {timeout}s")
    click.echo(f"Feedback: {'enabled' if feedback else 'disabled'}")


@swarm.command()
def list():
    """List all swarms."""
    if not swarms:
        click.echo("No swarms found.")
        return
    
    for swarm_id, swarm in swarms.items():
        click.echo(f"ID: {swarm_id}")
        click.echo(f"Name: {swarm.name}")
        click.echo(f"Topology: {swarm.config.topology_type}")
        click.echo(f"Agents: {len(swarm.agents)}")
        click.echo(f"Running: {swarm.is_running}")
        click.echo("---")


@swarm.command()
@click.argument('swarm_id')
async def start(swarm_id: str):
    """Start a swarm."""
    if swarm_id not in swarms:
        click.echo(f"Swarm {swarm_id} not found.")
        return
    
    swarm = swarms[swarm_id]
    await swarm.start()
    click.echo(f"Started swarm {swarm_id}")


@swarm.command()
@click.argument('swarm_id')
async def stop(swarm_id: str):
    """Stop a swarm."""
    if swarm_id not in swarms:
        click.echo(f"Swarm {swarm_id} not found.")
        return
    
    swarm = swarms[swarm_id]
    await swarm.stop()
    click.echo(f"Stopped swarm {swarm_id}")


@swarm.command()
@click.argument('swarm_id')
@click.argument('input_data')
@click.option('--timeout', default=None, type=float, help='Execution timeout')
async def run(swarm_id: str, input_data: str, timeout: Optional[float]):
    """Run a swarm with input data."""
    if swarm_id not in swarms:
        click.echo(f"Swarm {swarm_id} not found.")
        return
    
    swarm = swarms[swarm_id]
    
    if not swarm.is_running:
        click.echo("Starting swarm...")
        await swarm.start()
    
    try:
        # Try to parse input as JSON, fallback to string
        try:
            parsed_input = json.loads(input_data)
        except json.JSONDecodeError:
            parsed_input = input_data
        
        result = await swarm.execute(parsed_input, timeout)
        
        click.echo("Execution completed:")
        click.echo(json.dumps(result, indent=2))
        
    except Exception as e:
        click.echo(f"Execution failed: {e}")


@cli.group()
def agent():
    """Agent management commands."""
    pass


@agent.command()
@click.argument('name')
@click.option('--input-size', '-i', default=64, help='Input size')
@click.option('--output-size', '-o', default=10, help='Output size')
@click.option('--hidden-size', '-h', default=64, help='Hidden layer size')
@click.option('--learning-rate', '-l', default=0.01, help='Learning rate')
@click.option('--online-learning/--no-online-learning', default=True, help='Enable online learning')
def create(name: str, input_size: int, output_size: int, hidden_size: int, 
           learning_rate: float, online_learning: bool):
    """Create a new agent."""
    import uuid
    
    agent_id = str(uuid.uuid4())
    
    config = AgentConfig(
        agent_id=agent_id,
        name=name,
        input_size=input_size,
        output_size=output_size,
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        enable_online_learning=online_learning,
        tool_registry=tool_registry
    )
    
    agent = Agent(config)
    agents[agent_id] = agent
    
    # Register with feedback loop
    feedback_loop.register_agent(agent)
    
    click.echo(f"Created agent '{name}' with ID: {agent_id}")
    click.echo(f"Input size: {input_size}")
    click.echo(f"Output size: {output_size}")
    click.echo(f"Hidden size: {hidden_size}")
    click.echo(f"Learning rate: {learning_rate}")
    click.echo(f"Online learning: {'enabled' if online_learning else 'disabled'}")


@agent.command()
def list():
    """List all agents."""
    if not agents:
        click.echo("No agents found.")
        return
    
    for agent_id, agent in agents.items():
        click.echo(f"ID: {agent_id}")
        click.echo(f"Name: {agent.name}")
        click.echo(f"Input size: {agent.input_size}")
        click.echo(f"Output size: {agent.output_size}")
        click.echo(f"Feedback count: {len(agent.feedback_history)}")
        click.echo("---")


@swarm.command()
@click.argument('swarm_id')
@click.argument('agent_id')
def add_agent(swarm_id: str, agent_id: str):
    """Add an agent to a swarm."""
    if swarm_id not in swarms:
        click.echo(f"Swarm {swarm_id} not found.")
        return
    
    if agent_id not in agents:
        click.echo(f"Agent {agent_id} not found.")
        return
    
    swarm = swarms[swarm_id]
    agent = agents[agent_id]
    
    success = swarm.add_agent(agent)
    if success:
        click.echo(f"Added agent {agent_id} to swarm {swarm_id}")
    else:
        click.echo(f"Failed to add agent {agent_id} to swarm {swarm_id}")


@swarm.command()
@click.argument('swarm_id')
@click.argument('agent_id')
def remove_agent(swarm_id: str, agent_id: str):
    """Remove an agent from a swarm."""
    if swarm_id not in swarms:
        click.echo(f"Swarm {swarm_id} not found.")
        return
    
    swarm = swarms[swarm_id]
    success = swarm.remove_agent(agent_id)
    
    if success:
        click.echo(f"Removed agent {agent_id} from swarm {swarm_id}")
    else:
        click.echo(f"Agent {agent_id} not found in swarm {swarm_id}")


@cli.group()
def config():
    """Configuration management commands."""
    pass


@config.command()
@click.argument('filename')
def save(filename: str):
    """Save current configuration to file."""
    config_data = {
        'swarms': {
            swarm_id: {
                'name': swarm.name,
                'topology_type': swarm.config.topology_type,
                'max_execution_time': swarm.config.max_execution_time,
                'enable_feedback': swarm.config.enable_feedback,
                'agents': list(swarm.agents.keys())
            }
            for swarm_id, swarm in swarms.items()
        },
        'agents': {
            agent_id: {
                'name': agent.name,
                'input_size': agent.input_size,
                'output_size': agent.output_size,
                'hidden_size': agent.config.hidden_size,
                'learning_rate': agent.config.learning_rate,
                'enable_online_learning': agent.config.enable_online_learning
            }
            for agent_id, agent in agents.items()
        }
    }
    
    filepath = Path(filename)
    if filepath.suffix.lower() == '.yaml' or filepath.suffix.lower() == '.yml':
        with open(filepath, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)
    else:
        with open(filepath, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    click.echo(f"Configuration saved to {filename}")


@config.command()
@click.argument('filename')
def load(filename: str):
    """Load configuration from file."""
    filepath = Path(filename)
    
    if not filepath.exists():
        click.echo(f"Configuration file {filename} not found.")
        return
    
    if filepath.suffix.lower() == '.yaml' or filepath.suffix.lower() == '.yml':
        with open(filepath, 'r') as f:
            config_data = yaml.safe_load(f)
    else:
        with open(filepath, 'r') as f:
            config_data = json.load(f)
    
    # Clear existing state
    swarms.clear()
    agents.clear()
    
    # Load agents
    for agent_id, agent_data in config_data.get('agents', {}).items():
        config = AgentConfig(
            agent_id=agent_id,
            name=agent_data['name'],
            input_size=agent_data['input_size'],
            output_size=agent_data['output_size'],
            hidden_size=agent_data.get('hidden_size', 64),
            learning_rate=agent_data.get('learning_rate', 0.01),
            enable_online_learning=agent_data.get('enable_online_learning', True),
            tool_registry=tool_registry
        )
        
        agent = Agent(config)
        agents[agent_id] = agent
        feedback_loop.register_agent(agent)
    
    # Load swarms
    for swarm_id, swarm_data in config_data.get('swarms', {}).items():
        config = SwarmConfig(
            swarm_id=swarm_id,
            name=swarm_data['name'],
            topology_type=swarm_data['topology_type'],
            max_execution_time=swarm_data.get('max_execution_time', 30.0),
            enable_feedback=swarm_data.get('enable_feedback', True)
        )
        
        # Create swarm based on topology type
        topology_type = swarm_data['topology_type']
        if topology_type == "mesh":
            swarm = MeshSwarm(config)
        elif topology_type == "star":
            swarm = StarSwarm(config)
        elif topology_type == "ring":
            swarm = RingSwarm(config)
        elif topology_type == "hierarchical":
            swarm = HierarchicalSwarm(config)
        
        # Add agents to swarm
        for agent_id in swarm_data.get('agents', []):
            if agent_id in agents:
                swarm.add_agent(agents[agent_id])
        
        swarms[swarm_id] = swarm
    
    click.echo(f"Configuration loaded from {filename}")
    click.echo(f"Loaded {len(agents)} agents and {len(swarms)} swarms")


@cli.command()
def status():
    """Show system status."""
    click.echo("SaaS-Swarm System Status")
    click.echo("=" * 30)
    click.echo(f"Swarms: {len(swarms)}")
    click.echo(f"Agents: {len(agents)}")
    click.echo(f"Feedback loop: {'running' if feedback_loop.is_running else 'stopped'}")
    click.echo(f"Tools: {len(tool_registry.tools)}")
    
    if swarms:
        click.echo("\nSwarm Details:")
        for swarm_id, swarm in swarms.items():
            click.echo(f"  {swarm.name} ({swarm_id}): {len(swarm.agents)} agents, {'running' if swarm.is_running else 'stopped'}")


@cli.command()
def deploy():
    """Deploy all swarms."""
    async def deploy_swarms():
        for swarm_id, swarm in swarms.items():
            if not swarm.is_running:
                await swarm.start()
                click.echo(f"Deployed swarm {swarm_id}")
    
    asyncio.run(deploy_swarms())


if __name__ == '__main__':
    cli() 
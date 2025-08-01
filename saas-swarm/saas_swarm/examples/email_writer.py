"""
Email Writer Swarm Example.

Demonstrates a swarm with research and writing agents
collaborating to create emails based on topics.
"""

import asyncio
from typing import Dict, Any, List
import uuid

from saas_swarm.core.agent import Agent, AgentConfig
from saas_swarm.core.swarm import SwarmConfig, StarSwarm
from saas_swarm.core.feedback_loop import FeedbackLoop, FeedbackConfig
from saas_swarm.tools.registry import create_default_tool_registry


async def research_agent_inference(input_data: Any, tool_registry) -> Dict[str, Any]:
    """Custom inference function for research agent."""
    topic = str(input_data)
    
    # Use web search tool to gather information
    search_results = await tool_registry.call_tool("web_search", f"latest news about {topic}")
    
    # Summarize the research
    summary = await tool_registry.call_tool("summarize", str(search_results), 200)
    
    return {
        'topic': topic,
        'research': search_results,
        'summary': summary,
        'key_points': [
            f"Key point 1 about {topic}",
            f"Key point 2 about {topic}",
            f"Key point 3 about {topic}"
        ]
    }


async def writer_agent_inference(input_data: Any, tool_registry) -> str:
    """Custom inference function for writer agent."""
    if isinstance(input_data, dict) and 'research' in input_data:
        research = input_data
        topic = research.get('topic', 'unknown topic')
        summary = research.get('summary', '')
        key_points = research.get('key_points', [])
        
        # Generate email content
        email_content = f"""
Subject: Latest Updates on {topic}

Dear Team,

I wanted to share some important updates regarding {topic}.

{summary}

Key highlights:
"""
        
        for i, point in enumerate(key_points, 1):
            email_content += f"{i}. {point}\n"
        
        email_content += f"""
Please let me know if you have any questions or need additional information.

Best regards,
AI Assistant
"""
        
        return email_content
    else:
        return f"Unable to generate email for: {input_data}"


def create_email_writer_swarm() -> StarSwarm:
    """
    Create an email writer swarm with research and writing agents.
    
    Returns:
        Configured StarSwarm instance
    """
    # Create tool registry
    tool_registry = create_default_tool_registry()
    
    # Create feedback loop
    feedback_loop = FeedbackLoop(FeedbackConfig())
    
    # Create research agent
    research_agent_id = str(uuid.uuid4())
    research_config = AgentConfig(
        agent_id=research_agent_id,
        name="ResearchAgent",
        input_size=100,
        output_size=50,
        hidden_size=64,
        learning_rate=0.01,
        enable_online_learning=True,
        tool_registry=tool_registry,
        custom_inference_fn=research_agent_inference
    )
    research_agent = Agent(research_config)
    
    # Create writer agent
    writer_agent_id = str(uuid.uuid4())
    writer_config = AgentConfig(
        agent_id=writer_agent_id,
        name="WriterAgent",
        input_size=200,
        output_size=100,
        hidden_size=64,
        learning_rate=0.01,
        enable_online_learning=True,
        tool_registry=tool_registry,
        custom_inference_fn=writer_agent_inference
    )
    writer_agent = Agent(writer_config)
    
    # Register agents with feedback loop
    feedback_loop.register_agent(research_agent)
    feedback_loop.register_agent(writer_agent)
    
    # Create star swarm
    swarm_config = SwarmConfig(
        swarm_id=str(uuid.uuid4()),
        name="EmailWriterSwarm",
        topology_type="star",
        max_execution_time=60.0,
        enable_feedback=True
    )
    
    swarm = StarSwarm(swarm_config)
    
    # Add agents to swarm
    swarm.add_agent(research_agent)
    swarm.add_agent(writer_agent)
    
    # Set research agent as hub
    swarm.set_hub_agent(research_agent_id)
    
    return swarm


async def run_email_writer_example():
    """Run the email writer example."""
    print("Creating Email Writer Swarm...")
    swarm = create_email_writer_swarm()
    
    # Start the swarm
    await swarm.start()
    print("Swarm started!")
    
    # Test with different topics
    topics = [
        "artificial intelligence trends",
        "climate change initiatives",
        "remote work best practices"
    ]
    
    for topic in topics:
        print(f"\n--- Generating email about: {topic} ---")
        
        try:
            result = await swarm.execute(topic, timeout=30.0)
            
            if 'results' in result:
                print("Generated Email:")
                print(result['results'].get('final_result', 'No result generated'))
            else:
                print("Execution failed:", result)
                
        except Exception as e:
            print(f"Error: {e}")
    
    # Stop the swarm
    await swarm.stop()
    print("\nSwarm stopped!")


if __name__ == "__main__":
    asyncio.run(run_email_writer_example()) 
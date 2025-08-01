"""
Enhanced Email Writer Swarm Example.

Demonstrates a swarm with research and writing agents
using OpenAI for research and email sending capabilities.
"""

import asyncio
from typing import Dict, Any, List
import uuid

from saas_swarm.core.agent import Agent, AgentConfig
from saas_swarm.core.swarm import SwarmConfig, StarSwarm
from saas_swarm.core.feedback_loop import FeedbackLoop, FeedbackConfig
from saas_swarm.tools.registry import create_default_tool_registry
from saas_swarm.config import Config


async def enhanced_research_agent_inference(input_data: Any, tool_registry) -> Dict[str, Any]:
    """Enhanced research agent using OpenAI."""
    topic = str(input_data)
    
    # Use OpenAI for research
    research_result = await tool_registry.call_tool("openai_research", topic)
    
    if research_result.get('status') == 'success':
        research_content = research_result['research']
        
        # Summarize the research
        summary = await tool_registry.call_tool("summarize", research_content, 200)
        
        return {
            'topic': topic,
            'research': research_content,
            'summary': summary,
            'model': research_result.get('model', 'unknown'),
            'status': 'success'
        }
    else:
        # Fallback to mock research if OpenAI fails
        return {
            'topic': topic,
            'research': f"Research about {topic} - Key insights and trends",
            'summary': f"Summary of {topic} research",
            'status': 'fallback'
        }


async def enhanced_writer_agent_inference(input_data: Any, tool_registry) -> str:
    """Enhanced writer agent that can send emails."""
    if isinstance(input_data, dict) and 'research' in input_data:
        research = input_data
        topic = research.get('topic', 'unknown topic')
        research_content = research.get('research', '')
        summary = research.get('summary', '')
        
        # Generate email content using OpenAI
        email_prompt = f"""
        Based on the following research about {topic}, write a professional email:
        
        Research Summary: {summary}
        
        Please write an email that:
        1. Has a clear subject line
        2. Introduces the topic professionally
        3. Shares key insights from the research
        4. Includes actionable recommendations
        5. Has a professional closing
        
        Format the email properly with subject, body, and signature.
        """
        
        email_result = await tool_registry.call_tool("openai_generate", email_prompt)
        
        if email_result.get('status') == 'success':
            return email_result['content']
        else:
            # Fallback email generation
            return f"""
Subject: Research Update on {topic}

Dear Team,

I wanted to share some important insights from our research on {topic}.

{summary}

Key highlights from our analysis:
- Important development 1
- Important development 2  
- Important development 3

Please let me know if you have any questions or need additional information.

Best regards,
AI Research Assistant
"""
    else:
        return f"Unable to generate email for: {input_data}"


async def execute_agent_inference(input_data: Any, tool_registry) -> Dict[str, Any]:
    """Execute agent that sends emails."""
    if isinstance(input_data, str):
        # Parse email content (simple parsing)
        lines = input_data.split('\n')
        subject = ""
        body = ""
        to_address = "recipient@example.com"  # Default recipient
        
        # Extract subject and body
        for line in lines:
            if line.startswith('Subject:'):
                subject = line.replace('Subject:', '').strip()
            elif line.startswith('To:'):
                to_address = line.replace('To:', '').strip()
            elif line and not line.startswith('Subject:') and not line.startswith('To:'):
                body += line + '\n'
        
        if not subject:
            subject = "Research Update"
        
        # Send the email
        email_result = await tool_registry.call_tool(
            "send_email",
            to_address=to_address,
            subject=subject,
            body=body.strip()
        )
        
        return {
            'email_sent': email_result.get('status') == 'sent',
            'email_result': email_result,
            'to_address': to_address,
            'subject': subject,
            'content': input_data
        }
    else:
        return {
            'email_sent': False,
            'error': 'Invalid input format for email sending'
        }


def create_enhanced_email_writer_swarm() -> StarSwarm:
    """
    Create an enhanced email writer swarm with OpenAI research and email sending.
    
    Returns:
        Configured StarSwarm instance
    """
    # Validate configuration
    if not Config.validate():
        raise ValueError("Missing required configuration. Please check your .env file.")
    
    # Create tool registry
    tool_registry = create_default_tool_registry()
    
    # Create feedback loop
    feedback_loop = FeedbackLoop(FeedbackConfig())
    
    # Create enhanced research agent
    research_agent_id = str(uuid.uuid4())
    research_config = AgentConfig(
        agent_id=research_agent_id,
        name="EnhancedResearchAgent",
        input_size=100,
        output_size=50,
        hidden_size=64,
        learning_rate=0.01,
        enable_online_learning=True,
        tool_registry=tool_registry,
        custom_inference_fn=enhanced_research_agent_inference
    )
    research_agent = Agent(research_config)
    
    # Create enhanced writer agent
    writer_agent_id = str(uuid.uuid4())
    writer_config = AgentConfig(
        agent_id=writer_agent_id,
        name="EnhancedWriterAgent",
        input_size=200,
        output_size=100,
        hidden_size=64,
        learning_rate=0.01,
        enable_online_learning=True,
        tool_registry=tool_registry,
        custom_inference_fn=enhanced_writer_agent_inference
    )
    writer_agent = Agent(writer_config)
    
    # Create execute agent
    execute_agent_id = str(uuid.uuid4())
    execute_config = AgentConfig(
        agent_id=execute_agent_id,
        name="ExecuteAgent",
        input_size=200,
        output_size=50,
        hidden_size=64,
        learning_rate=0.01,
        enable_online_learning=True,
        tool_registry=tool_registry,
        custom_inference_fn=execute_agent_inference
    )
    execute_agent = Agent(execute_config)
    
    # Register agents with feedback loop
    feedback_loop.register_agent(research_agent)
    feedback_loop.register_agent(writer_agent)
    feedback_loop.register_agent(execute_agent)
    
    # Create star swarm
    swarm_config = SwarmConfig(
        swarm_id=str(uuid.uuid4()),
        name="EnhancedEmailWriterSwarm",
        topology_type="star",
        max_execution_time=120.0,  # Longer timeout for OpenAI calls
        enable_feedback=True
    )
    
    swarm = StarSwarm(swarm_config)
    
    # Add agents to swarm
    swarm.add_agent(research_agent)
    swarm.add_agent(writer_agent)
    swarm.add_agent(execute_agent)
    
    # Set research agent as hub
    swarm.set_hub_agent(research_agent_id)
    
    return swarm


async def run_enhanced_email_writer_example():
    """Run the enhanced email writer example."""
    print("Creating Enhanced Email Writer Swarm...")
    
    try:
        swarm = create_enhanced_email_writer_swarm()
        
        # Start the swarm
        await swarm.start()
        print("Swarm started!")
        
        # Test with different topics
        topics = [
            "artificial intelligence trends 2024",
            "climate change initiatives",
            "remote work best practices"
        ]
        
        for topic in topics:
            print(f"\n--- Generating email about: {topic} ---")
            
            try:
                result = await swarm.execute(topic, timeout=60.0)
                
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
        
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("Please ensure your .env file is properly configured.")


if __name__ == "__main__":
    asyncio.run(run_enhanced_email_writer_example()) 
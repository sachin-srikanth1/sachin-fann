"""
FastAPI main application for SaaS-Swarm platform.

Provides REST endpoints for:
- Swarm management
- Agent operations
- Feedback submission
- Real-time monitoring
"""

import asyncio
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid

from ..core.agent import Agent, AgentConfig
from ..core.swarm import SwarmConfig, MeshSwarm, StarSwarm, RingSwarm, HierarchicalSwarm
from ..core.feedback_loop import FeedbackLoop, FeedbackConfig
from ..tools.registry import create_default_tool_registry


# Pydantic models for API requests/responses
class SwarmCreateRequest(BaseModel):
    name: str
    topology_type: str = "mesh"
    max_execution_time: float = 30.0
    enable_feedback: bool = True


class AgentCreateRequest(BaseModel):
    name: str
    input_size: int
    output_size: int
    hidden_size: int = 64
    learning_rate: float = 0.01
    enable_online_learning: bool = True


class SwarmRunRequest(BaseModel):
    input_data: Any
    timeout: Optional[float] = None


class FeedbackRequest(BaseModel):
    reward: float
    target: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None


class SwarmResponse(BaseModel):
    swarm_id: str
    name: str
    topology_type: str
    agent_count: int
    is_running: bool


class AgentResponse(BaseModel):
    agent_id: str
    name: str
    input_size: int
    output_size: int
    feedback_count: int


# Global state
app = FastAPI(
    title="SaaS-Swarm API",
    description="Swarm-as-a-Service platform API",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global storage
swarms: Dict[str, Any] = {}
agents: Dict[str, Agent] = {}
feedback_loop = FeedbackLoop(FeedbackConfig())
tool_registry = create_default_tool_registry()


@app.on_event("startup")
async def startup_event():
    """Initialize the API on startup."""
    await feedback_loop.start()
    print("SaaS-Swarm API started")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    # Stop all swarms
    for swarm in swarms.values():
        await swarm.stop()
    
    await feedback_loop.stop()
    print("SaaS-Swarm API stopped")


# Swarm management endpoints
@app.post("/swarms", response_model=SwarmResponse)
async def create_swarm(request: SwarmCreateRequest):
    """Create a new swarm."""
    swarm_id = str(uuid.uuid4())
    
    config = SwarmConfig(
        swarm_id=swarm_id,
        name=request.name,
        topology_type=request.topology_type,
        max_execution_time=request.max_execution_time,
        enable_feedback=request.enable_feedback
    )
    
    # Create swarm based on topology type
    if request.topology_type == "mesh":
        swarm = MeshSwarm(config)
    elif request.topology_type == "star":
        swarm = StarSwarm(config)
    elif request.topology_type == "ring":
        swarm = RingSwarm(config)
    elif request.topology_type == "hierarchical":
        swarm = HierarchicalSwarm(config)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown topology type: {request.topology_type}")
    
    swarms[swarm_id] = swarm
    
    return SwarmResponse(
        swarm_id=swarm_id,
        name=swarm.name,
        topology_type=swarm.config.topology_type,
        agent_count=0,
        is_running=False
    )


@app.get("/swarms", response_model=List[SwarmResponse])
async def list_swarms():
    """List all swarms."""
    return [
        SwarmResponse(
            swarm_id=swarm_id,
            name=swarm.name,
            topology_type=swarm.config.topology_type,
            agent_count=len(swarm.agents),
            is_running=swarm.is_running
        )
        for swarm_id, swarm in swarms.items()
    ]


@app.get("/swarms/{swarm_id}", response_model=Dict[str, Any])
async def get_swarm(swarm_id: str):
    """Get detailed information about a swarm."""
    if swarm_id not in swarms:
        raise HTTPException(status_code=404, detail="Swarm not found")
    
    swarm = swarms[swarm_id]
    return {
        "swarm_id": swarm_id,
        "name": swarm.name,
        "topology_type": swarm.config.topology_type,
        "agent_count": len(swarm.agents),
        "is_running": swarm.is_running,
        "agent_info": swarm.get_agent_info(),
        "execution_history": swarm.get_execution_history()
    }


@app.post("/swarms/{swarm_id}/start")
async def start_swarm(swarm_id: str):
    """Start a swarm."""
    if swarm_id not in swarms:
        raise HTTPException(status_code=404, detail="Swarm not found")
    
    swarm = swarms[swarm_id]
    await swarm.start()
    
    return {"message": f"Swarm {swarm_id} started"}


@app.post("/swarms/{swarm_id}/stop")
async def stop_swarm(swarm_id: str):
    """Stop a swarm."""
    if swarm_id not in swarms:
        raise HTTPException(status_code=404, detail="Swarm not found")
    
    swarm = swarms[swarm_id]
    await swarm.stop()
    
    return {"message": f"Swarm {swarm_id} stopped"}


@app.delete("/swarms/{swarm_id}")
async def delete_swarm(swarm_id: str):
    """Delete a swarm."""
    if swarm_id not in swarms:
        raise HTTPException(status_code=404, detail="Swarm not found")
    
    swarm = swarms[swarm_id]
    await swarm.stop()
    del swarms[swarm_id]
    
    return {"message": f"Swarm {swarm_id} deleted"}


# Agent management endpoints
@app.post("/agents", response_model=AgentResponse)
async def create_agent(request: AgentCreateRequest):
    """Create a new agent."""
    agent_id = str(uuid.uuid4())
    
    config = AgentConfig(
        agent_id=agent_id,
        name=request.name,
        input_size=request.input_size,
        output_size=request.output_size,
        hidden_size=request.hidden_size,
        learning_rate=request.learning_rate,
        enable_online_learning=request.enable_online_learning,
        tool_registry=tool_registry
    )
    
    agent = Agent(config)
    agents[agent_id] = agent
    
    # Register with feedback loop
    feedback_loop.register_agent(agent)
    
    return AgentResponse(
        agent_id=agent_id,
        name=agent.name,
        input_size=agent.input_size,
        output_size=agent.output_size,
        feedback_count=len(agent.feedback_history)
    )


@app.get("/agents", response_model=List[AgentResponse])
async def list_agents():
    """List all agents."""
    return [
        AgentResponse(
            agent_id=agent_id,
            name=agent.name,
            input_size=agent.input_size,
            output_size=agent.output_size,
            feedback_count=len(agent.feedback_history)
        )
        for agent_id, agent in agents.items()
    ]


@app.get("/agents/{agent_id}", response_model=Dict[str, Any])
async def get_agent(agent_id: str):
    """Get detailed information about an agent."""
    if agent_id not in agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    agent = agents[agent_id]
    return {
        "agent_id": agent_id,
        "name": agent.name,
        "input_size": agent.input_size,
        "output_size": agent.output_size,
        "feedback_count": len(agent.feedback_history),
        "state": agent.get_state()
    }


@app.post("/swarms/{swarm_id}/agents/{agent_id}")
async def add_agent_to_swarm(swarm_id: str, agent_id: str):
    """Add an agent to a swarm."""
    if swarm_id not in swarms:
        raise HTTPException(status_code=404, detail="Swarm not found")
    
    if agent_id not in agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    swarm = swarms[swarm_id]
    agent = agents[agent_id]
    
    success = swarm.add_agent(agent)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to add agent to swarm")
    
    return {"message": f"Agent {agent_id} added to swarm {swarm_id}"}


@app.delete("/swarms/{swarm_id}/agents/{agent_id}")
async def remove_agent_from_swarm(swarm_id: str, agent_id: str):
    """Remove an agent from a swarm."""
    if swarm_id not in swarms:
        raise HTTPException(status_code=404, detail="Swarm not found")
    
    swarm = swarms[swarm_id]
    success = swarm.remove_agent(agent_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Agent not found in swarm")
    
    return {"message": f"Agent {agent_id} removed from swarm {swarm_id}"}


# Swarm execution endpoints
@app.post("/swarms/{swarm_id}/run", response_model=Dict[str, Any])
async def run_swarm(swarm_id: str, request: SwarmRunRequest):
    """Run a swarm with given input."""
    if swarm_id not in swarms:
        raise HTTPException(status_code=404, detail="Swarm not found")
    
    swarm = swarms[swarm_id]
    
    if not swarm.is_running:
        raise HTTPException(status_code=400, detail="Swarm is not running")
    
    try:
        result = await swarm.execute(request.input_data, request.timeout)
        
        # Evaluate result and propagate feedback
        if swarm.config.enable_feedback:
            evaluation = await feedback_loop.evaluate_swarm_output(result['results'])
            await feedback_loop.propagate_feedback(evaluation)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Swarm execution failed: {str(e)}")


# Feedback endpoints
@app.post("/agents/{agent_id}/feedback")
async def submit_feedback(agent_id: str, request: FeedbackRequest):
    """Submit feedback to an agent."""
    if agent_id not in agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    agent = agents[agent_id]
    
    feedback = {
        'reward': request.reward,
        'target': request.target,
        'metadata': request.metadata or {}
    }
    
    await agent.receive_feedback(feedback)
    
    return {"message": f"Feedback submitted to agent {agent_id}"}


@app.get("/feedback/history")
async def get_feedback_history(limit: int = 50):
    """Get feedback history."""
    return feedback_loop.get_feedback_history(limit)


@app.get("/feedback/agents/{agent_id}/performance")
async def get_agent_performance(agent_id: str):
    """Get performance statistics for an agent."""
    return feedback_loop.get_agent_performance(agent_id)


# Tool management endpoints
@app.get("/tools")
async def list_tools():
    """List all available tools."""
    return {
        "tools": tool_registry.list_tools(),
        "tool_info": {
            name: tool_registry.get_tool_info(name)
            for name in tool_registry.list_tools()
        }
    }


@app.get("/tools/{tool_name}/usage")
async def get_tool_usage(tool_name: str):
    """Get usage statistics for a tool."""
    return tool_registry.get_tool_usage_stats(tool_name)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "swarm_count": len(swarms),
        "agent_count": len(agents),
        "feedback_loop_running": feedback_loop.is_running,
        "tool_count": len(tool_registry.tools)
    }


# WebSocket endpoint for real-time monitoring (placeholder)
@app.websocket("/ws/monitor")
async def websocket_monitor(websocket):
    """WebSocket endpoint for real-time monitoring."""
    await websocket.accept()
    
    try:
        while True:
            # Send periodic status updates
            status = {
                "swarm_count": len(swarms),
                "agent_count": len(agents),
                "timestamp": asyncio.get_event_loop().time()
            }
            
            await websocket.send_text(str(status))
            await asyncio.sleep(5)  # Update every 5 seconds
            
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
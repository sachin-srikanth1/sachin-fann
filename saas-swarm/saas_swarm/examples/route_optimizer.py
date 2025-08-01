"""
Route Optimizer Swarm Example.

Demonstrates a swarm with multiple agents collaborating
to optimize delivery routes.
"""

import asyncio
from typing import Dict, Any, List, Tuple
import uuid
import random

from saas_swarm.core.agent import Agent, AgentConfig
from saas_swarm.core.swarm import SwarmConfig, MeshSwarm
from saas_swarm.core.feedback_loop import FeedbackLoop, FeedbackConfig
from saas_swarm.tools.registry import create_default_tool_registry


async def distance_agent_inference(input_data: Any, tool_registry) -> Dict[str, Any]:
    """Custom inference function for distance calculation agent."""
    if isinstance(input_data, dict) and 'locations' in input_data:
        locations = input_data['locations']
        
        # Calculate distances between all locations
        distances = {}
        for i, loc1 in enumerate(locations):
            for j, loc2 in enumerate(locations):
                if i != j:
                    # Simple distance calculation (can be enhanced with real APIs)
                    distance = random.uniform(5, 50)  # Mock distance
                    distances[f"{loc1}->{loc2}"] = distance
        
        return {
            'distances': distances,
            'total_locations': len(locations),
            'average_distance': sum(distances.values()) / len(distances) if distances else 0
        }
    else:
        return {'error': 'Invalid input format'}


async def traffic_agent_inference(input_data: Any, tool_registry) -> Dict[str, Any]:
    """Custom inference function for traffic analysis agent."""
    if isinstance(input_data, dict) and 'distances' in input_data:
        distances = input_data['distances']
        
        # Analyze traffic patterns
        traffic_factors = {}
        for route, distance in distances.items():
            # Mock traffic factor based on distance
            traffic_factor = 1.0 + (distance / 100) * random.uniform(0.1, 0.5)
            traffic_factors[route] = traffic_factor
        
        return {
            'traffic_factors': traffic_factors,
            'high_traffic_routes': [route for route, factor in traffic_factors.items() if factor > 1.3],
            'average_traffic_factor': sum(traffic_factors.values()) / len(traffic_factors) if traffic_factors else 1.0
        }
    else:
        return {'error': 'Invalid input format'}


async def optimization_agent_inference(input_data: Any, tool_registry) -> Dict[str, Any]:
    """Custom inference function for route optimization agent."""
    if isinstance(input_data, dict) and 'traffic_factors' in input_data:
        traffic_data = input_data
        traffic_factors = traffic_data['traffic_factors']
        
        # Simple optimization algorithm
        routes = list(traffic_factors.keys())
        if len(routes) < 2:
            return {'optimized_route': routes[0] if routes else 'No routes'}
        
        # Find route with lowest traffic factor
        best_route = min(traffic_factors.items(), key=lambda x: x[1])
        
        # Generate alternative routes
        alternative_routes = sorted(traffic_factors.items(), key=lambda x: x[1])[:3]
        
        return {
            'optimized_route': best_route[0],
            'traffic_factor': best_route[1],
            'alternative_routes': alternative_routes,
            'optimization_score': 1.0 / best_route[1] if best_route[1] > 0 else 0
        }
    else:
        return {'error': 'Invalid input format'}


async def delivery_agent_inference(input_data: Any, tool_registry) -> str:
    """Custom inference function for delivery planning agent."""
    if isinstance(input_data, dict) and 'optimized_route' in input_data:
        optimization = input_data
        route = optimization['optimized_route']
        score = optimization.get('optimization_score', 0)
        
        # Generate delivery plan
        delivery_plan = f"""
DELIVERY ROUTE OPTIMIZATION REPORT

Optimized Route: {route}
Optimization Score: {score:.2f}

Delivery Plan:
1. Start at origin
2. Follow optimized route: {route}
3. Estimated delivery time: {random.randint(30, 120)} minutes
4. Fuel efficiency: {random.uniform(0.8, 1.2):.2f} L/100km

Recommendations:
- Monitor traffic conditions
- Update route if needed
- Track delivery progress
"""
        
        return delivery_plan
    else:
        return "Unable to generate delivery plan"


def create_route_optimizer_swarm() -> MeshSwarm:
    """
    Create a route optimizer swarm with multiple specialized agents.
    
    Returns:
        Configured MeshSwarm instance
    """
    # Create tool registry
    tool_registry = create_default_tool_registry()
    
    # Create feedback loop
    feedback_loop = FeedbackLoop(FeedbackConfig())
    
    # Create distance calculation agent
    distance_agent_id = str(uuid.uuid4())
    distance_config = AgentConfig(
        agent_id=distance_agent_id,
        name="DistanceAgent",
        input_size=100,
        output_size=50,
        hidden_size=64,
        learning_rate=0.01,
        enable_online_learning=True,
        tool_registry=tool_registry,
        custom_inference_fn=distance_agent_inference
    )
    distance_agent = Agent(distance_config)
    
    # Create traffic analysis agent
    traffic_agent_id = str(uuid.uuid4())
    traffic_config = AgentConfig(
        agent_id=traffic_agent_id,
        name="TrafficAgent",
        input_size=100,
        output_size=50,
        hidden_size=64,
        learning_rate=0.01,
        enable_online_learning=True,
        tool_registry=tool_registry,
        custom_inference_fn=traffic_agent_inference
    )
    traffic_agent = Agent(traffic_config)
    
    # Create optimization agent
    optimization_agent_id = str(uuid.uuid4())
    optimization_config = AgentConfig(
        agent_id=optimization_agent_id,
        name="OptimizationAgent",
        input_size=100,
        output_size=50,
        hidden_size=64,
        learning_rate=0.01,
        enable_online_learning=True,
        tool_registry=tool_registry,
        custom_inference_fn=optimization_agent_inference
    )
    optimization_agent = Agent(optimization_config)
    
    # Create delivery planning agent
    delivery_agent_id = str(uuid.uuid4())
    delivery_config = AgentConfig(
        agent_id=delivery_agent_id,
        name="DeliveryAgent",
        input_size=100,
        output_size=50,
        hidden_size=64,
        learning_rate=0.01,
        enable_online_learning=True,
        tool_registry=tool_registry,
        custom_inference_fn=delivery_agent_inference
    )
    delivery_agent = Agent(delivery_config)
    
    # Register agents with feedback loop
    feedback_loop.register_agent(distance_agent)
    feedback_loop.register_agent(traffic_agent)
    feedback_loop.register_agent(optimization_agent)
    feedback_loop.register_agent(delivery_agent)
    
    # Create mesh swarm
    swarm_config = SwarmConfig(
        swarm_id=str(uuid.uuid4()),
        name="RouteOptimizerSwarm",
        topology_type="mesh",
        max_execution_time=60.0,
        enable_feedback=True
    )
    
    swarm = MeshSwarm(swarm_config)
    
    # Add agents to swarm
    swarm.add_agent(distance_agent)
    swarm.add_agent(traffic_agent)
    swarm.add_agent(optimization_agent)
    swarm.add_agent(delivery_agent)
    
    return swarm


async def run_route_optimizer_example():
    """Run the route optimizer example."""
    print("Creating Route Optimizer Swarm...")
    swarm = create_route_optimizer_swarm()
    
    # Start the swarm
    await swarm.start()
    print("Swarm started!")
    
    # Test with different delivery scenarios
    scenarios = [
        {
            'locations': ['Warehouse A', 'Customer 1', 'Customer 2', 'Customer 3'],
            'description': 'Multi-customer delivery'
        },
        {
            'locations': ['Distribution Center', 'Store A', 'Store B', 'Store C', 'Store D'],
            'description': 'Store delivery route'
        },
        {
            'locations': ['Hub', 'Location 1', 'Location 2'],
            'description': 'Simple delivery route'
        }
    ]
    
    for scenario in scenarios:
        print(f"\n--- Optimizing route for: {scenario['description']} ---")
        print(f"Locations: {scenario['locations']}")
        
        try:
            result = await swarm.execute(scenario, timeout=30.0)
            
            if 'results' in result:
                print("Optimization Result:")
                print(result['results'].get('final_result', 'No result generated'))
            else:
                print("Execution failed:", result)
                
        except Exception as e:
            print(f"Error: {e}")
    
    # Stop the swarm
    await swarm.stop()
    print("\nSwarm stopped!")


if __name__ == "__main__":
    asyncio.run(run_route_optimizer_example()) 
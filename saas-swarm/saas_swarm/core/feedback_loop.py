"""
FeedbackLoop module for SaaS-Swarm platform.

Provides evaluation and feedback mechanisms for swarm adaptation.
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np

from .agent import Agent


@dataclass
class FeedbackConfig:
    """Configuration for feedback loop."""
    evaluation_function: Optional[Callable] = None
    reward_scale: float = 1.0
    enable_adaptive_learning: bool = True
    feedback_delay: float = 0.1  # seconds


class FeedbackLoop:
    """
    System to evaluate swarm output and propagate reward signals.
    
    Supports:
    - Custom evaluation functions
    - Reward signal propagation
    - Adaptive learning rates
    - Feedback history tracking
    """
    
    def __init__(self, config: FeedbackConfig = None):
        self.config = config or FeedbackConfig()
        self.agents: Dict[str, Agent] = {}
        self.feedback_history: List[Dict] = []
        self.evaluation_cache: Dict[str, float] = {}
        self.is_running = False
        
    def register_agent(self, agent: Agent):
        """Register an agent for feedback."""
        self.agents[agent.agent_id] = agent
        print(f"Registered agent for feedback: {agent.agent_id}")
        
    def unregister_agent(self, agent_id: str):
        """Unregister an agent from feedback."""
        if agent_id in self.agents:
            del self.agents[agent_id]
            print(f"Unregistered agent from feedback: {agent_id}")
            
    async def start(self):
        """Start the feedback loop."""
        self.is_running = True
        print("FeedbackLoop started")
        
    async def stop(self):
        """Stop the feedback loop."""
        self.is_running = False
        print("FeedbackLoop stopped")
        
    async def evaluate_swarm_output(self, swarm_output: Dict[str, Any], 
                                  expected_output: Optional[Any] = None) -> Dict[str, Any]:
        """
        Evaluate swarm output and generate feedback signals.
        
        Args:
            swarm_output: Output from swarm execution
            expected_output: Expected output for comparison
            
        Returns:
            Evaluation results with feedback signals
        """
        if not self.is_running:
            return {'error': 'FeedbackLoop not running'}
            
        evaluation_id = str(asyncio.get_event_loop().time())
        
        # Use custom evaluation function if provided
        if self.config.evaluation_function:
            evaluation_result = await self.config.evaluation_function(swarm_output, expected_output)
        else:
            evaluation_result = self._default_evaluation(swarm_output, expected_output)
            
        # Store evaluation
        evaluation_record = {
            'evaluation_id': evaluation_id,
            'swarm_output': swarm_output,
            'expected_output': expected_output,
            'evaluation_result': evaluation_result,
            'timestamp': asyncio.get_event_loop().time()
        }
        
        self.feedback_history.append(evaluation_record)
        
        return evaluation_result
        
    def _default_evaluation(self, swarm_output: Dict[str, Any], 
                          expected_output: Optional[Any]) -> Dict[str, Any]:
        """Default evaluation function."""
        # Simple evaluation based on output structure and content
        score = 0.0
        feedback = {}
        
        # Check if output has expected structure
        if 'results' in swarm_output:
            score += 0.3
            
        # Check if output has multiple agent results
        if 'total_agents' in swarm_output:
            agent_count = swarm_output['total_agents']
            if agent_count > 0:
                score += min(0.3, agent_count * 0.1)
                
        # Check for errors
        if 'error' in swarm_output:
            score -= 0.5
            feedback['error'] = swarm_output['error']
            
        # Compare with expected output if provided
        if expected_output is not None:
            # Simple similarity check
            if isinstance(swarm_output, type(expected_output)):
                score += 0.2
                
        feedback['score'] = score
        feedback['recommendation'] = self._get_recommendation(score)
        
        return feedback
        
    def _get_recommendation(self, score: float) -> str:
        """Get recommendation based on evaluation score."""
        if score >= 0.8:
            return "Excellent performance"
        elif score >= 0.6:
            return "Good performance"
        elif score >= 0.4:
            return "Acceptable performance"
        elif score >= 0.2:
            return "Needs improvement"
        else:
            return "Poor performance - requires attention"
            
    async def propagate_feedback(self, evaluation_result: Dict[str, Any], 
                               agent_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Propagate feedback signals to agents.
        
        Args:
            evaluation_result: Result from evaluation
            agent_ids: Specific agents to send feedback to (None for all)
            
        Returns:
            Propagation results
        """
        if not self.is_running:
            return {'error': 'FeedbackLoop not running'}
            
        target_agents = agent_ids or list(self.agents.keys())
        propagation_results = {}
        
        for agent_id in target_agents:
            if agent_id not in self.agents:
                continue
                
            agent = self.agents[agent_id]
            
            # Create feedback signal
            feedback_signal = self._create_feedback_signal(evaluation_result, agent_id)
            
            # Send feedback to agent
            try:
                await agent.receive_feedback(feedback_signal)
                propagation_results[agent_id] = 'success'
            except Exception as e:
                propagation_results[agent_id] = f'error: {str(e)}'
                
        return propagation_results
        
    def _create_feedback_signal(self, evaluation_result: Dict[str, Any], 
                               agent_id: str) -> Dict[str, Any]:
        """Create a feedback signal for a specific agent."""
        score = evaluation_result.get('score', 0.0)
        
        # Scale reward based on configuration
        reward = score * self.config.reward_scale
        
        # Create target for supervised learning if score is low
        target = None
        if score < 0.5 and self.config.enable_adaptive_learning:
            # Generate a simple target for improvement
            target = [1.0] * 10  # Example target
            
        return {
            'reward': reward,
            'target': target,
            'score': score,
            'recommendation': evaluation_result.get('recommendation', ''),
            'metadata': {
                'agent_id': agent_id,
                'timestamp': asyncio.get_event_loop().time(),
                'evaluation_source': 'FeedbackLoop'
            }
        }
        
    async def adaptive_learning_update(self, agent_id: str, 
                                     performance_history: List[float]) -> Dict[str, Any]:
        """
        Update agent learning parameters based on performance history.
        
        Args:
            agent_id: ID of the agent to update
            performance_history: List of recent performance scores
            
        Returns:
            Update results
        """
        if agent_id not in self.agents:
            return {'error': f'Agent {agent_id} not found'}
            
        agent = self.agents[agent_id]
        
        if not self.config.enable_adaptive_learning:
            return {'status': 'adaptive_learning_disabled'}
            
        # Calculate performance trend
        if len(performance_history) < 2:
            return {'status': 'insufficient_history'}
            
        recent_performance = np.mean(performance_history[-5:])  # Last 5 scores
        performance_trend = np.polyfit(range(len(performance_history)), performance_history, 1)[0]
        
        # Adjust learning rate based on performance
        current_lr = agent.neural_network.learning_rate
        
        if performance_trend < 0 and recent_performance < 0.5:
            # Decreasing performance, increase learning rate
            new_lr = min(current_lr * 1.2, 0.1)
            agent.neural_network.learning_rate = new_lr
            adjustment = 'increased'
        elif performance_trend > 0 and recent_performance > 0.8:
            # Good performance, decrease learning rate
            new_lr = max(current_lr * 0.9, 0.001)
            agent.neural_network.learning_rate = new_lr
            adjustment = 'decreased'
        else:
            new_lr = current_lr
            adjustment = 'maintained'
            
        return {
            'agent_id': agent_id,
            'old_learning_rate': current_lr,
            'new_learning_rate': new_lr,
            'adjustment': adjustment,
            'recent_performance': recent_performance,
            'performance_trend': performance_trend
        }
        
    def get_feedback_history(self, limit: int = 50) -> List[Dict]:
        """Get recent feedback history."""
        return self.feedback_history[-limit:]
        
    def get_agent_performance(self, agent_id: str) -> Dict[str, Any]:
        """Get performance statistics for a specific agent."""
        if agent_id not in self.agents:
            return {'error': f'Agent {agent_id} not found'}
            
        # Extract scores for this agent from feedback history
        agent_scores = []
        for record in self.feedback_history:
            if 'evaluation_result' in record:
                score = record['evaluation_result'].get('score', 0.0)
                agent_scores.append(score)
                
        if not agent_scores:
            return {'agent_id': agent_id, 'no_data': True}
            
        return {
            'agent_id': agent_id,
            'average_score': np.mean(agent_scores),
            'score_std': np.std(agent_scores),
            'min_score': np.min(agent_scores),
            'max_score': np.max(agent_scores),
            'total_evaluations': len(agent_scores)
        }
        
    async def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the feedback loop."""
        return {
            'is_running': self.is_running,
            'registered_agents': len(self.agents),
            'feedback_history_size': len(self.feedback_history),
            'evaluation_cache_size': len(self.evaluation_cache)
        }


class RewardBasedFeedbackLoop(FeedbackLoop):
    """
    Specialized feedback loop for reinforcement learning scenarios.
    """
    
    def __init__(self, config: FeedbackConfig = None):
        super().__init__(config)
        self.reward_history: Dict[str, List[float]] = {}
        
    async def receive_reward(self, agent_id: str, reward: float, 
                           context: Optional[Dict] = None):
        """Receive a reward signal for a specific agent."""
        if agent_id not in self.agents:
            return
            
        if agent_id not in self.reward_history:
            self.reward_history[agent_id] = []
            
        self.reward_history[agent_id].append(reward)
        
        # Create feedback signal
        feedback_signal = {
            'reward': reward,
            'target': None,  # No target for pure RL
            'metadata': {
                'agent_id': agent_id,
                'context': context,
                'timestamp': asyncio.get_event_loop().time(),
                'feedback_type': 'reward'
            }
        }
        
        # Send to agent
        agent = self.agents[agent_id]
        await agent.receive_feedback(feedback_signal)
        
    def get_reward_statistics(self, agent_id: str) -> Dict[str, Any]:
        """Get reward statistics for an agent."""
        if agent_id not in self.reward_history:
            return {'agent_id': agent_id, 'no_rewards': True}
            
        rewards = self.reward_history[agent_id]
        
        return {
            'agent_id': agent_id,
            'total_rewards': len(rewards),
            'average_reward': np.mean(rewards),
            'reward_std': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'recent_rewards': rewards[-10:]  # Last 10 rewards
        } 
"""
Code Review Swarm Example.

Demonstrates a swarm with multiple agents collaborating
to perform comprehensive code reviews.
"""

import asyncio
from typing import Dict, Any, List
import uuid
import re

from saas_swarm.core.agent import Agent, AgentConfig
from saas_swarm.core.swarm import SwarmConfig, HierarchicalSwarm
from saas_swarm.core.feedback_loop import FeedbackLoop, FeedbackConfig
from saas_swarm.tools.registry import create_default_tool_registry


async def syntax_agent_inference(input_data: Any, tool_registry) -> Dict[str, Any]:
    """Custom inference function for syntax analysis agent."""
    code = str(input_data)
    
    # Basic syntax analysis
    syntax_issues = []
    
    # Check for common Python syntax issues
    if 'print ' in code:  # Python 2 style print
        syntax_issues.append("Python 2 style print statement detected")
    
    if 'def ' in code and ':' not in code:
        syntax_issues.append("Function definition missing colon")
    
    if 'if ' in code and ':' not in code:
        syntax_issues.append("If statement missing colon")
    
    # Check for indentation issues
    lines = code.split('\n')
    for i, line in enumerate(lines):
        if line.strip() and not line.startswith(' ') and ':' in line:
            next_line = lines[i + 1] if i + 1 < len(lines) else ''
            if next_line and not next_line.startswith(' ') and next_line.strip():
                syntax_issues.append(f"Indentation issue at line {i + 1}")
    
    return {
        'syntax_issues': syntax_issues,
        'total_lines': len(lines),
        'syntax_score': max(0, 10 - len(syntax_issues)),
        'language': 'python' if 'def ' in code or 'import ' in code else 'unknown'
    }


async def security_agent_inference(input_data: Any, tool_registry) -> Dict[str, Any]:
    """Custom inference function for security analysis agent."""
    code = str(input_data)
    
    # Security analysis
    security_issues = []
    security_score = 10
    
    # Check for common security issues
    dangerous_patterns = [
        ('eval(', 'Use of eval() function'),
        ('exec(', 'Use of exec() function'),
        ('os.system(', 'Use of os.system()'),
        ('subprocess.call(', 'Use of subprocess.call()'),
        ('input(', 'Use of input() without validation'),
        ('pickle.loads(', 'Use of pickle.loads()'),
        ('yaml.load(', 'Use of yaml.load()'),
    ]
    
    for pattern, issue in dangerous_patterns:
        if pattern in code:
            security_issues.append(issue)
            security_score -= 2
    
    # Check for hardcoded credentials
    if re.search(r'password\s*=\s*["\'][^"\']+["\']', code, re.IGNORECASE):
        security_issues.append("Hardcoded password detected")
        security_score -= 3
    
    if re.search(r'api_key\s*=\s*["\'][^"\']+["\']', code, re.IGNORECASE):
        security_issues.append("Hardcoded API key detected")
        security_score -= 3
    
    return {
        'security_issues': security_issues,
        'security_score': max(0, security_score),
        'risk_level': 'high' if security_score < 5 else 'medium' if security_score < 8 else 'low'
    }


async def performance_agent_inference(input_data: Any, tool_registry) -> Dict[str, Any]:
    """Custom inference function for performance analysis agent."""
    code = str(input_data)
    
    # Performance analysis
    performance_issues = []
    performance_score = 10
    
    # Check for performance issues
    if 'for ' in code and ' in ' in code and 'range(' in code:
        # Check for nested loops
        lines = code.split('\n')
        for_lines = [i for i, line in enumerate(lines) if 'for ' in line and ' in ' in line]
        if len(for_lines) > 1:
            performance_issues.append("Nested loops detected - consider optimization")
            performance_score -= 2
    
    if 'list(' in code and 'map(' in code:
        performance_issues.append("Consider using list comprehension instead of map()")
        performance_score -= 1
    
    if 'import *' in code:
        performance_issues.append("Wildcard import detected")
        performance_score -= 1
    
    # Check for memory issues
    if 'global ' in code:
        performance_issues.append("Global variables used - consider local scope")
        performance_score -= 1
    
    return {
        'performance_issues': performance_issues,
        'performance_score': max(0, performance_score),
        'complexity': 'high' if len(performance_issues) > 2 else 'medium' if len(performance_issues) > 0 else 'low'
    }


async def style_agent_inference(input_data: Any, tool_registry) -> Dict[str, Any]:
    """Custom inference function for code style analysis agent."""
    code = str(input_data)
    
    # Style analysis
    style_issues = []
    style_score = 10
    
    lines = code.split('\n')
    
    # Check line length
    long_lines = [i + 1 for i, line in enumerate(lines) if len(line) > 79]
    if long_lines:
        style_issues.append(f"Lines too long: {long_lines}")
        style_score -= 1
    
    # Check for proper spacing
    if 'def ' in code and 'def' in code:
        functions = re.findall(r'def\s+(\w+)', code)
        for func in functions:
            if not re.search(rf'def\s+{func}\s*\(', code):
                style_issues.append(f"Function '{func}' missing proper spacing")
                style_score -= 1
    
    # Check for consistent indentation
    indent_sizes = set()
    for line in lines:
        if line.strip():
            indent = len(line) - len(line.lstrip())
            if indent > 0:
                indent_sizes.add(indent)
    
    if len(indent_sizes) > 1:
        style_issues.append("Inconsistent indentation detected")
        style_score -= 2
    
    return {
        'style_issues': style_issues,
        'style_score': max(0, style_score),
        'readability': 'good' if style_score >= 8 else 'fair' if style_score >= 5 else 'poor'
    }


async def review_agent_inference(input_data: Any, tool_registry) -> str:
    """Custom inference function for final review agent."""
    if isinstance(input_data, dict):
        # Combine all analysis results
        syntax = input_data.get('syntax_analysis', {})
        security = input_data.get('security_analysis', {})
        performance = input_data.get('performance_analysis', {})
        style = input_data.get('style_analysis', {})
        
        # Generate comprehensive review
        review = f"""
CODE REVIEW REPORT
==================

OVERALL SCORES:
- Syntax: {syntax.get('syntax_score', 0)}/10
- Security: {security.get('security_score', 0)}/10
- Performance: {performance.get('performance_score', 0)}/10
- Style: {style.get('style_score', 0)}/10

DETAILED ANALYSIS:
"""

        # Syntax issues
        if syntax.get('syntax_issues'):
            review += f"\nSYNTAX ISSUES:\n"
            for issue in syntax['syntax_issues']:
                review += f"- {issue}\n"
        
        # Security issues
        if security.get('security_issues'):
            review += f"\nSECURITY ISSUES (Risk Level: {security.get('risk_level', 'unknown')}):\n"
            for issue in security['security_issues']:
                review += f"- {issue}\n"
        
        # Performance issues
        if performance.get('performance_issues'):
            review += f"\nPERFORMANCE ISSUES (Complexity: {performance.get('complexity', 'unknown')}):\n"
            for issue in performance['performance_issues']:
                review += f"- {issue}\n"
        
        # Style issues
        if style.get('style_issues'):
            review += f"\nSTYLE ISSUES (Readability: {style.get('readability', 'unknown')}):\n"
            for issue in style['style_issues']:
                review += f"- {issue}\n"
        
        # Recommendations
        review += f"\nRECOMMENDATIONS:\n"
        if syntax.get('syntax_score', 10) < 8:
            review += "- Fix syntax issues before deployment\n"
        if security.get('security_score', 10) < 7:
            review += "- Address security vulnerabilities immediately\n"
        if performance.get('performance_score', 10) < 7:
            review += "- Consider performance optimizations\n"
        if style.get('style_score', 10) < 7:
            review += "- Improve code style and readability\n"
        
        review += "\nOverall assessment: "
        total_score = (syntax.get('syntax_score', 0) + security.get('security_score', 0) + 
                      performance.get('performance_score', 0) + style.get('style_score', 0)) / 4
        
        if total_score >= 8:
            review += "EXCELLENT - Code is ready for production"
        elif total_score >= 6:
            review += "GOOD - Minor issues to address"
        elif total_score >= 4:
            review += "FAIR - Several issues need attention"
        else:
            review += "POOR - Major issues must be resolved"
        
        return review
    else:
        return "Unable to generate review - invalid input format"


def create_code_review_swarm() -> HierarchicalSwarm:
    """
    Create a code review swarm with multiple specialized agents.
    
    Returns:
        Configured HierarchicalSwarm instance
    """
    # Create tool registry
    tool_registry = create_default_tool_registry()
    
    # Create feedback loop
    feedback_loop = FeedbackLoop(FeedbackConfig())
    
    # Create syntax analysis agent
    syntax_agent_id = str(uuid.uuid4())
    syntax_config = AgentConfig(
        agent_id=syntax_agent_id,
        name="SyntaxAgent",
        input_size=200,
        output_size=50,
        hidden_size=64,
        learning_rate=0.01,
        enable_online_learning=True,
        tool_registry=tool_registry,
        custom_inference_fn=syntax_agent_inference
    )
    syntax_agent = Agent(syntax_config)
    
    # Create security analysis agent
    security_agent_id = str(uuid.uuid4())
    security_config = AgentConfig(
        agent_id=security_agent_id,
        name="SecurityAgent",
        input_size=200,
        output_size=50,
        hidden_size=64,
        learning_rate=0.01,
        enable_online_learning=True,
        tool_registry=tool_registry,
        custom_inference_fn=security_agent_inference
    )
    security_agent = Agent(security_config)
    
    # Create performance analysis agent
    performance_agent_id = str(uuid.uuid4())
    performance_config = AgentConfig(
        agent_id=performance_agent_id,
        name="PerformanceAgent",
        input_size=200,
        output_size=50,
        hidden_size=64,
        learning_rate=0.01,
        enable_online_learning=True,
        tool_registry=tool_registry,
        custom_inference_fn=performance_agent_inference
    )
    performance_agent = Agent(performance_config)
    
    # Create style analysis agent
    style_agent_id = str(uuid.uuid4())
    style_config = AgentConfig(
        agent_id=style_agent_id,
        name="StyleAgent",
        input_size=200,
        output_size=50,
        hidden_size=64,
        learning_rate=0.01,
        enable_online_learning=True,
        tool_registry=tool_registry,
        custom_inference_fn=style_agent_inference
    )
    style_agent = Agent(style_config)
    
    # Create review coordinator agent
    review_agent_id = str(uuid.uuid4())
    review_config = AgentConfig(
        agent_id=review_agent_id,
        name="ReviewAgent",
        input_size=200,
        output_size=100,
        hidden_size=64,
        learning_rate=0.01,
        enable_online_learning=True,
        tool_registry=tool_registry,
        custom_inference_fn=review_agent_inference
    )
    review_agent = Agent(review_config)
    
    # Register agents with feedback loop
    feedback_loop.register_agent(syntax_agent)
    feedback_loop.register_agent(security_agent)
    feedback_loop.register_agent(performance_agent)
    feedback_loop.register_agent(style_agent)
    feedback_loop.register_agent(review_agent)
    
    # Create hierarchical swarm
    swarm_config = SwarmConfig(
        swarm_id=str(uuid.uuid4()),
        name="CodeReviewSwarm",
        topology_type="hierarchical",
        max_execution_time=60.0,
        enable_feedback=True
    )
    
    swarm = HierarchicalSwarm(swarm_config)
    
    # Add agents to swarm in hierarchical structure
    # Review agent is the root
    swarm.add_agent(review_agent)
    
    # Analysis agents are children of review agent
    swarm.add_agent(syntax_agent, parent_id=review_agent_id)
    swarm.add_agent(security_agent, parent_id=review_agent_id)
    swarm.add_agent(performance_agent, parent_id=review_agent_id)
    swarm.add_agent(style_agent, parent_id=review_agent_id)
    
    return swarm


async def run_code_review_example():
    """Run the code review example."""
    print("Creating Code Review Swarm...")
    swarm = create_code_review_swarm()
    
    # Start the swarm
    await swarm.start()
    print("Swarm started!")
    
    # Test with different code samples
    code_samples = [
        {
            'code': '''
def calculate_sum(a, b):
    return a + b

def main():
    result = calculate_sum(5, 3)
    print result
''',
            'description': 'Python 2 style code'
        },
        {
            'code': '''
import os
import pickle

def load_data():
    password = "secret123"
    data = eval(input("Enter data: "))
    return pickle.loads(data)

def process_data(data):
    os.system("rm -rf /")
    return data
''',
            'description': 'Code with security issues'
        },
        {
            'code': '''
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def process_list(items):
    result = []
    for i in range(len(items)):
        for j in range(len(items)):
            result.append(items[i] + items[j])
    return result
''',
            'description': 'Code with performance issues'
        }
    ]
    
    for sample in code_samples:
        print(f"\n--- Reviewing code: {sample['description']} ---")
        print(f"Code:\n{sample['code']}")
        
        try:
            result = await swarm.execute(sample['code'], timeout=30.0)
            
            if 'results' in result:
                print("Review Result:")
                print(result['results'].get('final_result', 'No result generated'))
            else:
                print("Execution failed:", result)
                
        except Exception as e:
            print(f"Error: {e}")
    
    # Stop the swarm
    await swarm.stop()
    print("\nSwarm stopped!")


if __name__ == "__main__":
    asyncio.run(run_code_review_example()) 
#!/usr/bin/env python3
"""
Test script for enhanced SaaS-Swarm with OpenAI and email functionality.

This script demonstrates:
1. Configuration validation
2. Enhanced research agent using OpenAI
3. Enhanced writer agent using OpenAI
4. Execute agent sending emails
5. Feedback loops
"""

import asyncio
import sys
import os

# Add the package to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from saas_swarm.config import Config
from saas_swarm.examples.enhanced_email_writer import create_enhanced_email_writer_swarm, run_enhanced_email_writer_example


async def test_configuration():
    """Test if configuration is properly set up."""
    print("🔧 Testing Configuration...")
    
    if Config.validate():
        print("✅ Configuration is valid!")
        print(f"OpenAI API Key: {'*' * 10 + Config.OPENAI_API_KEY[-4:] if Config.OPENAI_API_KEY else 'Not set'}")
        print(f"Email Server: {Config.EMAIL_SMTP_SERVER}")
        print(f"Email Username: {Config.EMAIL_USERNAME}")
        print(f"Email From: {Config.EMAIL_FROM_ADDRESS}")
        return True
    else:
        print("❌ Configuration is invalid!")
        print("Please check your .env file and ensure all required variables are set.")
        return False


async def test_tool_registry():
    """Test if advanced tools are available."""
    print("\n🔧 Testing Tool Registry...")
    
    from saas_swarm.tools.registry import create_default_tool_registry
    
    registry = create_default_tool_registry()
    available_tools = registry.list_tools()
    
    print(f"Available tools: {available_tools}")
    
    # Check for advanced tools
    advanced_tools = ["openai_research", "send_email", "openai_generate"]
    missing_tools = [tool for tool in advanced_tools if tool not in available_tools]
    
    if missing_tools:
        print(f"❌ Missing advanced tools: {missing_tools}")
        return False
    else:
        print("✅ All advanced tools are available!")
        return True


async def test_enhanced_swarm():
    """Test the enhanced email writer swarm."""
    print("\n🚀 Testing Enhanced Email Writer Swarm...")
    
    try:
        # Create the swarm
        swarm = create_enhanced_email_writer_swarm()
        print("✅ Swarm created successfully!")
        
        # Start the swarm
        await swarm.start()
        print("✅ Swarm started successfully!")
        
        # Test with a simple topic
        test_topic = "artificial intelligence trends 2024"
        print(f"\n📧 Testing with topic: {test_topic}")
        
        result = await swarm.execute(test_topic, timeout=60.0)
        
        if 'results' in result:
            print("✅ Swarm execution completed!")
            print("Result:", result['results'].get('final_result', 'No result generated'))
        else:
            print("❌ Swarm execution failed:", result)
        
        # Stop the swarm
        await swarm.stop()
        print("✅ Swarm stopped successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing enhanced swarm: {e}")
        return False


async def test_feedback_loop():
    """Test feedback loop functionality."""
    print("\n🔄 Testing Feedback Loop...")
    
    from saas_swarm.core.feedback_loop import FeedbackLoop, FeedbackConfig
    from saas_swarm.core.agent import Agent, AgentConfig
    from saas_swarm.tools.registry import create_default_tool_registry
    
    # Create feedback loop
    feedback_loop = FeedbackLoop(FeedbackConfig())
    await feedback_loop.start()
    
    # Create a test agent
    tool_registry = create_default_tool_registry()
    agent_config = AgentConfig(
        agent_id="test-agent",
        name="TestAgent",
        input_size=10,
        output_size=5,
        hidden_size=8,
        enable_online_learning=True,
        tool_registry=tool_registry
    )
    
    agent = Agent(agent_config)
    feedback_loop.register_agent(agent)
    
    # Test feedback
    feedback = {
        'reward': 0.8,
        'target': [0.1, 0.2, 0.3, 0.4, 0.5],
        'metadata': {'test': True}
    }
    
    await agent.receive_feedback(feedback)
    print(f"✅ Feedback received. History length: {len(agent.feedback_history)}")
    
    await feedback_loop.stop()
    print("✅ Feedback loop stopped successfully!")
    
    return True


async def main():
    """Run all tests."""
    print("🧪 Enhanced SaaS-Swarm Test Suite")
    print("=" * 50)
    
    tests = [
        ("Configuration", test_configuration),
        ("Tool Registry", test_tool_registry),
        ("Enhanced Swarm", test_enhanced_swarm),
        ("Feedback Loop", test_feedback_loop)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your enhanced SaaS-Swarm is ready to use.")
        print("\nNext steps:")
        print("1. Create a .env file with your OpenAI API key and email credentials")
        print("2. Run: python saas_swarm/examples/enhanced_email_writer.py")
        print("3. Or use the CLI: swarm --help")
    else:
        print("⚠️  Some tests failed. Please check your configuration and try again.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\nTest failed: {e}")
        sys.exit(1) 
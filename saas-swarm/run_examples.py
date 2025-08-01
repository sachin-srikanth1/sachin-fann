#!/usr/bin/env python3
"""
Example runner for SaaS-Swarm platform.

This script demonstrates the platform capabilities by running
various example swarms and showing their outputs.
"""

import asyncio
import sys
import os

# Add the package to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from saas_swarm.examples.email_writer import run_email_writer_example
from saas_swarm.examples.route_optimizer import run_route_optimizer_example
from saas_swarm.examples.code_review import run_code_review_example


async def run_all_examples():
    """Run all available examples."""
    print("=" * 60)
    print("SaaS-SWARM PLATFORM DEMONSTRATION")
    print("=" * 60)
    
    examples = [
        ("Email Writer Swarm", run_email_writer_example),
        ("Route Optimizer Swarm", run_route_optimizer_example),
        ("Code Review Swarm", run_code_review_example),
    ]
    
    for name, example_func in examples:
        print(f"\n{'='*20} {name} {'='*20}")
        try:
            await example_func()
            print(f"\n✅ {name} completed successfully!")
        except Exception as e:
            print(f"\n❌ {name} failed: {e}")
        
        print("\n" + "-" * 60)
    
    print("\n🎉 All examples completed!")
    print("\nTo explore more features:")
    print("- Run 'swarm --help' for CLI commands")
    print("- Start the API server with 'python -m saas_swarm.api.main'")
    print("- Check the documentation for advanced usage")


if __name__ == "__main__":
    try:
        asyncio.run(run_all_examples())
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\nDemo failed: {e}")
        sys.exit(1) 
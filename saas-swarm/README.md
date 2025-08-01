# SaaS-Swarm: Swarm-as-a-Service Platform

A modular and efficient Python platform for implementing lightweight, trainable AI agent swarms with flexible topologies and real-time collaboration.

## Features

- **Modular Agent System**: Define lightweight AI agents with custom inference functions
- **Flexible Swarm Topologies**: Mesh, Star, Ring, and Hierarchical configurations
- **Real-time Communication**: Asynchronous message passing between agents
- **Advanced Tool Integration**: OpenAI research, email sending, and pluggable external tools
- **Feedback System**: Reward signals and adaptation mechanisms
- **Neural Core**: Lightweight NumPy-based neural networks
- **REST API**: FastAPI endpoints for swarm management
- **CLI Tools**: Command-line interface for swarm configuration and deployment
- **OpenAI Integration**: Research and content generation using GPT models
- **Email Functionality**: SMTP-based email sending with configurable providers

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment (see SETUP_GUIDE.md for details)
cp env.example .env
# Edit .env with your OpenAI API key and email credentials

# Test your setup
python3 test_enhanced_swarm.py

# Run enhanced email writer example
python3 saas_swarm/examples/enhanced_email_writer.py
```

## Architecture

```
saas-swarm/
├── core/           # Base agent, swarm, and message classes
├── models/         # Neural network modules
├── tools/          # Pluggable external tools
├── api/            # FastAPI server
├── cli/            # CLI logic
├── examples/       # Sample swarms
└── tests/          # Unit tests
```

## Core Components

### Agent
- Encapsulates input/output sizes and inference functions
- Supports online learning and feedback handling
- Optional tool integration via ToolRegistry

### SwarmTopology
- Base class with pluggable topologies
- Manages agent messaging and execution
- Supports Mesh, Star, Ring, and Hierarchical patterns

### MessageBus
- Asynchronous communication layer
- In-memory queues with extensibility for Redis/ZeroMQ

### FeedbackLoop
- Evaluates swarm output and propagates reward signals
- Enables RL-like adaptation and supervised fine-tuning

## API Endpoints

- `POST /run/{swarm_id}` - Run a swarm with given input
- `POST /feedback/{agent_id}` - Submit feedback signal to agent
- `GET /swarms` - List all swarms
- `GET /agents` - List all agents

## Examples

See `examples/` directory for sample swarms:
- **Enhanced Email Writer**: OpenAI research + email sending
- **Route Optimization**: Multi-agent route planning
- **Code Review**: Hierarchical code analysis
- **Basic Email Writer**: Simple research and writing agents

## Enhanced Features

### OpenAI Integration
- Research topics using GPT models
- Generate content and summaries
- Configurable models and parameters

### Email Functionality
- SMTP-based email sending
- Support for Gmail, Outlook, Yahoo, and custom providers
- Secure credential management

### Feedback Loops
- Evaluate research quality
- Track email delivery and response rates
- Adapt agent behavior based on feedback

## Development

```bash
# Run tests
python -m pytest tests/

# Start API server
python -m saas_swarm.api.main

# Run CLI
python -m saas_swarm.cli.main
```

## Extensibility

The platform is designed for easy extension:
- Add new agent types
- Implement custom swarm topologies
- Integrate external tools and APIs
- Deploy to edge devices or WASM 
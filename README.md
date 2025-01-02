# Evolvra AI

Evolvra AI is a decentralized AI framework designed to facilitate the creation and management of autonomous agent networks on the Solana blockchain. By combining decision-making algorithms, peer-to-peer communication, and efficient state management, Evolvra AI enables secure, scalable, and customizable agent behaviors for diverse use cases, such as distributed learning, collaborative decision-making, and task execution.

## Features

- **Multi-Agent Architecture**: Evolvra AI supports the creation, deployment, and management of networks comprising multiple autonomous agents. Agents in the network can communicate, collaborate, and share resources to accomplish complex tasks.
- **Solana Blockchain Integration**: Agents' state information is securely stored and managed on the Solana blockchain, leveraging its high throughput and decentralization for scalability and fault tolerance.
- **Advanced Communication Protocols**: Peer-to-peer communication is facilitated between agents using robust encoding and decoding techniques, ensuring secure and reliable message exchange.
- **Task Scheduling**: Evolvra AI includes a built-in task scheduling system that allows agents to assign, prioritize, and execute tasks in an efficient and flexible manner.
- **Extensibility**: The framework allows users to extend agent functionality by adding custom plugins or behaviors, making it adaptable to various domains and use cases.
- **Customizable Behaviors**: Agents can be fine-tuned to specialize in different capabilities, including learning, collaborative problem solving, and efficient task execution, giving developers complete control over agent interactions.
- **Resource Management**: The framework efficiently manages agent memory, communication bandwidth, and processing resources to ensure optimal performance, even with large-scale networks.

---

## Project Structure

```
Evolvra/
├── agents/
│   ├── __init__.py
│   ├── autonomous.py
│   ├── base.py
│   ├── collaborative.py
│
├── blockchain/
│   ├── __init__.py
│   ├── solana.py
│
├── core/
│   ├── __init__.py
│   ├── messaging.py
│   ├── neural.py
│   ├── scheduler.py
│   ├── state.py
│   ├── tasks.py
│   ├── tasks_implementations.py
│
├── protocols/
│   ├── __init__.py
│   ├── communication.py
│
├── utils/
│   ├── __init__.py
│   ├── cryptography.py
│   ├── logger.py
│
├── examples/
│   ├── __init__.py
│   ├── basic_agent.py
│   ├── multi_agent_network.py
│
├── tests/
│   ├── __init__.py
│   ├── test_agents.py
│   ├── test_protocols.py
│
├── .github/
│   ├── workflows/
│       ├── ci.yml
│
├── config/
│   ├── config.yaml
│
├── __init__.py
├── Dockerfile
├── .env
├── pyproject.toml
├── requirements.txt
├── setup.py
├── .gitignore
├── LICENSE
├── CONTRIBUTING.md
└── README.md
```

---

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Install Evolvra AI
```bash
pip install evolvra-ai
```

---

## Quick Start Guide

This guide will help you set up a simple agent and run it in a network.

### Create an Agent
```python
from evolvra.agents.base import AgentConfig, BaseAgent

# Define agent configuration
config = AgentConfig(
    name="agent_1",
    capabilities=["task_execution", "learning"]
)

# Create an agent
class SimpleAgent(BaseAgent):
    async def process_message(self, message):
        self.logger.info(f"Processing message: {message}")

    async def make_decision(self, context):
        return {"type": "task", "action": "example_action"}

agent = SimpleAgent(config=config)
```

### Initialize the Network
```python
from evolvra.core.messaging import MessageBus
from evolvra.core.state import StateManager

# Set up a message bus and state manager
message_bus = MessageBus()
state_manager = StateManager()

# Add the agent to the network
agent.message_bus = message_bus
agent.state = state_manager

# Start the agent
import asyncio
asyncio.run(agent.start())
```

---

## Detailed Module Overview

### `agents.base.py`
- **`AgentConfig`**: Configuration class for defining agent properties like memory size, learning rate, and capabilities.
- **`BaseAgent`**: The foundational class for all agents, providing methods for:
  - Communication (`send_message`, `connect`)
  - Decision-making (`make_decision`)
  - Task handling (`_process_task_queue`, `_execute_decision`)

### `core.messaging.py`
Handles message exchange between agents using an asynchronous message bus. It ensures secure and reliable communication.

### `core.scheduler.py`
Provides task scheduling capabilities, allowing agents to queue, prioritize, and execute tasks.

### `protocols.communication.py`
Defines encoding and decoding mechanisms for secure message exchange.

### `blockchain.solana.py`
Interacts with the Solana blockchain for decentralized state storage and retrieval.

### `utils.logger.py`
Centralized logging utility for consistent debug and information logs across all modules.

---

## Contributing

We welcome contributions to Evolvra AI! Please follow these steps:
1. Fork the repository.
2. Clone your fork locally.
3. Submit a pull request with detailed descriptions of your changes.

## Follow us on X for updates

https://x.com/evolvra_ai

# Gitbook

Welcome to the Evolvra project! For detailed documentation, check out our GitBook:

[Visit the Evolvra GitBook](https://evolvra-ai.gitbook.io/evolvra)


---

## License

Evolvra AI is licensed under the MIT License. See `LICENSE` for details.

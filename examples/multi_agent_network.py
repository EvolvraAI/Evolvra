from evolvra.agents.collaborative import CollaborativeAgent
from evolvra.core.messaging import MessageBus
from evolvra.core.state import StateManager
from evolvra.agents.base import AgentConfig
import asyncio

async def setup_network():
    # Shared state and message bus for simplicity in this example
    state_manager = StateManager()
    message_bus = MessageBus()

    # Create agents
    config1 = AgentConfig(name="Agent1", capabilities=["collaboration"])
    agent1 = CollaborativeAgent(config1)
    agent1.state = state_manager
    agent1.message_bus = message_bus

    config2 = AgentConfig(name="Agent2", capabilities=["collaboration"])
    agent2 = CollaborativeAgent(config2)
    agent2.state = state_manager
    agent2.message_bus = message_bus

    # Establish connections
    await agent1.connect(agent2)
    await agent2.connect(agent1)

    # Start agents
    await asyncio.gather(agent1.start(), agent2.start())

    # Simulate knowledge sharing
    knowledge = {"shared_key": "shared_value"}
    await agent1.share_knowledge(agent2.id, knowledge)

    # Stop agents after demonstration
    await asyncio.sleep(2)
    await agent1.stop()
    await agent2.stop()

if __name__ == "__main__":
    asyncio.run(setup_network())
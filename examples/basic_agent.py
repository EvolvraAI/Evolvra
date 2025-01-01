from evolvra.agents.base import BaseAgent, AgentConfig
import asyncio

class BasicAgent(BaseAgent):
    """
    A simple implementation of the BaseAgent class.
    """
    async def process_message(self, message):
        self.logger.info(f"Processing message: {message}")

    async def make_decision(self, context):
        self.logger.info(f"Making decision based on context: {context}")
        return {"type": "log", "content": "Decision logged"}

async def main():
    config = AgentConfig(name="BasicAgent", capabilities=["logging"])
    agent = BasicAgent(config)

    await agent.start()

    # Simulating sending and processing a message
    message = {"type": "test", "content": "Hello from BasicAgent"}
    await agent.process_message(message)

    # Stop the agent
    await asyncio.sleep(1)
    await agent.stop()

if __name__ == "__main__":
    asyncio.run(main())

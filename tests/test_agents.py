import unittest
from unittest.mock import AsyncMock, MagicMock
from evolvra.agents.base import BaseAgent, AgentConfig
from evolvra.agents.collaborative import CollaborativeAgent

class TestBaseAgent(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        config = AgentConfig(name="test_agent", capabilities=["test_capability"])
        self.agent = BaseAgent(config)
        self.agent.state = MagicMock()
        self.agent.message_bus = MagicMock()
        self.agent.protocol = MagicMock()

    async def test_start_stop(self):
        self.agent._process_task_queue = AsyncMock()
        self.agent._listen_for_messages = AsyncMock()
        
        await self.agent.start()
        self.assertTrue(self.agent.running)
        
        await self.agent.stop()
        self.assertFalse(self.agent.running)

    async def test_send_message(self):
        self.agent.protocol.encode_message = MagicMock(return_value="encoded_message")
        self.agent.message_bus.send = AsyncMock(return_value=True)
        
        success = await self.agent.send_message("target_id", {"content": "test"})
        self.assertTrue(success)
        self.agent.protocol.encode_message.assert_called_once_with({"content": "test"})
        self.agent.message_bus.send.assert_called_once_with("target_id", "encoded_message")

class TestCollaborativeAgent(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        config = AgentConfig(name="collaborative_agent", capabilities=["collaboration"])
        self.agent = CollaborativeAgent(config, collaboration_strategy="default")

    async def test_process_message_knowledge_share(self):
        self.agent._update_shared_knowledge = MagicMock()
        message = {"type": "knowledge_share", "content": {"key": "value"}}
        
        await self.agent.process_message(message)
        self.agent._update_shared_knowledge.assert_called_once_with({"key": "value"})

    async def test_process_message_task_assignment(self):
        self.agent._handle_task_assignment = AsyncMock()
        message = {"type": "task_assignment", "content": {"task": "do_something"}}
        
        await self.agent.process_message(message)
        self.agent._handle_task_assignment.assert_awaited_once_with({"task": "do_something"})

    async def test_share_knowledge(self):
        self.agent.send_message = AsyncMock(return_value=True)
        knowledge = {"fact": "shared_data"}
        
        await self.agent.share_knowledge("target_id", knowledge)
        self.agent.send_message.assert_awaited_once_with("target_id", {"type": "knowledge_share", "content": knowledge})

if __name__ == "__main__":
    unittest.main()
from typing import Dict, Any, List
from .base import BaseAgent

class CollaborativeAgent(BaseAgent):
    """
    A specialized agent designed for collaborative tasks.
    """
    def __init__(self, config, collaboration_strategy: str = "default"):
        super().__init__(config)
        self.collaboration_strategy = collaboration_strategy
        self.shared_knowledge = {}

    async def process_message(self, message: Dict[str, Any]) -> None:
        """
        Process incoming messages and update shared knowledge or tasks.
        """
        try:
            if message.get("type") == "knowledge_share":
                self._update_shared_knowledge(message["content"])
            elif message.get("type") == "task_assignment":
                await self._handle_task_assignment(message["content"])
            else:
                self.logger.warning(f"Unknown message type: {message['type']}")
        except Exception as e:
            self.logger.error(f"Failed to process message: {str(e)}")

    async def make_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make decisions based on the collaboration strategy and current context.
        """
        if self.collaboration_strategy == "default":
            decision = {"type": "task_contribution", "content": "Performing assigned task"}
        else:
            decision = {"type": "adaptive_contribution", "content": "Adapting strategy based on peers"}
        self.logger.info(f"Decision made: {decision}")
        return decision

    async def _handle_task_assignment(self, task: Dict[str, Any]) -> None:
        """
        Handle tasks assigned by other agents or systems.
        """
        self.logger.info(f"Received task assignment: {task}")
        await self._task_queue.put(task)

    def _update_shared_knowledge(self, knowledge: Dict[str, Any]) -> None:
        """
        Update shared knowledge base.
        """
        self.shared_knowledge.update(knowledge)
        self.logger.debug(f"Shared knowledge updated: {knowledge}")

    async def share_knowledge(self, target_agent_id: str, knowledge: Dict[str, Any]) -> None:
        """
        Share knowledge with another agent.
        """
        message = {"type": "knowledge_share", "content": knowledge}
        success = await self.send_message(target_agent_id, message)
        if success:
            self.logger.info(f"Shared knowledge with {target_agent_id}: {knowledge}")
        else:
            self.logger.error(f"Failed to share knowledge with {target_agent_id}")

    async def start_collaboration(self) -> None:
        """
        Start collaborative workflows.
        """
        self.logger.info(f"Collaborative agent {self.config.name} starting collaboration")
        while self.running:
            try:
                context = await self.state.get_context()
                decision = await self.make_decision(context)
                await self._execute_decision(decision)
            except Exception as e:
                self.logger.error(f"Error during collaboration: {str(e)}")

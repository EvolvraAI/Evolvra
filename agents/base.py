import uuid
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import asyncio
import logging

from ..core.messaging import MessageBus
from ..core.state import StateManager
from ..protocols.communication import CommunicationProtocol
from ..utils.logger import get_logger

@dataclass
class AgentConfig:
    """Configuration class for agents"""
    name: str
    capabilities: List[str]
    memory_size: int = 1000
    learning_rate: float = 0.01
    max_connections: int = 10
    extra_params: Dict[str, Any] = field(default_factory=dict)

class BaseAgent(ABC):
    """Base class for all agents in the Evolvra AI system"""
    
    def __init__(self, config: AgentConfig):
        self.id = str(uuid.uuid4())
        self.config = config
        self.logger = get_logger(f"agent.{config.name}")
        self.state = StateManager()
        self.message_bus = MessageBus()
        self.protocol = CommunicationProtocol()
        self.running = False
        self._connections: Dict[str, 'BaseAgent'] = {}
        self._task_queue = asyncio.Queue()
        
    @abstractmethod
    async def process_message(self, message: Dict[str, Any]) -> None:
        """Process incoming messages"""
        pass
    
    @abstractmethod
    async def make_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Make autonomous decisions based on current context"""
        pass
    
    async def send_message(self, target_id: str, message: Dict[str, Any]) -> bool:
        """Send message to another agent"""
        try:
            encoded_message = self.protocol.encode_message(message)
            await self.message_bus.send(target_id, encoded_message)
            self.logger.debug(f"Sent message to {target_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to send message: {str(e)}")
            return False
            
    async def connect(self, other_agent: 'BaseAgent') -> bool:
        """Establish connection with another agent"""
        if len(self._connections) >= self.config.max_connections:
            return False
        
        self._connections[other_agent.id] = other_agent
        await self.protocol.establish_connection(other_agent)
        return True
        
    async def start(self) -> None:
        """Start the agent's main loop"""
        self.running = True
        self.logger.info(f"Agent {self.config.name} starting")
        
        try:
            await asyncio.gather(
                self._process_task_queue(),
                self._listen_for_messages()
            )
        except Exception as e:
            self.logger.error(f"Error in agent main loop: {str(e)}")
            self.running = False
            
    async def stop(self) -> None:
        """Stop the agent"""
        self.running = False
        self.logger.info(f"Agent {self.config.name} stopping")
        
    async def _process_task_queue(self) -> None:
        """Process tasks in the queue"""
        while self.running:
            try:
                task = await self._task_queue.get()
                context = await self.state.get_context()
                decision = await self.make_decision(context)
                await self._execute_decision(decision)
                self._task_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error processing task: {str(e)}")
                
    async def _listen_for_messages(self) -> None:
        """Listen for incoming messages"""
        while self.running:
            try:
                message = await self.message_bus.receive()
                decoded_message = self.protocol.decode_message(message)
                await self.process_message(decoded_message)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error processing message: {str(e)}")
                
    async def _execute_decision(self, decision: Dict[str, Any]) -> None:
        """Execute a decision made by the agent"""
        action_type = decision.get('type')
        if not action_type:
            self.logger.error("Decision missing action type")
            return
            
        handler = getattr(self, f"_handle_{action_type}", None)
        if handler:
            await handler(decision)
        else:
            self.logger.warning(f"No handler for action type: {action_type}")
            
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} id={self.id} name={self.config.name}>"
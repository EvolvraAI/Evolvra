from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass
import asyncio
import json
from datetime import datetime
import uuid

from ..blockchain.solana import SolanaManager
from ..utils.logger import get_logger

@dataclass
class TaskDefinition:
    """Defines a task type in the system"""
    name: str
    description: str
    parameters: Dict[str, Dict[str, Any]]
    required_capabilities: List[str]
    estimated_duration: float
    resource_requirements: Dict[str, float]
    verification_method: str
    solana_program_id: Optional[str] = None

class TaskRegistry:
    """Registry for available task types"""
    
    def __init__(self):
        self.tasks: Dict[str, TaskDefinition] = {}
        self.logger = get_logger("task_registry")
        
    def register_task(self, task_def: TaskDefinition) -> None:
        """Register a new task type"""
        self.tasks[task_def.name] = task_def
        self.logger.info(f"Registered task type: {task_def.name}")
        
    def get_task(self, name: str) -> Optional[TaskDefinition]:
        """Get task definition by name"""
        return self.tasks.get(name)
        
    def list_tasks(self) -> List[str]:
        """List all registered task types"""
        return list(self.tasks.keys())

class TaskExecutor:
    """Handles task execution and verification"""
    
    def __init__(
        self,
        registry: TaskRegistry,
        solana_manager: Optional[SolanaManager] = None
    ):
        self.registry = registry
        self.solana_manager = solana_manager
        self.logger = get_logger("task_executor")
        self.running_tasks: Dict[str, asyncio.Task] = {}
        
    async def execute_task(
        self,
        task_name: str,
        parameters: Dict[str, Any],
        agent_capabilities: List[str]
    ) -> Dict[str, Any]:
        """Execute a task with given parameters"""
        task_def = self.registry.get_task(task_name)
        if not task_def:
            raise ValueError(f"Unknown task type: {task_name}")
            
        # Validate capabilities
        missing_capabilities = set(task_def.required_capabilities) - set(agent_capabilities)
        if missing_capabilities:
            raise ValueError(f"Missing required capabilities: {missing_capabilities}")
            
        # Validate parameters
        self._validate_parameters(task_def, parameters)
        
        # Generate task ID
        task_id = str(uuid.uuid4())
        
        try:
            # Execute task
            result = await self._run_task(task_def, parameters)
            
            # Verify result
            verified = await self._verify_result(task_def, result)
            if not verified:
                raise Exception("Task result verification failed")
                
            # Store on Solana if configured
            if task_def.solana_program_id and self.solana_manager:
                await self._store_on_solana(task_def, task_id, result)
                
            return {
                'task_id': task_id,
                'status': 'completed',
                'result': result,
                'verified': True
            }
            
        except Exception as e:
            self.logger.error(f"Task execution failed: {str(e)}")
            return {
                'task_id': task_id,
                'status': 'failed',
                'error': str(e)
            }
            
    async def _run_task(
        self,
        task_def: TaskDefinition,
        parameters: Dict[str, Any]
    ) -> Any:
        """Run task implementation"""
        # Implement task execution logic
        # This is where you'd add specific task implementations
        task_implementations = {
            'data_processing': self._run_data_processing,
            'model_training': self._run_model_training,
            'prediction': self._run_prediction,
            'optimization': self._run_optimization,
            'solana_transaction': self._run_solana_transaction
        }
        
        runner = task_implementations.get(task_def.name)
        if not runner:
            raise NotImplementedError(f"Task type not implemented: {task_def.name}")
            
        return await runner(parameters)
        
    async def _verify_result(self, task_def: TaskDefinition, result: Any) -> bool:
        """Verify task result"""
        verification_methods = {
            'hash_check': self._verify_hash,
            'range_check': self._verify_range,
            'solana_verify': self._verify_on_solana,
            'consensus': self._verify_consensus
        }
        
        verifier = verification_methods.get(task_def.verification_method)
        if not verifier:
            raise ValueError(f"Unknown verification method: {task_def.verification_method}")
            
        return await verifier(result)
        
    async def _store_on_solana(
        self,
        task_def: TaskDefinition,
        task_id: str,
        result: Any
    ) -> None:
        """Store task result on Solana blockchain"""
        if not self.solana_manager:
            return
            
        try:
            # Prepare result data
            data = {
                'task_id': task_id,
                'task_type': task_def.name,
                'result_hash': hash(str(result)),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Store on Solana
            signature = await self.solana_manager.store_state(task_id, data)
            
            if not signature:
                raise Exception("Failed to store
import asyncio
from typing import Dict, Any, List, Optional, Callable, Coroutine
from dataclasses import dataclass
import time
import heapq
from datetime import datetime, timedelta
import uuid

from ..utils.logger import get_logger
from .state import StateManager
from ..blockchain.solana import SolanaManager

@dataclass
class Task:
    """Represents a scheduled task in the system"""
    id: str
    name: str
    coroutine: Coroutine
    priority: int
    dependencies: List[str]
    timeout: float
    retries: int
    metadata: Dict[str, Any]
    solana_verification: bool = False
    created_at: float = time.time()
    
    def __lt__(self, other):
        return self.priority > other.priority  # Higher priority first

class TaskScheduler:
    """Advanced task scheduler with Solana integration"""
    
    def __init__(
        self,
        max_concurrent_tasks: int = 10,
        state_manager: Optional[StateManager] = None,
        solana_manager: Optional[SolanaManager] = None
    ):
        self.logger = get_logger("task_scheduler")
        self.max_concurrent_tasks = max_concurrent_tasks
        self.state_manager = state_manager
        self.solana_manager = solana_manager
        
        self.task_queue: List[Task] = []
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.completed_tasks: Dict[str, Any] = {}
        self.failed_tasks: Dict[str, Exception] = {}
        self.task_dependencies: Dict[str, List[str]] = {}
        
        self.running = False
        self._task_results: Dict[str, asyncio.Future] = {}
        
    async def start(self) -> None:
        """Start the task scheduler"""
        self.running = True
        asyncio.create_task(self._scheduler_loop())
        self.logger.info("Task scheduler started")
        
    async def stop(self) -> None:
        """Stop the task scheduler"""
        self.running = False
        # Cancel all running tasks
        for task in self.running_tasks.values():
            task.cancel()
        await asyncio.gather(*self.running_tasks.values(), return_exceptions=True)
        self.logger.info("Task scheduler stopped")
        
    async def schedule_task(
        self,
        name: str,
        coroutine: Coroutine,
        priority: int = 0,
        dependencies: List[str] = None,
        timeout: float = 300,
        retries: int = 3,
        metadata: Dict[str, Any] = None,
        solana_verification: bool = False
    ) -> str:
        """Schedule a new task"""
        task_id = str(uuid.uuid4())
        task = Task(
            id=task_id,
            name=name,
            coroutine=coroutine,
            priority=priority,
            dependencies=dependencies or [],
            timeout=timeout,
            retries=retries,
            metadata=metadata or {},
            solana_verification=solana_verification
        )
        
        # Store dependencies
        if dependencies:
            self.task_dependencies[task_id] = dependencies
            
        # Create result future
        self._task_results[task_id] = asyncio.Future()
        
        # Add to queue
        heapq.heappush(self.task_queue, task)
        self.logger.debug(f"Scheduled task {task_id}: {name}")
        
        return task_id
        
    async def get_task_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Get the result of a task"""
        if task_id not in self._task_results:
            raise KeyError(f"Task {task_id} not found")
            
        try:
            result = await asyncio.wait_for(
                self._task_results[task_id],
                timeout=timeout
            )
            return result
        except asyncio.TimeoutError:
            raise TimeoutError(f"Task {task_id} result not available within timeout")
            
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a scheduled or running task"""
        # Cancel if running
        if task_id in self.running_tasks:
            self.running_tasks[task_id].cancel()
            return True
            
        # Remove from queue if scheduled
        self.task_queue = [t for t in self.task_queue if t.id != task_id]
        heapq.heapify(self.task_queue)
        
        return True
        
    async def _scheduler_loop(self) -> None:
        """Main scheduler loop"""
        while self.running:
            try:
                await self._process_next_task()
                await asyncio.sleep(0.1)  # Prevent CPU overload
            except Exception as e:
                self.logger.error(f"Error in scheduler loop: {str(e)}")
                
    async def _process_next_task(self) -> None:
        """Process the next task in queue"""
        if not self.task_queue or len(self.running_tasks) >= self.max_concurrent_tasks:
            return
            
        task = heapq.heappop(self.task_queue)
        
        # Check dependencies
        if not await self._check_dependencies(task):
            heapq.heappush(self.task_queue, task)
            return
            
        # Start task execution
        asyncio_task = asyncio.create_task(
            self._execute_task(task)
        )
        self.running_tasks[task.id] = asyncio_task
        
    async def _execute_task(self, task: Task) -> None:
        """Execute a task with retries and timeout"""
        for attempt in range(task.retries + 1):
            try:
                # Execute with timeout
                result = await asyncio.wait_for(
                    task.coroutine,
                    timeout=task.timeout
                )
                
                # Verify on Solana if required
                if task.solana_verification and self.solana_manager:
                    verified = await self._verify_on_solana(task, result)
                    if not verified:
                        raise Exception("Failed to verify task on Solana")
                        
                # Store result
                self.completed_tasks[task.id] = result
                self._task_results[task.id].set_result(result)
                
                # Update state
                if self.state_manager:
                    await self.state_manager.update_task_state(task.id, {
                        'status': 'completed',
                        'result': result,
                        'completed_at': time.time()
                    })
                    
                self.logger.info(f"Task {task.id} completed successfully")
                break
                
            except asyncio.TimeoutError:
                self.logger.warning(f"Task {task.id} timed out (attempt {attempt + 1}/{task.retries + 1})")
                if attempt == task.retries:
                    await self._handle_task_failure(task, "Task timed out")
                    
            except Exception as e:
                self.logger.error(f"Task {task.id} failed: {str(e)}")
                if attempt == task.retries:
                    await self._handle_task_failure(task, str(e))
                    
            finally:
                if task.id in self.running_tasks:
                    del self.running_tasks[task.id]
                    
    async def _verify_on_solana(self, task: Task, result: Any) -> bool:
        """Verify task completion on Solana blockchain"""
        try:
            if not self.solana_manager:
                return False
                
            # Store task result hash
            signature = await self.solana_manager.store_state(
                task.id,
                {
                    'task_id': task.id,
                    'result_hash': hash(str(result)),
                    'timestamp': time.time()
                }
            )
            
            # Verify transaction
            if signature:
                verified = await self.solana_manager.verify_transaction(signature)
                return verified
                
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to verify task on Solana: {str(e)}")
            return False
            
    async def _check_dependencies(self, task: Task) -> bool:
        """Check if all task dependencies are met"""
        for dep_id in task.dependencies:
            if dep_id not in self.completed_tasks:
                return False
        return True
        
    async def _handle_task_failure(self, task: Task, error: str) -> None:
        """Handle task failure"""
        self.failed_tasks[task.id] = Exception(error)
        self._task_results[task.id].set_exception(Exception(error))
        
        if self.state_manager:
            await self.state_manager.update_task_state(task.id, {
                'status': 'failed',
                'error': error,
                'failed_at': time.time()
            })
            
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get current status of a task"""
        if task_id in self.completed_tasks:
            return {'status': 'completed', 'result': self.completed_tasks[task_id]}
        elif task_id in self.failed_tasks:
            return {'status': 'failed', 'error': str(self.failed_tasks[task_id])}
        elif task_id in self.running_tasks:
            return {'status': 'running'}
        elif any(t.id == task_id for t in self.task_queue):
            return {'status': 'scheduled'}
        else:
            return {'status': 'not_found'}
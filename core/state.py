from typing import Dict, Any, List, Optional, Set
import asyncio
from dataclasses import dataclass, field
import json
import time
from datetime import datetime
import uuid

from ..blockchain.solana import SolanaManager
from ..utils.logger import get_logger
from ..utils.crypto import hash_state

@dataclass
class StateSnapshot:
    """Represents a snapshot of agent state"""
    id: str
    timestamp: float
    state: Dict[str, Any]
    hash: str
    parent_hash: Optional[str] = None
    verified: bool = False

class StateManager:
    """Manages agent state and synchronization"""
    
    def __init__(
        self,
        solana_manager: Optional[SolanaManager] = None,
        max_history: int = 1000
    ):
        self.logger = get_logger("state_manager")
        self.solana_manager = solana_manager
        self.max_history = max_history
        
        self._current_state: Dict[str, Any] = {}
        self._state_history: List[StateSnapshot] = []
        self._locks: Dict[str, asyncio.Lock] = {}
        self._subscribers: Dict[str, Set[callable]] = {}
        
    async def update(
        self,
        updates: Dict[str, Any],
        verify: bool = True
    ) -> Optional[str]:
        """Update current state"""
        try:
            # Create state snapshot
            snapshot = await self._create_snapshot(updates)
            
            # Verify and store on Solana if configured
            if verify and self.solana_manager:
                verified = await self._verify_and_store(snapshot)
                if not verified:
                    raise Exception("State verification failed")
                    
            # Update current state
            async with self._get_lock('state'):
                self._current_state.update(updates)
                self._state_history.append(snapshot)
                
                # Trim history if needed
                if len(self._state_history) > self.max_history:
                    self._state_history.pop(0)
                    
            # Notify subscribers
            await self._notify_subscribers(updates)
            
            return snapshot.id
            
        except Exception as e:
            self.logger.error(f"State update failed: {str(e)}")
            return None
            
    async def get_state(self, key: Optional[str] = None) -> Any:
        """Get current state or specific key"""
        async with self._get_lock('state'):
            if key is None:
                return self._current_state.copy()
            return self._current_state.get(key)
            
    async def get_history(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> List[StateSnapshot]:
        """Get state history within time range"""
        async with self._get_lock('state'):
            if start_time is None and end_time is None:
                return self._state_history.copy()
                
            return [
                snapshot for snapshot in self._state_history
                if (start_time is None or snapshot.timestamp >= start_time) and
                (end_time is None or snapshot.timestamp <= end_time)
            ]
            
    async def subscribe(self, callback: callable, keys: Optional[List[str]] = None) -> None:
        """Subscribe to state changes"""
        if keys is None:
            keys = ['*']  # Subscribe to all changes
            
        for key in keys:
            if key not in self._subscribers:
                self._subscribers[key] = set()
            self._subscribers[key].add(callback)
            
    async def unsubscribe(self, callback: callable, keys: Optional[List[str]] = None) -> None:
        """Unsubscribe from state changes"""
        if keys is None:
            keys = list(self._subscribers.keys())
            
        for key in keys:
            if key in self._subscribers:
                self._subscribers[key].discard(callback)
                
    async def rollback(self, snapshot_id: str) -> bool:
        """Rollback state to specific snapshot"""
        try:
            async with self._get_lock('state'):
                # Find snapshot
                snapshot = next(
                    (s for s in self._state_history if s.id == snapshot_id),
                    None
                )
                
                if not snapshot:
                    return False
                    
                # Verify snapshot if possible
                if self.solana_manager:
                    verified = await self._verify_snapshot(snapshot)
                    if not verified:
                        return False
                        
                # Rollback state
                self._current_state = snapshot.state.copy()
                
                # Trim history
                idx = self._state_history.index(snapshot)
                self._state_history = self._state_history[:idx + 1]
                
                return True
                
        except Exception as e:
            self.logger.error(f"State rollback failed: {str(e)}")
            return False
            
    async def merge_states(self, other_state: Dict[str, Any]) -> bool:
        """Merge another state with current state"""
        try:
            async with self._get_lock('state'):
                # Create merged state
                merged_state = self._current_state.copy()
                merged_state.update(other_state)
                
                # Create snapshot
                snapshot = await self._create_snapshot(merged_state)
                
                # Verify if possible
                if self.solana_manager:
                    verified = await self._verify_and_store(snapshot)
                    if not verified:
                        return False
                        
                # Update state
                self._current_state = merged_state
                self._state_history.append(snapshot)
                
                # Notify subscribers
                await self._notify_subscribers(other_state)
                
                return True
                
        except Exception as e:
            self.logger.error(f"State merge failed: {str(e)}")
            return False
            
    async def _create_snapshot(self, updates: Dict[str, Any]) -> StateSnapshot:
        """Create new state snapshot"""
        # Get parent hash
        parent_hash = self._state_history[-1].hash if self._state_history else None
        
        # Create new state
        new_state = self._current_state.copy()
        new_state.update(updates)
        
        # Create snapshot
        snapshot = StateSnapshot(
            id=str(uuid.uuid4()),
            timestamp=time.time(),
            state=new_state,
            hash=hash_state(new_state),
            parent_hash=parent_hash
        )
        
        return snapshot
        
    async def _verify_and_store(self, snapshot: StateSnapshot) -> bool:
        """Verify and store snapshot on Solana"""
        if not self.solana_manager:
            return True
            
        try:
            # Store on Solana
            signature = await self.solana_manager.store_state(
                snapshot.id,
                {
                    'hash': snapshot.hash,
                    'parent_hash': snapshot.parent_hash,
                    'timestamp': snapshot.timestamp
                }
            )
            
            if not signature:
                return False
                
            # Verify transaction
            verified = await self.solana_manager.verify_transaction(signature)
            return verified
            
        except Exception as e:
            self.logger.error(f"State verification failed: {str(e)}")
            return False
            
    async def _verify_snapshot(self, snapshot: StateSnapshot) -> bool:
        """Verify a state snapshot"""
        if not self.solana_manager:
            return True
            
        try:
            # Get stored state from Solana
            stored_state = await self.solana_manager.retrieve_state(snapshot.id)
            if not stored_state:
                return False
                
            # Verify hash matches
            return stored_state['hash'] == snapshot.hash
            
        except Exception as e:
            self.logger.error(f"Snapshot verification failed: {str(e)}")
            return False
            
    def _get_lock(self, name: str) -> asyncio.Lock:
        """Get or create a named lock"""
        if name not in self._locks:
            self._locks[name] = asyncio.Lock()
        return self._locks[name]
        
    async def _notify_subscribers(self, updates: Dict[str, Any]) -> None:
        """Notify subscribers of state changes"""
        tasks = []
        
        # Notify specific key subscribers
        for key, value in updates.items():
            if key in self._subscribers:
                for callback in self._subscribers[key]:
                    tasks.append(callback(key, value))
                    
        # Notify global subscribers
        if '*' in self._subscribers:
            for callback in self._subscribers['*']:
                tasks.append(callback(updates))
                
        # Execute callbacks
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
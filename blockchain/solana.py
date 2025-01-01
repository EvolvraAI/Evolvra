from typing import Dict, Any, List, Optional
import asyncio
import base58
from solana.rpc.async_api import AsyncClient
from solana.transaction import Transaction
from solana.system_program import TransactionInstruction, transfer
from solana.keypair import Keypair
import json

from ..utils.logger import get_logger

class SolanaManager:
    """Manages Solana blockchain interactions for the agent network"""
    
    def __init__(
        self,
        rpc_url: str = "https://api.mainnet-beta.solana.com",
        commitment: str = "confirmed"
    ):
        self.logger = get_logger("solana_manager")
        self.client = AsyncClient(rpc_url, commitment=commitment)
        self.program_id = None
        self.keypair = None
        
    async def initialize(self, program_id: str, keypair: Optional[Keypair] = None) -> bool:
        """Initialize the Solana manager"""
        try:
            self.program_id = program_id
            self.keypair = keypair or Keypair()
            
            # Verify connection
            await self.client.is_connected()
            self.logger.info("Solana manager initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Solana manager: {str(e)}")
            return False
            
    async def store_state(self, agent_id: str, state: Dict[str, Any]) -> Optional[str]:
        """Store agent state on Solana blockchain"""
        try:
            # Prepare state data
            data = json.dumps(state).encode()
            
            # Create instruction
            instruction = TransactionInstruction(
                keys=[
                    {"pubkey": self.keypair.public_key, "isSigner": True, "isWritable": True}
                ],
                program_id=self.program_id,
                data=data
            )
            
            # Create and sign transaction
            transaction = Transaction().add(instruction)
            transaction.sign(self.keypair)
            
            # Send transaction
            result = await self.client.send_transaction(transaction)
            signature = result["result"]
            
            self.logger.debug(f"Stored state for agent {agent_id}: {signature}")
            return signature
            
        except Exception as e:
            self.logger.error(f"Failed to store state: {str(e)}")
            return None
            
    async def retrieve_state(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve agent state from Solana blockchain"""
        try:
            # Get account info
            account_info = await self.client.get_account_info(self.keypair.public_key)
            
            if not account_info["result"]["value"]:
                return None
                
            # Decode state data
            data = base58.b58decode(account_info["result"]["value"]["data"][0])
            state = json.loads(data)
            
            return state
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve state: {str(e)}")
            return None
            
    async def verify_transaction(self, signature: str) -> bool:
        """Verify a transaction on the blockchain"""
        try:
            result = await self.client.confirm_transaction(signature)
            return result["result"]
            
        except Exception as e:
            self.logger.error(f"Failed to verify transaction: {str(e)}")
            return False
            
    async def create_agent_account(self, agent_id: str) -> Optional[str]:
        """Create a new account for an agent"""
        try:
            new_keypair = Keypair()
            
            # Calculate required space
            space = 1000  # Adjust based on expected state size
            
            # Calculate minimum balance
            min_balance = await self.client.get_minimum_balance_for_rent_exemption(space)
            
            # Create account
            transaction = Transaction().add(
                transfer(
                    self.keypair.public_key,
                    new_keypair.public_key,
                    min_balance["result"]
                )
            )
            
            # Sign and send transaction
            transaction.sign(self.keypair)
            result = await self.client.send_transaction(transaction)
            
            return str(new_keypair.public_key)
            
        except Exception as e:
            self.logger.error(f"Failed to create agent account: {str(e)}")
            return None
            
    async def monitor_transactions(
        self,
        callback: callable,
        filter_addresses: Optional[List[str]] = None
    ) -> None:
        """Monitor blockchain for relevant transactions"""
        try:
            async for transaction in self.client.transaction_stream():
                if not filter_addresses or any(
                    addr in transaction for addr in filter_addresses
                ):
                    await callback(transaction)
                    
        except Exception as e:
            self.logger.error(f"Error monitoring transactions: {str(e)}")
            
    async def close(self) -> None:
        """Close the Solana connection"""
        try:
            await self.client.close()
            self.logger.info("Solana manager closed")
        except Exception as e:
            self.logger.error(f"Error closing Solana manager: {str(e)}")
            
class SolanaProgram:
    """Custom Solana program for agent operations"""
    
    def __init__(self, program_id: str):
        self.program_id = program_id
        self.logger = get_logger("solana_program")
        
    async def create_instruction(
        self,
        instruction_type: str,
        accounts: List[Dict[str, Any]],
        data: bytes
    ) -> TransactionInstruction:
        """Create a program instruction"""
        return TransactionInstruction(
            keys=accounts,
            program_id=self.program_id,
            data=data
        )
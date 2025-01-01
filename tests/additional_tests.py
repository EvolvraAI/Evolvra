import pytest
import asyncio
from evolvra.core.messaging import MessageBus
from evolvra.protocols.communication import encode_message, decode_message

@pytest.mark.asyncio
async def test_message_bus():
    bus = MessageBus()
    test_message = {"type": "command", "action": "test"}
    
    received_messages = []
    async def message_handler(message):
        received_messages.append(message)
    
    await bus.subscribe("test_channel", message_handler)
    await bus.publish("test_channel", test_message)
    
    # Allow time for message processing
    await asyncio.sleep(0.1)
    assert len(received_messages) == 1
    assert received_messages[0] == test_message

# tests/test_scheduler.py
import pytest
from evolvra.core.scheduler import TaskScheduler
from evolvra.core.tasks import Task

def test_task_scheduling():
    scheduler = TaskScheduler()
    task = Task(id="test_task", priority=1)
    
    scheduler.add_task(task)
    assert scheduler.get_next_task() == task
    assert scheduler.task_count == 1

# tests/test_blockchain.py
import pytest
from evolvra.blockchain.solana import SolanaClient
from unittest.mock import Mock, patch

@pytest.fixture
def mock_solana_client():
    with patch('solana.rpc.api.Client') as mock_client:
        yield SolanaClient(rpc_url="https://api.devnet.solana.com")

def test_blockchain_connection(mock_solana_client):
    assert mock_solana_client.is_connected()
    
def test_state_persistence(mock_solana_client):
    test_data = {"agent_id": "test", "state": "active"}
    result = mock_solana_client.store_state(test_data)
    assert result.success

# tests/test_neural.py
import pytest
import numpy as np
from evolvra.core.neural import NeuralNetwork

def test_neural_network():
    nn = NeuralNetwork(input_size=4, hidden_size=8, output_size=2)
    test_input = np.random.rand(1, 4)
    output = nn.forward(test_input)
    assert output.shape == (1, 2)

# tests/test_security.py
import pytest
from evolvra.utils.cryptography import encrypt_message, decrypt_message

def test_encryption():
    original_message = "test message"
    key = b"test_key_12345678"
    encrypted = encrypt_message(original_message, key)
    decrypted = decrypt_message(encrypted, key)
    assert decrypted == original_message

# tests/__init__.py
"""
Evolvra AI Test Suite

This package contains all test modules for the Evolvra AI framework.
Test modules are organized by component and functionality.

Available test modules:
- test_agents.py: Tests for agent creation and behavior
- test_messaging.py: Tests for inter-agent communication
- test_protocols.py: Tests for communication protocols
- test_scheduler.py: Tests for task scheduling
- test_blockchain.py: Tests for Solana blockchain integration
- test_neural.py: Tests for neural network functionality
- test_security.py: Tests for cryptographic operations
"""

from .test_agents import *
from .test_messaging import *
from .test_protocols import *
from .test_scheduler import *
from .test_blockchain import *
from .test_neural import *
from .test_security import *

__version__ = "0.1.0"
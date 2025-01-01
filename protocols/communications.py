import json
from typing import Any, Dict
from cryptography.fernet import Fernet
from evolvra.utils.crypto import generate_key, sign_message, verify_signature

class Message:
    def __init__(self, sender: str, recipient: str, content: Dict[str, Any], signature: str = None):
        self.sender = sender
        self.recipient = recipient
        self.content = content
        self.signature = signature

    def to_json(self) -> str:
        return json.dumps({
            'sender': self.sender,
            'recipient': self.recipient,
            'content': self.content,
            'signature': self.signature
        })

    @staticmethod
    def from_json(message_str: str):
        data = json.loads(message_str)
        return Message(
            sender=data['sender'],
            recipient=data['recipient'],
            content=data['content'],
            signature=data.get('signature')
        )

class CommunicationProtocol:
    def __init__(self, key: bytes = None):
        self.key = key or generate_key()
        self.cipher = Fernet(self.key)

    def encrypt_message(self, message: Message) -> str:
        raw_message = message.to_json().encode('utf-8')
        encrypted_message = self.cipher.encrypt(raw_message)
        return encrypted_message.decode('utf-8')

    def decrypt_message(self, encrypted_message: str) -> Message:
        decrypted_message = self.cipher.decrypt(encrypted_message.encode('utf-8'))
        return Message.from_json(decrypted_message.decode('utf-8'))

    def send_message(self, sender: str, recipient: str, content: Dict[str, Any]) -> str:
        message = Message(sender=sender, recipient=recipient, content=content)
        message.signature = sign_message(message.to_json())
        return self.encrypt_message(message)

    def receive_message(self, encrypted_message: str) -> Message:
        message = self.decrypt_message(encrypted_message)
        if not verify_signature(message.to_json(), message.signature):
            raise ValueError('Message signature verification failed!')
        return message

# Example usage (for testing only)
if __name__ == '__main__':
    protocol = CommunicationProtocol()
    encrypted = protocol.send_message('agent_1', 'agent_2', {'task': 'execute', 'payload': 'data'})
    print('Encrypted Message:', encrypted)
    
    message = protocol.receive_message(encrypted)
    print('Decrypted Message:', message.content)

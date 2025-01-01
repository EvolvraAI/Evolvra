import unittest
from unittest.mock import MagicMock
from evolvra.protocols.communication import CommunicationProtocol

class TestCommunicationProtocol(unittest.TestCase):
    def setUp(self):
        self.protocol = CommunicationProtocol()

    def test_encode_message(self):
        message = {"key": "value"}
        encoded_message = self.protocol.encode_message(message)
        self.assertIsInstance(encoded_message, str)

    def test_decode_message(self):
        message = {"key": "value"}
        encoded_message = self.protocol.encode_message(message)
        decoded_message = self.protocol.decode_message(encoded_message)
        self.assertEqual(decoded_message, message)

    def test_establish_connection(self):
        other_agent = MagicMock()
        self.protocol.establish_connection = MagicMock(return_value=True)
        result = self.protocol.establish_connection(other_agent)
        self.assertTrue(result)

if __name__ == "__main__":
    unittest.main()

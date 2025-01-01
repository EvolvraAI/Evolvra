import hashlib
import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
import base64
import logging

# Logger utility
def get_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """
    Create and configure a logger.
    
    :param name: Name of the logger
    :param level: Logging level
    :return: Configured logger instance
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger

# Crypto utilities
class CryptoUtils:
    """
    Utility class for cryptographic operations.
    """
    def __init__(self, password: str):
        self.password = password.encode()
        self.backend = default_backend()

    def _derive_key(self, salt: bytes, iterations: int = 100000) -> bytes:
        """
        Derive a cryptographic key using PBKDF2.
        
        :param salt: Salt value for key derivation
        :param iterations: Number of iterations
        :return: Derived key
        """
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=iterations,
            backend=self.backend
        )
        return kdf.derive(self.password)

    def encrypt(self, data: str) -> str:
        """
        Encrypt data using AES-GCM.
        
        :param data: Data to encrypt
        :return: Encrypted data in Base64 format
        """
        salt = os.urandom(16)
        key = self._derive_key(salt)
        iv = os.urandom(12)
        cipher = Cipher(algorithms.AES(key), modes.GCM(iv), backend=self.backend)
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(data.encode()) + encryptor.finalize()
        result = base64.b64encode(salt + iv + encryptor.tag + ciphertext).decode()
        return result

    def decrypt(self, encrypted_data: str) -> str:
        """
        Decrypt data encrypted with AES-GCM.
        
        :param encrypted_data: Encrypted data in Base64 format
        :return: Decrypted data as a string
        """
        decoded = base64.b64decode(encrypted_data)
        salt, iv, tag, ciphertext = decoded[:16], decoded[16:28], decoded[28:44], decoded[44:]
        key = self._derive_key(salt)
        cipher = Cipher(algorithms.AES(key), modes.GCM(iv, tag), backend=self.backend)
        decryptor = cipher.decryptor()
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        return plaintext.decode()

# Utility for hashing
def hash_data(data: str, algorithm: str = "sha256") -> str:
    """
    Generate a cryptographic hash for the given data.
    
    :param data: Data to hash
    :param algorithm: Hashing algorithm to use
    :return: Hexadecimal hash string
    """
    hasher = hashlib.new(algorithm)
    hasher.update(data.encode())
    return hasher.hexdigest()

# Example usage of utilities
if __name__ == "__main__":
    logger = get_logger("utils_example")

    crypto = CryptoUtils(password="securepassword")
    original_data = "This is a secret message."
    encrypted = crypto.encrypt(original_data)
    decrypted = crypto.decrypt(encrypted)

    logger.info(f"Original: {original_data}")
    logger.info(f"Encrypted: {encrypted}")
    logger.info(f"Decrypted: {decrypted}")

    hashed = hash_data(original_data)
    logger.info(f"Hashed: {hashed}")
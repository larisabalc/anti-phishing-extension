import json
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import base64

KEY = b'\x01\x23\x45\x67\x89\xab\xcd\xef\xfe\xdc\xba\x98\x76\x54\x32\x10'
IV = b'\x00\x11\x22\x33\x44\x55\x66\x77\x88\x99\xaa\xbb\xcc\xdd\xee\xff'

class CryptoUtils:
    """
    Utility class for encrypting and decrypting messages using AES encryption.

    Methods:
        encrypt_message(data, key=KEY, iv=IV):
            Encrypts a JSON message using AES encryption.

        decrypt_message(encrypted_object, key=KEY, iv=IV):
            Decrypts an encrypted message using AES decryption.
    """

    @staticmethod
    def encrypt_message(data, key=KEY, iv=IV):
        """
        Encrypts a JSON message using AES encryption.

        Args:
            data (dict): The JSON data to be encrypted.
            key (bytes, optional): The encryption key. Defaults to KEY.
            iv (bytes, optional): The initialization vector. Defaults to IV.

        Returns:
            str: The base64 encoded encrypted message.
        """
        json_string = json.dumps(data)
        data_bytes = json_string.encode('utf-8')

        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()

        padder = padding.PKCS7(128).padder()
        padded_data = padder.update(data_bytes) + padder.finalize()
        encrypted = encryptor.update(padded_data) + encryptor.finalize()

        return base64.b64encode(encrypted).decode('utf-8')

    @staticmethod
    def decrypt_message(encrypted_object, key=KEY, iv=IV):
        """
        Decrypts an encrypted message using AES decryption.

        Args:
            encrypted_object (str): The base64 encoded encrypted message.
            key (bytes, optional): The decryption key. Defaults to KEY.
            iv (bytes, optional): The initialization vector. Defaults to IV.

        Returns:
            str: The decrypted JSON message.
        """
        encrypted_bytes = base64.b64decode(encrypted_object)

        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        decryptor = cipher.decryptor()
        decrypted = decryptor.update(encrypted_bytes) + decryptor.finalize()

        unpadder = padding.PKCS7(128).unpadder()
        unpadded_data = unpadder.update(decrypted) + unpadder.finalize()

        return unpadded_data.decode('utf-8')  
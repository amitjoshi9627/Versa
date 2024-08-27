from cryptography.fernet import Fernet
from huggingface_hub import login


def login_user(access_token: str, login_from_dashboard: bool = False) -> bool:
    try:
        login(token=access_token)
    except ValueError:
        if login_from_dashboard:
            return False
        raise ValueError("Invalid token passed! Login not successful.")
    return True


def decrypt_string(encrypted_token: str, key: bytes | str) -> str:
    """Decrypts an encrypted access token using a given key.

    Args:
        encrypted_token (str): The encrypted access token.
        key (bytes | str): The encryption key.

    Returns:
        str: The decrypted access token.
    """

    cipher = Fernet(key)
    decrypted_token = cipher.decrypt(encrypted_token.encode())
    return decrypted_token.decode()

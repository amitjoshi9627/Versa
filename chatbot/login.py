from cryptography.fernet import Fernet
from huggingface_hub import login


def login_user(access_token: str, login_from_dashboard: bool = False) -> bool:
    """Logs in the user using the provided access token.

    Args:
        access_token (str): The access token to use for login.
        login_from_dashboard (bool, optional): Whether the login is initiated from the dashboard. Defaults to False.

    Returns:
        bool: True if login is successful, False if login fails due to an invalid token and login_from_dashboard is True
    """

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

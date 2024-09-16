import platform
from dataclasses import dataclass


@dataclass
class ResponseMessage:
    """Represents a response message from a chatbot.

    Attributes:
        query (str): The original query that prompted the response.
        response (str): The generated response from the chatbot.
    """

    query: str
    response: str


def get_os() -> str:
    """Gets the operating system on which the code is running.

    Returns:
        str: The name of the operating system.
    """

    return platform.system()

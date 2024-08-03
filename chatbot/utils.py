import platform
from dataclasses import dataclass


@dataclass
class ResponseMessage:
    query: str
    response: str


def get_os() -> str:
    return platform.system()

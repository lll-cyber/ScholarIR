from .deepseek_client import chat_completion as deepseek_chat
from .deepseek_client import deepseek_configured
from .local_client import chat_completion as local_chat

__all__ = ["deepseek_chat", "deepseek_configured", "local_chat"]

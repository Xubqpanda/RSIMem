"""Built-in runtime adapters."""

from .base import RuntimeAdapter
from .cli_agent import ClaudeCodeCliAdapter, CodexCliAdapter
from .openclaw import OpenClawResponsesAdapter
from .openai_compat import OpenAICompatChatAdapter
from .zeroclaw import ZeroClawAdapter
from .hermes import HermesAdapter
from .agent_zero import AgentZeroAdapter
from .nanobot import NanobotAdapter

__all__ = [
    "RuntimeAdapter",
    "OpenAICompatChatAdapter",
    "OpenClawResponsesAdapter",
    "CodexCliAdapter",
    "ClaudeCodeCliAdapter",
    "ZeroClawAdapter",
    "HermesAdapter",
    "AgentZeroAdapter",
    "NanobotAdapter",
]

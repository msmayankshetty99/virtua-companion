"""Provider-neutral application core for Riko.

The names below load on first use (PEP 562), so importing a leaf module (a store, a kernel utility, the tool worker)
no longer imports the whole core and its failure modes with it. Inside app_core, import from the defining module.
"""
from importlib import import_module
from typing import TYPE_CHECKING

_EXPORTS = {
    "AppConfig": "configuration.config", "load_config": "configuration.config",
    "ChatMessage": "kernel.messages", "ModelResponse": "kernel.messages", "ToolCall": "kernel.messages", "ToolResult": "kernel.messages",
    "ModelProvider": "inference.provider", "ChatService": "conversation.chat",
    "JuliaEmotionEngine": "emotion", "EmotionState": "emotion", "ExpressionActionBridge": "emotion",
    "RuntimeEvent": "events.bus", "event_bus": "events.bus", "SessionManager": "runtime.session",
    "ActionController": "runtime.actions", "CompanionAction": "runtime.actions",
}
__all__ = list(_EXPORTS)

if TYPE_CHECKING:
    from .configuration.config import AppConfig, load_config
    from .kernel.messages import ChatMessage, ModelResponse, ToolCall, ToolResult
    from .inference.provider import ModelProvider
    from .conversation.chat import ChatService
    from .emotion import JuliaEmotionEngine, EmotionState, ExpressionActionBridge
    from .events.bus import RuntimeEvent, event_bus
    from .runtime.session import SessionManager
    from .runtime.actions import ActionController, CompanionAction


def __getattr__(name):
    if name not in _EXPORTS: raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = globals()[name] = getattr(import_module("." + _EXPORTS[name], __name__), name)
    return value


def __dir__(): return sorted({*globals(), *__all__})

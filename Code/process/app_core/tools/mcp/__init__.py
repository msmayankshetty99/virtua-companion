"""MCP servers as tools: the stdio and HTTP transports, the tools a server lists, and the servers mcp.json configures."""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from ..tool import RegisteredTool
from .http import HTTPMCPClient
from .stdio import StdioMCPClient, resolve_command


@dataclass
class MCPTool(RegisteredTool):
    """A tool an MCP server runs (tools/call) under remote_name, or under name when register kept the server's own."""
    client: Any = None

    @property
    def cancellable(self): return isinstance(self.client, StdioMCPClient)
    def execute(self, arguments, cancelled=lambda: False):
        name = self.remote_name or self.name
        return self.client.call(name, arguments, cancelled) if self.cancellable else self.client.call(name, arguments)
    def result(self, value):
        """A tools/call result: its structured content, or else the text of its content, and isError."""
        if not isinstance(value, dict): return value, False
        error, value = bool(value.get('isError')), value.get('structuredContent', value.get('content', value))
        if isinstance(value, list): value = '\n'.join(item.get('text', '') for item in value if isinstance(item, dict) and item.get('type') == 'text')
        return value, error
    def abandon(self):
        if self.cancellable: self.client.close()  # a server stuck past the deadline stops; the next call restarts it


def server_tools(client, *, trusted, owner=''):
    """An MCPTool for each tool the client's server lists. Only a trusted server (Riko's own TaskMCP) may mark a tool
    read-only through its readOnlyHint annotation."""
    return [MCPTool(definition['name'], definition.get('description', ''), definition.get('inputSchema', {'type': 'object'}), client=client,
        owner=owner, read_only=trusted and (definition.get('annotations') or {}).get('readOnlyHint') is True) for definition in client.list_tools()]


def configured_servers(path):
    """{name: server} from the mcpServers (or servers) section of the mcp.json at path; none without the file."""
    raw = json.loads(path.read_text(encoding='utf-8')) if path and path.exists() else {}
    return raw.get('mcpServers', raw.get('servers', {}))


def connect(server):
    """A started client for one configured server: url means HTTP, command a stdio process."""
    return HTTPMCPClient(server['url'], server.get('headers')) if server.get('url') else StdioMCPClient(server['command'], server.get('args'), server.get('env'))


__all__ = ['HTTPMCPClient', 'MCPTool', 'StdioMCPClient', 'configured_servers', 'connect', 'resolve_command', 'server_tools']

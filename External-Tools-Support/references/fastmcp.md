# FastMCP Reference

The local source at `Huawei/Projects/MCP/fastmcp` contains a full FastMCP checkout with docs, tests, examples, and SDK internals. This repository does not vendor that full upstream tree.

Use FastMCP when a Python tool should be exposed as an MCP server with decorators, stdio transport, SSE transport, or streamable HTTP transport. The copied examples use FastMCP in:

- `../android-mcp/main.py`
- `../simple-mcp/math_server.py`
- `../simple-mcp/memory_server.py`

For production use, pin FastMCP or `mcp` dependencies in the target environment and keep transport URLs explicit.

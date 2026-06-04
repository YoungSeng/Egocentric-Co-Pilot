# Simple MCP Examples

This folder contains compact MCP examples copied from `Huawei/Projects/MCP/simple`.

## Files

- `math_server.py`: stdio FastMCP server for evaluation, equation solving, differentiation, and integration.
- `memory_server.py`: SSE FastMCP server for creating, listing, and deleting simple memory entries.
- `mcp_client.py`: Qwen and LangChain MCP adapter sketch for loading tools from multiple MCP servers.
- `utils/memory_fuc.py`: small helper added during curation so `memory_server.py` is self-contained.

## Run

```bash
python math_server.py
python memory_server.py
```

The memory server writes `memory_store.json` in the current working directory.

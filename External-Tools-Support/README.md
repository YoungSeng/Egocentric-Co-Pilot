# External Tools Support

## Contents

| Folder | Purpose |
| --- | --- |
| `android-mcp/` | Lightweight Android MCP server exposing device state, tap, swipe, type, drag, press, notification, wait, and shell tools through FastMCP. |
| `simple-mcp/` | Small MCP examples for math tools, memory tools, and a Qwen-based multiserver MCP client integration sketch. |
| `mcp-client-examples/` | FastMCP client scripts for listing tools and driving a mobile YouTube scenario over SSE. |
| `glass2phone-quickstart/` | Selected phone/glasses bridge examples: WebSocket audio tests, Ktor WebSocket sample, Android phone server, glasses app, simple server app, and voice-to-text app source. |
| `references/` | Lightweight notes for FastMCP, Kotlin MCP SDK, Mobile Next MCP, and Glass2Phone bridge context. |

## Quick Start

### Android MCP

```bash
cd External-Tools-Support/android-mcp
uv sync
uv run main.py --emulator
```

Use `--emulator` for `emulator-5554`; omit it for a connected physical Android device. Android Platform Tools and a working `adb` setup are required.

### Simple MCP Math Server

```bash
cd External-Tools-Support/simple-mcp
python math_server.py
```

The math server runs over stdio and requires `mcp`, `sympy`, and `numpy`.

### Simple MCP Memory Server

```bash
cd External-Tools-Support/simple-mcp
python memory_server.py
```

The memory server runs over SSE at `127.0.0.1:4201` and writes `memory_store.json` in the current working directory.

### MCP Client Examples

```bash
cd External-Tools-Support/mcp-client-examples
python test_mcp_connection.py
```

The mobile examples expect an MCP server at `http://localhost:6001/mcp` unless edited.

### Glass2Phone WebSocket Test

```bash
cd External-Tools-Support/glass2phone-quickstart/websocket-test
python TestServer.py
```

Then point the Android/glasses sender at the displayed WebSocket endpoint, or use `TestClient.py` after updating its `SERVER_URL`.

## Attribution And Licenses

- `android-mcp/` is derived from CursorTouch/Android-MCP and includes its upstream MIT license.
- `mcp-client-examples/` and Mobile Next notes reference Mobile Next MCP, Apache-2.0 upstream.
- `references/` summarizes copied local experiments and points to upstream packages rather than vendoring full SDKs.

## Integration Notes

These examples are support material, not a single production integration. For Egocentric Co-Pilot integration, use them as templates for exposing external tools to the orchestration layer through MCP, then adapt tool names, transport URLs, and device assumptions to the target deployment.

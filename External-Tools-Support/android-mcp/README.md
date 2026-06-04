# Android MCP

Curated copy of the Android MCP server from `Huawei/Projects/MCP/Mobile/Android-MCP`.

## Tools

- `State-Tool`: list visible, enabled interactive UI elements and center coordinates.
- `Click-Tool`: tap at screen coordinates.
- `Long-Click-Tool`: long press at screen coordinates.
- `Swipe-Tool`: swipe between coordinates.
- `Type-Tool`: type text through the Android input method.
- `Drag-Tool`: drag between coordinates.
- `Press-Tool`: press Android buttons such as back or volume keys.
- `Notification-Tool`: open notifications.
- `Wait-Tool`: sleep for a specified duration.
- `Shell-Tool`: execute an Android shell command through the connected device.

## Run

```bash
uv sync
uv run main.py --emulator
```

Use `--emulator` for `emulator-5554`; omit it for the default connected Android device.

## Source

The upstream README is preserved as `README-upstream.md`, and the upstream license is preserved in `LICENSE`.

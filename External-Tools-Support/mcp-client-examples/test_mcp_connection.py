import asyncio
from fastmcp import Client
from fastmcp.client.transports import SSETransport

async def example():
    async with Client(transport=SSETransport("http://localhost:6001/mcp")) as client:
    # await client.ping()
        tools = await client.list_tools()
        print(f"Available tools: {tools}")
        # devices = await client.call_tool("mobile_list_available_devices", {
        #     "noParams": {}
        # })
        # print("Available devices:", devices)


        # # Optional: Check installed apps
        # apps = await client.call_tool("mobile_list_apps", {
        #     "noParams": {}
        # })
        # print("Installed apps:", apps)

        # # Step 2: Launch YouTube
        # result = await client.call_tool("mobile_launch_app", {
        #     "packageName": "com.google.android.youtube"
        # })
        # print("Launch result:", result)
        # devices = await client.call_tool("mobile_list_available_devices", {
        #     "noParams": {}
        # })
        # print("Available devices:", devices)

        # await client.call_tool("mobile_use_device", {
        #     "deviceType": "android",
        #     "device": "emulator-5554"  # replace if your emulator has a different ID
        # })

        # ✅ Now launch YouTube
        # result = await client.call_tool("mobile_launch_app", {
        #     "packageName": "com.google.android.youtube"
        # })

        # print("Launch result:", result)




if __name__ == "__main__":
    import os
    # print(os.environ["PATH"])

    asyncio.run(example())
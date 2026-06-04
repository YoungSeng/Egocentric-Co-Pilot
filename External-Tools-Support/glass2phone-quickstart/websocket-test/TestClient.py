import asyncio
import websockets

# !!! IMPORTANT !!!
# Change this URL to the one displayed on your Android app's screen.
SERVER_URL = "ws://10.0.2.16:8765/echo"

async def test_client():
    # The 'async with' statement handles connecting and closing automatically
    async with websockets.connect(SERVER_URL) as websocket:
        print(f"Connected to server at {SERVER_URL}")
        
        # First, let's wait for any initial message from the server (if any)
        # and also send our first message.
        
        # Send a message
        first_message = "Hello from Python!"
        await websocket.send(first_message)
        print(f"> Sent: {first_message}")
        
        # Receive the response
        response = await websocket.recv()
        print(f"< Received: {response}")

        # Now, let's enter a loop to chat with the server
        print("\n--- Enter a message to send (or type 'exit' to quit) ---")
        while True:
            message_to_send = await asyncio.to_thread(input, "> ")
            if message_to_send.lower() == 'exit':
                break
            
            await websocket.send(message_to_send)
            response = await websocket.recv()
            print(f"< Received: {response}")

if __name__ == "__main__":
    try:
        asyncio.run(test_client())
    except websockets.exceptions.ConnectionClosedError:
        print("Connection to server closed.")
    except ConnectionRefusedError:
        print("Connection refused. Is the server running on the phone?")
    except KeyboardInterrupt:
        print("\nClient stopped.")
import asyncio
import websockets
import numpy as np
import wave
import datetime

# --- Configuration ---
# These settings must match the ones in your Android app
HOST = "0.0.0.0"  # Listen on all available network interfaces
PORT = 8765
RECORDER_SAMPLERATE = 44100
RECORDER_CHANNELS = 1  # The Android app code uses MONO
RECORDER_AUDIO_ENCODING_BITS = 16
# Corresponding numpy data type for 16-bit PCM audio
NUMPY_DTYPE = np.int16

# This will keep track of how many packets we've received
packet_count = 0

async def audio_handler(websocket, path):
    """
    Handles a single WebSocket connection for audio streaming.
    """
    global packet_count
    print(f"Client connected from {websocket.remote_address}")
    
    # Generate a unique filename for each recording session
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"received_audio_{timestamp}.wav"
    
    # Setup the .wav file writer
    # The 'wave' module is part of Python's standard library
    with wave.open(output_filename, 'wb') as wav_file:
        wav_file.setnchannels(RECORDER_CHANNELS)
        wav_file.setsampwidth(RECORDER_AUDIO_ENCODING_BITS // 8) # 16 bits = 2 bytes
        wav_file.setframerate(RECORDER_SAMPLERATE)
        
        try:
            # The main loop that receives audio data
            async for message in websocket:
                # The Android app sends raw bytes. We write them to the wav file.
                wav_file.writeframes(message)
                
                # --- Real-time analysis (optional but cool) ---
                # Convert the raw bytes to a numpy array for analysis
                audio_data = np.frombuffer(message, dtype=NUMPY_DTYPE)
                
                # Calculate the volume (Root Mean Square)
                rms_volume = np.sqrt(np.mean(audio_data.astype(float)**2))
                
                print(f"Received audio packet: {len(message)} bytes, RMS Volume: {rms_volume:.2f}")
                
                # --- Send a message back to the client occasionally ---
                packet_count += 1
                if packet_count % 50 == 0: # Every 50 packets
                    response = f"Server received {packet_count} packets."
                    print(f"Sending response to client: {response}")
                    await websocket.send(response)

        except websockets.exceptions.ConnectionClosed as e:
            print(f"Connection closed by client: {e.code} {e.reason}")
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            print(f"Audio stream saved to {output_filename}")
            print(f"Client {websocket.remote_address} disconnected.")


async def main():
    """
    Starts the WebSocket server.
    """
    # The 'with' statement ensures the server is properly shut down
    async with websockets.serve(audio_handler, HOST, PORT):
        print(f"WebSocket server started at ws://{HOST}:{PORT}")
        await asyncio.Future()  # Keep the server running indefinitely

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Server shutting down.")
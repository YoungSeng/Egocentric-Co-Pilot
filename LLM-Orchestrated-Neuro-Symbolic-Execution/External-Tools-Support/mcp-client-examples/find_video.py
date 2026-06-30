import asyncio
import re
from typing import List, Dict, Optional

from fastmcp import Client
from fastmcp.client.transports import SSETransport

# --- Helper Functions (No changes needed here) ---

def find_element_by_label(elements: List[Dict], label_text: str, exact_match: bool = False) -> Optional[Dict]:
    """
    Finds an element in the list of UI elements by its accessibility label or text.
    """
    label_text_lower = label_text.lower()
    for element in elements:
        # In the new agent, coordinates are nested
        coords = element.get('coordinates', {})
        element_data = {
            'text': element.get('text'),
            'accessibility_label': element.get('label'), # The label key is 'label'
            'x': coords.get('x'),
            'y': coords.get('y')
        }

        text = element_data.get('text') or element_data.get('accessibility_label')
        if not text:
            continue
            
        # Check for coordinates, as they are essential for clicking
        if element_data['x'] is None or element_data['y'] is None:
            continue

        if exact_match:
            if text == label_text:
                return element_data
        else:
            if label_text_lower in text.lower():
                return element_data
    return None

def find_video_with_million_views(elements: List[Dict], title_keyword: str) -> Optional[Dict]:
    """
    Finds the first video element containing the title keyword and over 1M views.
    """
    view_pattern = re.compile(r'(\d+(\.\d+)?)M views')
    keyword_lower = title_keyword.lower()
    for element in elements:
        # Adapt to the new agent's element structure
        coords = element.get('coordinates', {})
        element_data = {
            'text': element.get('text'),
            'accessibility_label': element.get('label'),
            'x': coords.get('x'),
            'y': coords.get('y')
        }
        
        element_text = element_data.get('text') or element_data.get('accessibility_label')
        if not element_text:
            continue

        # Check for coordinates
        if element_data['x'] is None or element_data['y'] is None:
            continue

        if keyword_lower in element_text.lower() and view_pattern.search(element_text):
            print(f"  -> Found potential video: '{element_text}'")
            return element_data
    print(f"  -> No video matching '{title_keyword}' with 1M+ views found on current screen.")
    return None

# --- Main Asynchronous Function (With Corrections) ---

async def test_youtube_scenario():
    """
    Main function to execute the test case using the UI-based search click.
    """
    async with Client(transport=SSETransport("http://localhost:6001/mcp")) as client:
        print("--- Starting YouTube Test Scenario ---")
        
        device_id = "emulator-5554" 
        print(f"\n[Step 1] Selecting device: {device_id}")
        await client.call_tool("mobile_use_device", {"deviceType": "android", "device": device_id})
        print("  -> Device selected successfully.")

        youtube_package_name = "com.google.android.youtube"
        print(f"\n[Step 2] Launching YouTube (package: {youtube_package_name})")
        await client.call_tool("mobile_launch_app", {"packageName": youtube_package_name})
        print("  -> YouTube launched. Waiting for UI to load...")
        await asyncio.sleep(5)

        print("\n[Step 3] Sending ADB keyevent to open search directly.")
        await client.call_tool("mobile_adb_keyevent", {"keyCode": "84"})
        print("  -> Sent keyevent 84 (search). Waiting for search input field...")
        await asyncio.sleep(2)

        search_query = "Linux tech tips"
        print(f"\n[Step 4] Typing search query: '{search_query}'")
        await client.call_tool("mobile_type_keys", {"text": search_query, "submit": True})
        print("  -> Search submitted. Waiting for results to load...")
        await asyncio.sleep(5)

        print("\n[Step 5] Searching for a video with 'Linux tech tips' in title and >1M views.")
        target_video = None
        max_scrolls = 3

        for i in range(max_scrolls):
            print(f"  -> Analyzing screen content (Attempt {i + 1}/{max_scrolls})...")
            search_results = await client.call_tool("mobile_list_elements_on_screen", {"noParams": {}})
            
            target_video = find_video_with_million_views(search_results.result, "Linux tech tips")
            
            if target_video:
                print("  -> SUCCESS: Target video found on screen!")
                break
            
            if i < max_scrolls - 1:
                print("  -> Video not found, scrolling down...")
                # Note: The swipe tool in the agent is `swipe_on_screen`
                await client.call_tool("swipe_on_screen", {"direction": "down"})
                await asyncio.sleep(2)
        
        if target_video:
            video_text = target_video.get('text') or target_video.get('accessibility_label')
            print(f"\n[Step 6] Clicking on target video: '{video_text}'")
            await client.call_tool("mobile_click_on_screen_at_coordinates", {"x": target_video['x'], "y": target_video['y']})
            print("\n--- ✅ Test Scenario Completed Successfully! ---")
        else:
            print("\n--- ❌ Test Scenario Failed: Could not find a matching video after multiple scrolls. ---")


if __name__ == "__main__":
    try:
        asyncio.run(test_youtube_scenario())
    except Exception as e:
        print(f"An error occurred: {e}")
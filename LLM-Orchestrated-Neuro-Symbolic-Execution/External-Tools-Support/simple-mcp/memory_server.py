import json
from typing import List
from fastmcp import FastMCP
from datetime import datetime
from typing import Optional
from utils.memory_fuc import classify_time_label, is_similar






mcp = FastMCP("memory_server", host="0.0.0.0")




MEMORY_FILE = "memory_store.json"
# === Utility Functions ===

def load_memory() -> List[dict]:
    try:
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def save_memory(memory: List[dict]):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)


# === Memory Tools ===

# @mcp.tool()
# async def create_memory(event: str, people: str, observation: str, when: str = None) -> str:
#     """
#     Store an observation related to a specific event and person.

#     Args:
#         event: The name of the event (e.g., 'meeting', 'project_start').
#         people: The person related to this memory (e.g., 'John_Smith').
#         observation: The content of the memory (e.g., 'Prefers morning meetings').
#     """

#     if not when:
#         when = datetime.now()


#     memory = load_memory()

#     # Check for similar observation under same (people, event)
#     for m in memory:
#         # if m["people"] == people and m["event"] == event:
#         if m["people"] == people:
#             if is_similar(m["observation"], observation):
#                 return f"⚠️ Similar memory already exists for '{people}' and '{event}', not saved."


#     memory.append({
#         "event": event,
#         "people": people,
#         "observation": observation,
#         "when": when,
#     })
#     save_memory(memory)
#     return f"✅ Memory saved for '{people}' about '{event}' at '{when}'"
@mcp.tool()
async def create_memory(
    event: str,
    people: str,
    observation: str,
    date: Optional[str] = None,
    time: Optional[str] = None,
    time_label: Optional[str] = None
) -> str:
    """
    Store an observation related to a specific event and person.

    Args:
        event: The name of the event (e.g., 'meeting', 'project_start').
        people: The person related to this memory (e.g., 'John_Smith').
        observation: The content of the memory.
        date: Date in YYYY-MM-DD format.
        time: Time in HH:MM:SSZ format.
        time_label: Semantic label (e.g., morning, afternoon). Will be inferred if not provided.
    """

    # Auto-fill current date/time if date/time is missing
    if not date or not time:
        now = datetime.utcnow()
        date = now.strftime("%Y-%m-%d") if not date else date
        time = now.strftime("%H:%M:%SZ") if not time else time
        if not time_label:
            time_label = classify_time_label(now.hour)
    elif not time_label and time:
        try:
            hour = int(time[:2])
            time_label = classify_time_label(hour)
        except:
            time_label = None  # fallback

    memory = load_memory()

    for m in memory:
        if m["people"] == people and is_similar(m["observation"], observation) and  m["date"] == (date or None):
            return f"⚠️ Similar memory already exists for '{people}' and '{event}', not saved."

    memory.append({
        "event": event,
        "people": people,
        "observation": observation,
        "date": date,
        "time": time,
        "time_label": time_label
    })

    save_memory(memory)
    return f"✅ Memory saved for '{people}' about '{event}' at {date} {time or ''}"



@mcp.tool()
async def read_all_memory() -> List[dict]:
    """
    Read and return all stored memory entries.
    """
    memory = load_memory()
    memory.sort(key=lambda m: ((m.get("date") or ""), (m.get("time") or "")))
    return memory

@mcp.tool()
async def delete_memory(event: str, people: str) -> str:
    """
    Delete a specific memory entry by event and person.

    Args:
        event: The name of the event (e.g., 'project_kickoff').
        people: The person involved in the memory (e.g., 'Alice').
    """
    memory = load_memory()
    original_len = len(memory)
    memory = [m for m in memory if not (m["event"] == event and m["people"] == people)]
    save_memory(memory)

    if len(memory) == original_len:
        return f"⚠️ No memory found for '{people}' about '{event}'"
    return f"🗑️ Deleted memory for '{people}' about '{event}'"




if __name__ == "__main__":
    mcp.run(
        transport="sse", 
        host="127.0.0.1",
        port=4201
    )


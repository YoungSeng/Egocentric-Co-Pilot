# utils/local_models.py

import asyncio
import logging
import os
import wave
import base64
import numpy as np
import cv2
import re
import json  # For Qwen MCP agent tool calls
import torch

# Existing model imports (Whisper, F5, SentenceTransformer, Mistral)
# ...

# New imports for Qwen and MCP
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastmcp import Client
from fastmcp.client.transports import SSETransport
from datetime import datetime
## Multiserver handler
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools



logger = logging.getLogger(__name__)



# --- New Global Model Variables for Qwen ---
AGENT_MODEL = None
AGENT_TOKENIZER = None
MODEL_DEVICE = None


# persistent MCP client and its tools
MCP_CLIENT = None
ALL_MCP_TOOLS = {}


MULTISERVER_CONFIGS = {
    "math": {
        "command": "python",
        "args": [str("servers/math_server.py")],
        "transport": "stdio",
    },
    "mobile": {
        "url": "mobile_mcp_url/mcp",
        "transport": "sse",
    },
    "memory": {
        "url": "localhost:4201/sse",
        "transport": "sse",
    }
}
MCP_AGENT_MODEL = "Qwen/Qwen3-8B"

MCP_SYSTEM_PROMPT = "multiserver_prompt.txt"


# --- New Load Function for Agent model (Qwen) ---
def load_agent_model_and_tokenizer():
    global AGENT_MODEL, AGENT_TOKENIZER, MODEL_DEVICE
    logger.info("🔥 load_agent_model_and_tokenizer() was called.")
    # logger.debug("Called load_agent_model_and_tokenizer()")
    # logger.debug(f"Model path: {config.MCP_AGENT_MODEL}")

    if AGENT_MODEL is not None and AGENT_TOKENIZER is not None:
        logger.info(f"Qwen model '{MCP_AGENT_MODEL}' already loaded.")
        return

    if not MCP_AGENT_MODEL:
        logger.error("QWEN_MODEL_NAME not set in config. Cannot load Qwen model.")
        return

    try:
        logger.info(f"Loading Qwen model and tokenizer: {MCP_AGENT_MODEL}...")
        AGENT_TOKENIZER = AutoTokenizer.from_pretrained(
            MCP_AGENT_MODEL)
        
        AGENT_MODEL = AutoModelForCausalLM.from_pretrained(
            MCP_AGENT_MODEL,
            torch_dtype="auto",  # Uses bfloat16 if available, float16 otherwise
            device_map="auto",  # Automatically uses CUDA if available and accelerate is installed
        )

        # Determine the device the model was loaded onto
        if hasattr(AGENT_MODEL, "device"):
            MODEL_DEVICE = AGENT_MODEL.device
        elif torch.cuda.is_available():
            MODEL_DEVICE = torch.device("cuda")
            # If device_map="auto" worked, model is already on device(s).
            # If not, this explicit .to(MODEL_DEVICE) might be needed for single-GPU or if device_map failed.
            # However, with device_map="auto", model parts can be on different devices.
            # For simplicity, we assume 'auto' places it correctly or AGENT_MODEL.device is indicative.
        else:
            MODEL_DEVICE = torch.device("cpu")
            # AGENT_MODEL = AGENT_MODEL.to(MODEL_DEVICE) # Ensure it's on CPU if no CUDA and device_map didn't specify

        # Handle pad_token for generation if not set (common for some Qwen models)
        if AGENT_TOKENIZER.pad_token_id is None:
            if AGENT_TOKENIZER.eos_token_id is not None:
                AGENT_TOKENIZER.pad_token_id = AGENT_TOKENIZER.eos_token_id
                logger.info(
                    f"Qwen tokenizer pad_token_id set to eos_token_id ({AGENT_TOKENIZER.eos_token_id})"
                )
            else:
                # Add a dummy pad token if no eos_token_id either (less ideal)
                AGENT_TOKENIZER.add_special_tokens({"pad_token": "[PAD]"})
                AGENT_MODEL.resize_token_embeddings(len(AGENT_TOKENIZER))
                logger.warning(
                    "Qwen tokenizer pad_token_id and eos_token_id were None. Added a new pad_token."
                )

        logger.info(
            f"Agent model '{MCP_AGENT_MODEL}' loaded successfully on device(s) via device_map (primary device likely: {MODEL_DEVICE})."
        )

    except Exception as e:
        logger.error(
            f"Failed to load Qwen model ('{MCP_AGENT_MODEL}') or tokenizer: {e}",
            exc_info=True,
        )
        AGENT_MODEL = None
        AGENT_TOKENIZER = None
        MODEL_DEVICE = None
    logger.info(f"✅ Agent model loaded on device: {MODEL_DEVICE}")



# --- New function to initialize the client once at startup ---
async def initialize_mcp_client_and_tools():
    """
    Initializes the MultiServerMCPClient and loads its tools.
    This should be called only once when the server starts.
    """
    global MCP_CLIENT, ALL_MCP_TOOLS

    if MCP_CLIENT:
        logger.info("MCP client and tools are already initialized.")
        return

    logger.info("Initializing MultiServerMCPClient and loading tools for the first time...")
    try:
        MCP_CLIENT = MultiServerMCPClient(MULTISERVER_CONFIGS)
        tools_list = await MCP_CLIENT.get_tools()
        ALL_MCP_TOOLS = {tool.name: tool for tool in tools_list}
        logger.info(f"✅ Successfully initialized MCP client and loaded {len(ALL_MCP_TOOLS)} tools.")
    except Exception as e:
        logger.error(f"Fatal error: Failed to initialize MCP client or load tools: {e}", exc_info=True)
        MCP_CLIENT = None
        ALL_MCP_TOOLS = {}




async def process_with_multiserver_mcp(user_input_text: str, detected_language: str = "en") -> str:
    """
    Uses Qwen and a GLOBAL MultiServerMCPClient to handle tool-augmented reasoning.
    """
    # 1. Check if global resources are ready
    if not AGENT_MODEL or not MCP_CLIENT or not ALL_MCP_TOOLS:
        logger.error("Agent model or MCP client is not initialized. Cannot process request.")
        return "The assistant is not ready. Please wait a moment and try again."

    try:
        # 2. Use the globally loaded tools. No need to fetch them again.
        tool_map = ALL_MCP_TOOLS
        logger.info(f"Using {len(tool_map)} pre-loaded tools.")

        # 3. Prepare tool schemas for the LLM
        available_functions = []
        for tool in tool_map.values():
            schema_dict = tool.args_schema
            function_def = {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": schema_dict.get("properties", {}),
                },
            }
            required = schema_dict.get("required", [])
            if required:
                function_def["parameters"]["required"] = required
            available_functions.append({"type": "function", "function": function_def})

        # 4. Construct prompt (this part is the same)
        with open(MCP_SYSTEM_PROMPT, "r", encoding="utf-8") as f:
            system_prompt = f.read().strip()
        messages = [
            {"role": "system", "content": system_prompt},
            # {"role": "user", "content": f"I am using Android which id is 38311FDJG00GQT, {user_input_text}"}
            {"role": "user", "content": f"{user_input_text}"}
        ]

        # 5. Generate LLM response in a thread (this part is the same)
        def _generate_response():
            text_prompt = AGENT_TOKENIZER.apply_chat_template(
                messages, tools=available_functions, tokenize=False
            )
            inputs = AGENT_TOKENIZER(text_prompt, return_tensors="pt").to(MODEL_DEVICE)
            gen_kwargs = {"max_new_tokens": 512, "pad_token_id": AGENT_TOKENIZER.pad_token_id}
            outputs = AGENT_MODEL.generate(**inputs, **gen_kwargs)
            generated = outputs[0][inputs.input_ids.shape[1]:]
            return AGENT_TOKENIZER.decode(generated, skip_special_tokens=True)

        model_output = await asyncio.to_thread(_generate_response)
        logger.info(f"Agent output:\n{model_output}")

        # 6. Parse tool calls (this part is the same)
        tool_calls = []
        for match in re.finditer(r"<tool_call>(.*?)</tool_call>", model_output, re.DOTALL):
            payload = match.group(1).strip()
            try:
                # Add robust JSON parsing
                parsed = json.loads(payload)
                if "name" in parsed and "arguments" in parsed:
                    tool_calls.append(parsed)
                    logger.info(f"Parsed tool call: {parsed}")
            except json.JSONDecodeError:
                logger.warning(f"Could not parse tool call JSON: {payload}")

        if not tool_calls:
            clean_output = re.sub(r"</?think>", "", model_output).strip()
            return clean_output or "I'm ready. What would you like me to do next?"

        # 7. Execute tool calls (this part is the same, but now more robust)
        results = []
        TIMEOUT_SECONDS = 15
        for call in tool_calls:
            tool_name = call["name"]
            tool_args = call.get("arguments", {}) # Use .get for safety
            try:
                if tool_name in tool_map:
                    logger.info(f"Calling tool: {tool_name}: ~~~~~~")
                    result = await asyncio.wait_for(tool_map[tool_name].ainvoke(tool_args), timeout=TIMEOUT_SECONDS)
                    logger.info(f"Tool {tool_name} executed successfully: {result}")
                    results.append(f"{tool_name}: {result}")
                else:
                    logger.error(f"Tool '{tool_name}' not found in loaded tools.")
                    results.append(f"{tool_name}: error - tool not found")
            except asyncio.TimeoutError:
                logger.error(f"TimeoutError: Tool '{tool_name}' took too long to respond.")
                results.append(f"{tool_name}: error - timeout")
            except Exception as e:
                logger.error(f"Failed to invoke {tool_name}: {e}", exc_info=True)
                results.append(f"{tool_name}: error - {str(e)}")
        
        # This function will now correctly return and unblock the waiting task
        return "\n".join(results)

    except Exception as e:
        logger.error(f"Critical error in MCP Agent processing loop: {e}", exc_info=True)
        return f"I encountered a major issue while processing your request."










if __name__ == "__main__":
    import logging
    import sys
    import os

    # Add project root so we can import config.py
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


    logging.basicConfig(level=logging.INFO)

    print("🔧 Running Qwen MCP Agent model loader test...")

    print(f"📦 Attempting to load model from: {MCP_AGENT_MODEL}")

    try:
        load_agent_model_and_tokenizer()

        if AGENT_MODEL is not None and AGENT_TOKENIZER is not None:
            print("✅ Qwen model and tokenizer loaded successfully.")
            print(f"🧠 Model device: {MODEL_DEVICE}")
        else:
            print("❌ Model or tokenizer not loaded. Check logs or config.")

    except Exception as e:
        print(f"❌ Exception during model loading: {e}")
        import traceback
        traceback.print_exc()
import os

# --- Model Paths (Default) ---
# Set these to your local paths or HuggingFace IDs
DEFAULT_VLM_MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"  # Can be a local path
DEFAULT_LLM_MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"     # Can be a local path
DEFAULT_ASR_MODEL_PATH = "openai/whisper-large-v3"

# --- Data Paths (Default) ---
DEFAULT_VIDEO_ROOT = "./datasets/videos/"
DEFAULT_OUTPUT_DIR = "./outputs/"
DEFAULT_ASR_OUTPUT_DIR = "./outputs/transcripts/"

# Ensure output directories exist
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
os.makedirs(DEFAULT_ASR_OUTPUT_DIR, exist_ok=True)
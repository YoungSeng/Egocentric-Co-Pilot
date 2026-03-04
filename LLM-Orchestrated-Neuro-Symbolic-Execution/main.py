"""
Open Source Refactored: Simple websocket based audio and image transfer.
Features:
- WebSocket server for Image/Audio streaming.
- Integration with ASR (Whisper), TTS (F5-TTS), and VLM (Qwen-VL).
- Chinese Chess analysis logic.
"""

import os
import sys
import argparse
import asyncio
import base64
import collections
import datetime
import io
import json
import logging
import pdb
import queue
import random
import re
import subprocess
import time
import wave
from concurrent.futures import ThreadPoolExecutor

# Third-party imports
import cv2
import numpy as np
import requests
import tomli
import torch
import websockets
import soundfile as sf
from PIL import Image
from omegaconf import OmegaConf
from cached_path import cached_path
from scipy.spatial.distance import cdist

# --- Local Imports (Assuming these files are in the same directory) ---
try:
    from chess_engine import Chinese_Chessboard_New
    from llm_intent import LLMInferenceUnified
    from api_providers import siliconflow_API, openai_API
except ImportError as e:
    print(f"Warning: Local modules not found ({e}). Ensure 'mycode_Chinese_chess_API.py', 'refined_LLM.py', etc. are present.")

# --- Directory Setup ---
# Base directory is the directory containing this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
RECEIVED_AUDIO_DIR = os.path.join(BASE_DIR, "received_audio")
RECEIVED_FRAMES_DIR = os.path.join(BASE_DIR, "received_frames")

# Ensure directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RECEIVED_AUDIO_DIR, exist_ok=True)
os.makedirs(RECEIVED_FRAMES_DIR, exist_ok=True)

# Add F5-TTS source to sys.path dynamically
F5_TTS_PATH = os.path.join(BASE_DIR, "F5-TTS", "src")
if os.path.exists(F5_TTS_PATH):
    sys.path.append(F5_TTS_PATH)
    try:
        from f5_tts.model import DiT, UNetT
        from f5_tts.infer.utils_infer import (
            mel_spec_type, target_rms, cross_fade_duration, nfe_step,
            cfg_strength, sway_sampling_coef, speed, fix_duration,
            infer_process, load_model, load_vocoder,
            preprocess_ref_audio_text, remove_silence_for_generated_wav,
        )
    except ImportError:
        print("Error: F5-TTS modules could not be imported. Check submodule installation.")
else:
    print(f"Warning: F5-TTS path not found at {F5_TTS_PATH}")

# --- Global Configuration ---
CONFIG = {
    "USE_REMOTE_SERVER": False,  # Set to True to use remote inference
    "SERVER_IP": "10.206.165.151",
    "SERVER_PORT": "5002",
    "WEBSOCKET_PORT": 5000,
    "MODELS": {
        "INTENTION": "Qwen/Qwen2.5-0.5B-Instruct",
        "VLM": "Qwen/Qwen2-VL-2B-Instruct",
        "ASR": "large",
        "TTS_CKPT": os.path.join(BASE_DIR, "pre-trained-models", "model_1200000.safetensors"),
        "TTS_CONFIG": os.path.join(BASE_DIR, "F5-TTS", "src", "f5_tts", "infer", "examples", "basic", "basic.toml"),
        "TTS_MODEL_CFG": os.path.join(BASE_DIR, "F5-TTS", "src", "f5_tts", "configs", "F5TTS_Base_train.yaml"),
        "REF_AUDIO": os.path.join(BASE_DIR, "F5-TTS", "src", "f5_tts", "infer", "examples", "basic", "basic_ref_en.wav"),
        # Assuming checkpoints are one level up from the script based on original "../checkpoints" logic
        "VOCODER_DIR": os.path.abspath(os.path.join(BASE_DIR, "..", "checkpoints")),
    }
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Global Variables ---
Using_remote_server = CONFIG["USE_REMOTE_SERVER"]
SERVER_IP = CONFIG["SERVER_IP"]
SERVER_PORT = CONFIG["SERVER_PORT"]

latest_frame = None
audio_buffer = []
HEADER_SIZE = 4
Max_New_Tokens = 50
visualization_mode = False

# --- Visualization Setup ---
if visualization_mode:
    cv2.namedWindow('Received Frame', cv2.WINDOW_NORMAL)

# --- Model Loading Logic ---

loaded_tts_model = None
loaded_vocoder = None
tts_config = None
tts_args = None
text_based_llm = None
ASR_model = None

def load_local_models():
    """Loads all local AI models if remote server is not used."""
    global ASR_model, loaded_tts_model, loaded_vocoder, tts_config, tts_args, text_based_llm

    # 1. Load ASR (Whisper)
    import whisper
    logger.info(f"Loading local ASR model: whisper-{CONFIG['MODELS']['ASR']}")
    ASR_model = whisper.load_model(CONFIG['MODELS']['ASR'])
    logger.info("ASR model loaded.")

    # 2. Load TTS (F5-TTS)
    def load_tts_models(config_path, model_name, model_cfg_path, ckpt_file, vocab_file, vocoder_name, load_vocoder_from_local):
        config = tomli.load(open(config_path, "rb"))

        # Resolve vocoder path relative to configured directory
        if vocoder_name == "vocos":
            vocoder_local_path = os.path.join(CONFIG["MODELS"]["VOCODER_DIR"], "vocos-mel-24khz")
        elif vocoder_name == "bigvgan":
            vocoder_local_path = os.path.join(CONFIG["MODELS"]["VOCODER_DIR"], "bigvgan_v2_24khz_100band_256x")
        else:
            raise ValueError(f"Unsupported vocoder name: {vocoder_name}")

        logger.info(f"Loading vocoder: {vocoder_name} from {vocoder_local_path}...")
        try:
            loaded_vocoder_inst = load_vocoder(vocoder_name=vocoder_name, is_local=load_vocoder_from_local, local_path=vocoder_local_path)
        except Exception as e:
             logger.warning(f"Could not load local vocoder from {vocoder_local_path}, trying default or download. Error: {e}")
             loaded_vocoder_inst = load_vocoder(vocoder_name=vocoder_name, is_local=False) # Fallback

        logger.info("Vocoder loaded.")

        if model_name == "F5-TTS":
            model_cls = DiT
            model_cfg = OmegaConf.load(model_cfg_path).model.arch
            if not ckpt_file or not os.path.exists(ckpt_file):
                logger.info("Checkpoint file not found locally, downloading from HF...")
                repo_name = "F5-TTS"
                exp_name = "F5TTS_Base"
                ckpt_step = 1200000
                ckpt_file = str(cached_path(f"hf://SWivid/{repo_name}/{exp_name}/model_{ckpt_step}.safetensors"))
        elif model_name == "E2-TTS":
            model_cls = UNetT
            model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4)
            if not ckpt_file:
                repo_name = "E2-TTS"
                exp_name = "E2TTS_Base"
                ckpt_step = 1200000
                ckpt_file = str(cached_path(f"hf://SWivid/{repo_name}/{exp_name}/model_{ckpt_step}.safetensors"))
        else:
            raise ValueError(f"Unsupported model name: {model_name}")

        logger.info(f"Loading TTS model {model_name} from {ckpt_file}...")
        loaded_model_inst = load_model(model_cls, model_cfg, ckpt_file, mel_spec_type=vocoder_name, vocab_file=vocab_file)
        logger.info("TTS model loaded.")
        return loaded_model_inst, loaded_vocoder_inst

    # Setup TTS Arguments
    tts_args = argparse.Namespace()
    tts_args.model = "F5-TTS"
    tts_args.ref_audio = CONFIG["MODELS"]["REF_AUDIO"]
    tts_args.ref_text = "Some call me nature, others call me mother nature."
    tts_args.load_vocoder_from_local = False # Set to True if you have the files
    tts_args.vocoder_name = 'vocos'
    tts_args.vocab_file = ""
    tts_args.ckpt_file = CONFIG["MODELS"]["TTS_CKPT"]
    tts_args.model_cfg = CONFIG["MODELS"]["TTS_MODEL_CFG"]
    tts_args.config = CONFIG["MODELS"]["TTS_CONFIG"]
    tts_args.save_chunk = False
    tts_args.remove_silence = False

    loaded_tts_model, loaded_vocoder = load_tts_models(
        config_path=tts_args.config,
        model_name=tts_args.model,
        model_cfg_path=tts_args.model_cfg,
        ckpt_file=tts_args.ckpt_file,
        vocab_file=tts_args.vocab_file,
        vocoder_name=tts_args.vocoder_name,
        load_vocoder_from_local=tts_args.load_vocoder_from_local
    )
    tts_config = tomli.load(open(tts_args.config, "rb"))

    # 3. Load Intention Model (Text-based LLM)
    logger.info(f"Loading local Intention model: {CONFIG['MODELS']['INTENTION']}")
    text_based_llm = LLMInferenceUnified(model_name=CONFIG['MODELS']['INTENTION'])
    logger.info("Intention model loaded.")

if not Using_remote_server:
    load_local_models()

# --- Intent Templates ---
INTENT_TEMPLATES = {
    "object_recognition": [
        "Please describe what the finger is pointing to.",
        "Please tell me the details of the object the finger is pointing to.",
        "Please identify the object the finger is pointing at.",
        "What is this?"
    ],
    "color_recognition": [
        "Please describe the color of the object the finger is pointing to.",
        "Please tell me the color of the object the finger is pointing at.",
        "Please identify the color of the object the finger is pointing to.",
        "What color is this?"
    ],
    "scene_description": [
        "Please describe the contents of the current view.",
        "Please tell me what the scene you see is.",
        "Please analyze the environment in your current view.",
        "What can you see in this image?"
    ],
    "feature_explanation": [
        "Please explain how the function the finger is pointing to works.",
        "Please provide detailed information about the function the finger is pointing to.",
        "Please describe how the function the finger is pointing to operates."
    ],
    "image_analysis": [
        "Please analyze the content of the image in your current view.",
        "Please describe the details of the current image.",
        "Please identify the main elements in the current image."
    ],
    "play_Chinesee_chess": [
        "What should be my next move?",
        "How should I move next?",
        "Please suggest my next move.",
        "Help me see, which move is better now?",
        "What is the best next move?",
        "Which move do you recommend?",
        "Do you have any suggestions?",
        "How should I move next for a better position?",
        "Give the best three next moves.",
        "What is the main strategy?",
        "What move do you recommend?"
    ],
    "memorization_required": [
        "Where is the... I forgot.",
        "I forgot.",
        "Where did I just put the cup?",
        "Where did I put my phone?",
        "Where are the keys?",
        "Where did I put the pen?",
        "Test the memory function.",
        "Please retrieve from memory...",
        "Where did I just put my...",
        "Where is my...?",
        "Help me remember...",
        "I forgot... tell me where it is.",
        "Where did I put my Chinese chessboard?",
        "Where is my green water bottle?",
        "Where is my red pen?"
    ],
    "high_frame_rate_mode": [
        "Enter high frame rate mode.",
        "Enter game mode.",
        "Remember the positions of these three pieces.",
        "Remember the piece positions.",
        "Remember the current Chinese chess piece positions."
    ],
    "low_frame_rate_mode_withinference": [
        "What piece is at this position?",
        "Where is the red...?",
        "Where is the black...?",
        "Where is the...?",
        "What are the pieces at these positions?",
        "What are the three pieces' positions now?",
        "What are the pieces from left to right?",
        "From left to right, what are they?",
        "Is this a red/black Chariot/Cannon?",
        "What color is this piece?",
        "Is this a Red Elephant?",
        "Is this a Black Horse?",
        "Is this a Black Elephant?"
    ],
    "low_frame_rate_mode_woinference": [
        "Exit high frame rate mode.",
        "Exit game mode.",
        "Have you remembered the piece positions?",
        "I've finished moving.",
        "I have finished moving."
    ],
    "continuous_guidance_mode": [
        "Enter continuous guidance mode.",
        "Continuous chess guidance mode.",
        "Please guide me continuously in chess.",
        "Can you teach me to play Chinese chess?"
    ],
    "cancel_continuous_guidance_mode": [
        "Exit continuous guidance mode.",
        "No need for continuous chess guidance.",
        "Stop guiding me in chess.",
        "Stop teaching me chess."
    ]
}

# --- Utility Functions ---

def mean_pooling(model_output, attention_mask):
    """Mean Pooling - Take attention mask into account for correct averaging."""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class CustomQueue:
    """A custom queue with a fixed size that discards the oldest element when full."""
    def __init__(self, maxsize):
        if maxsize <= 0:
            raise ValueError("maxsize must be positive")
        self.maxsize = maxsize
        self.queue = collections.deque(maxlen=maxsize)

    def put(self, item):
        self.queue.append(item)

    def get_latest(self):
        try:
            return self.queue[-1]
        except IndexError:
            raise IndexError("Queue is empty")

    def get_all(self, timestamp_tracking=None):
        all_items = list(self.queue)
        if not timestamp_tracking:
            memory_length = 190
            tmp_memory = [item for item in all_items if isinstance(item, tuple) and len(item) >= 3]
            logger.info(f"Full memory length: {len(tmp_memory)}, returning last {memory_length} items.")
            return tmp_memory[-memory_length:]
        else:
            frame_list = []
            logger.info(f"Timestamp tracking enabled, processing frames at or after {timestamp_tracking}.")
            for item in all_items:
                if isinstance(item, tuple) and len(item) >= 3:
                    item_timestamp = item[2]
                    if item_timestamp >= timestamp_tracking:
                        frame_list.append(item)
            if not frame_list:
                logger.warning("No valid frames in queue after timestamp filtering.")
            return frame_list[-40:]

    def empty(self):
        return not self.queue

    def to_mp4(self, output_filename="output.mp4", fps=5, timestamp_tracking=None):
        if not self.queue:
            logger.warning("Queue is empty, cannot generate MP4 file.")
            return False

        frame_list = []
        all_items = list(self.queue)

        if timestamp_tracking:
            logger.info(f"Filtering frames for MP4 generation with timestamp >= {timestamp_tracking}.")
            for item in all_items:
                if isinstance(item, tuple) and len(item) >= 3 and item[2] >= timestamp_tracking:
                    frame_list.append(item[1])
        else:
            logger.info("No timestamp filter, using all valid frames for MP4 generation.")
            for item in all_items:
                 if isinstance(item, tuple) and len(item) >= 3 and item[1] is not None:
                    frame_list.append(item[1])

        if not frame_list:
            logger.warning("No frames to write to video after filtering.")
            return False

        try:
            height, width, _ = frame_list[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # Save to output directory
            full_path = os.path.join(OUTPUT_DIR, output_filename)
            video_writer = cv2.VideoWriter(full_path, fourcc, fps, (width, height))
            for frame in frame_list:
                video_writer.write(frame)
            video_writer.release()
            logger.info(f"MP4 video saved as: {full_path}")
            return True
        except Exception as e:
            logger.error(f"Error generating MP4 video: {e}")
            return False


# --- Global State & Initialization ---
Chinese_Chess_Agent_Prompts = [
    "Analyzing the board, calculating the best move, please wait.",
    "One moment, retrieving the optimal strategy.",
    "Processing your request, please wait.",
    "Analyzing... please wait a moment.",
    "I'm working on the calculation and will provide a suggestion shortly.",
    "Please wait, I am deeply analyzing the current game state.",
    "I'm evaluating all possible moves, please wait.",
    "Querying for the best Chinese chess move..."
]
image_queue = CustomQueue(600)
executor = ThreadPoolExecutor()
final_results = None
piece_result = None
timestamp_tracking = None
frame_rate_LOW = True
tracking_result = None
CONTINUOUS_GUIDANCE_MODE = False
guidance_task = None

class DataType:
    IMAGE = 1
    AUDIO = 0

def get_time_now():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

def identify_data_type(header):
    try:
        header_value = int.from_bytes(header, byteorder='big')
        return DataType.AUDIO if header_value == 0 else DataType.IMAGE
    except Exception as e:
        logger.error(f"Error identifying data type from header: {e}")
        return None

# --- Inference and Processing Functions ---

def inference_llm(messages, mode='image'):
    from qwen_vl_utils import process_vision_info
    current_processor = llm_processor_video if mode == 'video' else llm_processor

    text = current_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = current_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    generated_ids = llm_model.generate(**inputs, max_new_tokens=Max_New_Tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = current_processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]

def prompt_refine(input_text: str, similarity_threshold: float = 0.725) -> str:
    from sentence_transformers import util
    processed_text = re.sub(r'[.?!]', '', input_text).strip()
    input_embedding = process_embedding(processed_text)

    best_intent = None
    best_score = 0.0

    for intent, embeddings in INTENT_EMBEDDINGS.items():
        similarities = util.pytorch_cos_sim(input_embedding, embeddings)[0]
        max_similarity = torch.max(similarities).item()
        if max_similarity > best_score:
            best_score = max_similarity
            best_intent = intent

    if best_intent and best_score >= similarity_threshold:
        refined_prompt = INTENT_TEMPLATES[best_intent][0]
        over_threshold = True
    else:
        refined_prompt = processed_text
        over_threshold = False
    return refined_prompt, over_threshold

def save_audio_file(audio_data, filename):
    filepath = os.path.join(RECEIVED_AUDIO_DIR, filename)
    with wave.open(filepath, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(audio_data)
    return filepath

def process_asr(filepath):
    if Using_remote_server:
        logger.info("Using remote server for ASR.")
        with open(filepath, 'rb') as audio_file_to_send:
            files = {'audio_file': audio_file_to_send}
            response = requests.post(f"http://{SERVER_IP}:{SERVER_PORT}/asr", files=files)
        response.raise_for_status()
        return response.json().get("text", "")
    else:
        logger.info("Using local server for ASR.")
        result = ASR_model.transcribe(filepath, language='en')
        os.remove(filepath)
        return result["text"]

def process_tts(output_text):
    if Using_remote_server:
        logger.info("Using remote server for TTS.")
        audio_filename = f"output_audio_{get_time_now()}.wav"
        audio_filepath = os.path.join(OUTPUT_DIR, audio_filename)
        response = requests.post(f"http://{SERVER_IP}:{SERVER_PORT}/tts", json={'text': output_text}, stream=True)
        response.raise_for_status()
        with open(audio_filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return audio_filepath
    else:
        logger.info("Using local server for TTS.")
        audio_filepath = os.path.join(OUTPUT_DIR, "output_audio.wav")

        def _infer_local_tts(gen_text, output_path):
            main_voice = {"ref_audio": tts_args.ref_audio, "ref_text": tts_args.ref_text}
            voices = tts_config.get("voices", {})
            voices["main"] = main_voice
            for voice in voices:
                voices[voice]["ref_audio"], voices[voice]["ref_text"] = preprocess_ref_audio_text(
                    voices[voice]["ref_audio"], voices[voice]["ref_text"]
                )

            audio_segment, final_sample_rate, _ = infer_process(
                voices["main"]["ref_audio"],
                voices["main"]["ref_text"],
                gen_text,
                loaded_tts_model,
                loaded_vocoder
            )
            with open(output_path, "wb") as f:
                sf.write(f, audio_segment, final_sample_rate)
            return output_path

        return _infer_local_tts(output_text, audio_filepath)

def process_embedding(text):
    if Using_remote_server:
        logger.info("Using remote server for embedding.")
        response = requests.post(f"http://{SERVER_IP}:{SERVER_PORT}/embed", json={"text": text})
        response.raise_for_status()
        return response.json().get("embedding", [])
    else:
        logger.info("Using local server for embedding.")
        return text_based_llm.embedding_model.encode(text, convert_to_tensor=True)

def get_intent_embeddings_as_tensor():
    if Using_remote_server:
        logger.info("Using remote server for intent embeddings.")
        response = requests.get(f"http://{SERVER_IP}:{SERVER_PORT}/intent_embeddings", timeout=60)
        response.raise_for_status()
        embeddings_dict = response.json()
        for key, value in embeddings_dict.items():
            embeddings_dict[key] = torch.tensor(value)
        return embeddings_dict
    else:
        logger.info("Using local server for intent embeddings.")
        return text_based_llm.INTENT_EMBEDDINGS

def analyze_text(text, help_threshold=0.75, image_threshold=0.7, similarity_threshold=0.75):
    if Using_remote_server:
        logger.info("Using remote server for text analysis.")
        payload = {"text": text, "help_threshold": help_threshold, "image_threshold": image_threshold, "similarity_threshold": similarity_threshold, "language": "en"}
        response = requests.post(f"http://{SERVER_IP}:{SERVER_PORT}/analyze", json=payload)
        response.raise_for_status()
        return response.json()
    else:
        logger.info("Using local server for text analysis.")
        return text_based_llm.analyze_input(text, help_threshold, image_threshold, similarity_threshold, "en")

def generate_chess_strategy_response(messages):
    if Using_remote_server:
        response = requests.post(f"http://{SERVER_IP}:{SERVER_PORT}/generate", json={"messages": messages})
        response.raise_for_status()
        return response.json().get("response", "")
    else:
        logger.info("Using local server for chess strategy generation.")
        return text_based_llm.generate_response(messages).strip()

def job_2_chess_strategy():
    start_time = time.time()
    chess_strategy = None
    timeout_message = "No suitable strategy found, the board is not fully visible, try adjusting the view."

    while time.time() - start_time < 3:
        try:
            if not image_queue.empty():
                _, image_data, _ = image_queue.get_latest()
                _, _, chess_strategy, board_change = my_Chinese_Chessboard.add_picture(image_data)
                if board_change and chess_strategy:
                    break
        except Exception as e:
            logger.error(f"Error in chess strategy job: {e}")
            break

    if not chess_strategy:
        return timeout_message

    if chess_strategy == "This board state is not in the opening book, please try another move.":
        return chess_strategy

    if chess_strategy[0].endswith(', winrate:-)'):
        system_message = (
            "As an AI assistant, translate chess engine moves into friendly, natural language. "
            "Input is like `Move (score: value, winrate: -)`. `score` > 0 is an advantage. "
            "Describe the move and assess the situation concisely, like talking to a friend."
        )
    else:
        system_message = (
            "As an AI assistant, translate chess engine moves into friendly, natural language. "
            "Input is like `Move (score: value, winrate: value)`. `score` shows advantage, `winrate` is the red player's win chance. "
            "Describe the move and the situation concisely and naturally."
        )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": str(chess_strategy[:2])}
    ]
    return generate_chess_strategy_response(messages)

async def continuous_guidance_mode_task(websocket):
    global CONTINUOUS_GUIDANCE_MODE
    while CONTINUOUS_GUIDANCE_MODE:
        advice = await asyncio.to_thread(job_2_chess_strategy)
        if advice != "No suitable strategy found, the board is not fully visible, try adjusting the view.":
            try:
                audio_filepath = await asyncio.to_thread(process_tts, advice)
                logger.info(f"Continuous guidance TTS completed: {audio_filepath}")
                await send_audio_to_frontend(websocket, audio_filepath)
            except Exception as e:
                logger.error(f"Error in continuous guidance TTS: {e}")
        await asyncio.sleep(4)

def process_question(text_input, image=None, needs_image=True, system_prompt=None):
    base_system_prompt = (
        "You are a professional and cautious smart glasses AI assistant. Follow these principles:\n"
        "0. Ensure answers are concise and clear.\n"
        "1. Provide answers only when completely certain.\n"
        "2. Maintain professionalism, accuracy, and objectivity.\n"
        "3. Avoid subjective judgment.\n"
        "4. Clearly state when something is uncertain."
    )
    image_system_prompt = (
        f"{base_system_prompt}\n"
        "5. When analyzing images, only describe what is clearly visible.\n"
        "6. If the image is blurry or difficult to recognize, clearly state this."
    )

    final_system_prompt = system_prompt or (image_system_prompt if needs_image else base_system_prompt)

    messages = [{"role": "system", "content": final_system_prompt}]
    user_content = []
    if needs_image and image:
        user_content.append({"type": "image", "image": image})
    user_content.append({"type": "text", "text": text_input})
    messages.append({"role": "user", "content": user_content})

    return messages

async def send_text_parameters(websocket, interval, quality):
    try:
        message = json.dumps({
            "imageAnalysis_interval": interval,
            "imageAnalysis_quality": quality
        })
        logger.info(f"Sending parameters to frontend: {message}")
        await websocket.send(message)
    except Exception as e:
        logger.error(f"Error sending parameters: {e}")

async def send_audio_to_frontend(websocket, audio_filepath):
    try:
        if not os.path.exists(audio_filepath):
            logger.error(f"Audio file not found: {audio_filepath}")
            return
        with open(audio_filepath, 'rb') as audio_file:
            await websocket.send(audio_file.read())
        logger.info(f"Successfully sent audio to frontend: {audio_filepath}")
    except Exception as e:
        logger.error(f"Error sending audio to frontend: {e}")

async def process_audio(data, websocket):
    global timestamp_tracking, tracking_result, frame_rate_LOW, CONTINUOUS_GUIDANCE_MODE, guidance_task, piece_result, final_results

    try:
        # --- ASR ---
        audio_data = data[HEADER_SIZE:]
        filename = f"audio_{get_time_now()}.wav"
        filepath = await asyncio.to_thread(save_audio_file, audio_data, filename)
        text_input = await asyncio.to_thread(process_asr, filepath)
        logger.info(f"ASR result: '{text_input}'")

        if not text_input or text_input.lower().strip() in ["thank you.", "you."]:
            return

        # --- Intent Analysis ---
        if not image_queue.empty():
            image_data_b64, _, _ = image_queue.get_latest()
        else:
            image_data_b64 = None

        refined_text, is_template_match = await asyncio.to_thread(prompt_refine, text_input)

        if not is_template_match:
            analysis = await asyncio.to_thread(analyze_text, text_input)
            is_help_request = analysis['help_request']
            needs_image = analysis['need_image']
            if not is_help_request:
                return
        else:
            is_help_request = True
            needs_image = True

        output_text = ""
        # --- Action Logic ---
        if is_help_request and needs_image:
            logger.info(f"Processing image-based request. Refined prompt: '{refined_text}'")

            if refined_text == "What should be my next move?":
                logger.info("Entering Chinese Chess analysis mode.")
                await send_audio_to_frontend(websocket, await asyncio.to_thread(process_tts, random.choice(Chinese_Chess_Agent_Prompts)))
                output_text = await asyncio.to_thread(job_2_chess_strategy)

            elif refined_text == "Where is the... I forgot.":
                logger.info("Entering Memory analysis mode.")
                memory_frames = image_queue.get_all(timestamp_tracking)
                video_frame_base64_list = ["data:image;base64," + frame[0] for frame in memory_frames]
                messages = [{"role": "user", "content": [
                    {"type": "video", "video": video_frame_base64_list, "fps": 2.0},
                    {"type": "text", "text": text_input}
                ]}]
                output_text = await asyncio.to_thread(inference_llm, messages, "video")

            elif refined_text == "Enter high frame rate mode.":
                logger.info("Switching to high frame rate mode for piece tracking.")
                frame_rate_LOW = False
                await send_text_parameters(websocket, 200, 25)
                # Code for job_3 would run here
                output_text = "Okay, I've entered high-speed mode. Please move the pieces."

            elif refined_text in ["What piece is at this position?", "Exit high frame rate mode."]:
                logger.info("Exiting high frame rate mode and performing inference.")
                if not frame_rate_LOW:
                    image_queue.to_mp4(timestamp_tracking=timestamp_tracking)
                    frame_rate_LOW = True
                    await send_text_parameters(websocket, 500, 80)
                    await send_audio_to_frontend(websocket, await asyncio.to_thread(process_tts, "Analyzing, please wait..."))
                    # Placeholder for tracking logic
                    output_string = "The pieces from left to right are Red Chariot, Black Cannon, and Red Horse."

                    if "this" in text_input.lower():
                         output_text = await asyncio.to_thread(openai_API,
                                                                  system_prompt="As an AI assistant, answer questions about Chinese chess pieces based on their positions. Be natural and direct.",
                                                                  test_text=f"Given pieces from left to right are: {output_string}. Now answer: {text_input}",
                                                                  model="gpt-4o", image_base64_strings=image_data_b64)
                    else:
                        output_text = f"Based on the movement, {output_string}"
                    tracking_result = output_string

            elif refined_text == "Enter continuous guidance mode.":
                logger.info("Activating continuous guidance mode.")
                CONTINUOUS_GUIDANCE_MODE = True
                if guidance_task is None or guidance_task.done():
                    guidance_task = asyncio.create_task(continuous_guidance_mode_task(websocket))
                output_text = "Okay, I will now provide suggestions proactively."

            elif refined_text == "Exit continuous guidance mode.":
                logger.info("Deactivating continuous guidance mode.")
                CONTINUOUS_GUIDANCE_MODE = False
                if guidance_task and not guidance_task.done():
                    guidance_task.cancel()
                output_text = "Okay, I will stop guiding the chess game."

            else:
                CONTINUOUS_GUIDANCE_MODE = False
                if image_data_b64:
                    messages = process_question(refined_text, "data:image;base64," + image_data_b64, needs_image=True)
                    output_text = await asyncio.to_thread(inference_llm, messages)
                else:
                    output_text = "I don't have a recent image to analyze. Please try again."

        elif is_help_request:
            logger.info(f"Processing text-only request. Refined prompt: '{refined_text}'")
            CONTINUOUS_GUIDANCE_MODE = False
            messages = process_question(refined_text, None, needs_image=False)
            output_text = await asyncio.to_thread(inference_llm, messages)

        if output_text:
            logger.info(f"Generated response: '{output_text}'")
            audio_filepath = await asyncio.to_thread(process_tts, output_text)
            await send_audio_to_frontend(websocket, audio_filepath)

    except Exception as e:
        logger.error(f"Error in process_audio: {e}", exc_info=True)

async def websocket_handler(websocket, path):
    client_id = get_time_now()
    logger.info(f"New client connected: {client_id}")
    global guidance_task
    try:
        async for message in websocket:
            if not isinstance(message, bytes) or len(message) < HEADER_SIZE:
                logger.warning(f"Received invalid message from {client_id}")
                continue

            data_type = identify_data_type(message[:HEADER_SIZE])

            if data_type == DataType.IMAGE:
                image_data = message[HEADER_SIZE:]
                nparr = np.frombuffer(image_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is not None:
                    frame_base64 = base64.b64encode(image_data).decode('utf-8')
                    timestamp = get_time_now()
                    image_queue.put((frame_base64, frame, timestamp))
                    if visualization_mode:
                        cv2.imshow('Received Frame', frame)
                        cv2.waitKey(1)

            elif data_type == DataType.AUDIO:
                asyncio.create_task(process_audio(message, websocket))

    except websockets.exceptions.ConnectionClosed:
        logger.info(f"Client disconnected: {client_id}")
    except Exception as e:
        logger.error(f"Error handling client {client_id}: {e}")
    finally:
        if guidance_task and not guidance_task.done():
            guidance_task.cancel()
            logger.info("Continuous guidance task cancelled for disconnected client.")
        global CONTINUOUS_GUIDANCE_MODE
        CONTINUOUS_GUIDANCE_MODE = False

async def main():
    server = await websockets.serve(
        websocket_handler, "localhost", CONFIG["WEBSOCKET_PORT"],
        max_size=None, ping_interval=None
    )
    logger.info(f"WebSocket server started on ws://localhost:{CONFIG['WEBSOCKET_PORT']}")
    await server.wait_closed()

def init_llm(model_path_or_name):
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    logger.info(f"Loading VLM model: {model_path_or_name}")
    start_time = time.perf_counter()

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path_or_name,
        torch_dtype=torch.bfloat16,
    ).cuda()

    processor = AutoProcessor.from_pretrained(model_path_or_name)
    processor_video = AutoProcessor.from_pretrained(
        model_path_or_name, min_pixels=256*28*28, max_pixels=360*28*28
    )

    load_time = time.perf_counter() - start_time
    logger.info(f"VLM model and processors loaded in {load_time:.2f} seconds.")
    return model, processor, processor_video

if __name__ == "__main__":
    if not Using_remote_server:
        # Initialize VLM
        llm_model, llm_processor, llm_processor_video = init_llm(CONFIG['MODELS']['VLM'])
        INTENT_EMBEDDINGS = get_intent_embeddings_as_tensor()

    cls_mapping = {
        1: "Red Pawn", 2: "Red Cannon", 3: "Red Chariot", 4: "Red Horse",
        5: "Red Elephant", 6: "Red Advisor", 7: "Red General",
        8: "Black Chariot", 9: "Black Horse", 10: "Black Elephant",
        11: "Black Advisor", 12: "Black General", 13: "Black Pawn",
        14: "Black Cannon"
    }

    Traking_local = True
    # Ensure this class handles initialization properly given the path changes above if it relies on internal paths
    my_Chinese_Chessboard = Chinese_Chessboard_New(memory_length=5, stability_threshold_ratio=0.75)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server shutting down...")
    finally:
        if visualization_mode:
            cv2.destroyAllWindows()
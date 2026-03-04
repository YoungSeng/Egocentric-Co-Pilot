import os
import json
import glob
import argparse
from tqdm import tqdm
import torch
from transformers import pipeline
import logging
import subprocess
import tempfile
import shutil
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEFAULT_ASR_MODEL_PATH, DEFAULT_VIDEO_ROOT, DEFAULT_OUTPUT_DIR

# --- Configuration ---
LANGUAGE = "chinese"
BATCH_SIZE = 8

logging.getLogger("transformers.feature_extraction_utils").setLevel(logging.ERROR)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Video ASR Transcripts")
    parser.add_argument("--model_path", type=str, default=DEFAULT_ASR_MODEL_PATH)
    parser.add_argument("--video_root", type=str, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--output_file", type=str, default=os.path.join(DEFAULT_OUTPUT_DIR, "video_asr_transcripts.json"))
    return parser.parse_args()

def extract_audio_from_video(video_path, audio_output_path):
    command = [
        'ffmpeg',
        '-i', video_path,
        '-vn',
        '-acodec', 'pcm_s16le',
        '-ar', '16000',
        '-ac', '1',
        '-loglevel', 'error',
        '-y',
        audio_output_path
    ]
    try:
        subprocess.run(command, check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def main():
    args = parse_args()
    print("--- Start Video ASR Task ---")

    if not shutil.which("ffmpeg"):
        print("[Error] `ffmpeg` not found.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Loading ASR pipeline from '{args.model_path}'...")
    try:
        asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=args.model_path,
            torch_dtype=torch.bfloat16,
            device=device,
            model_kwargs={"attn_implementation": "flash_attention_2"}
        )
        print("ASR pipeline loaded.")
    except Exception as e:
        print(f"[Error] Failed to load pipeline: {e}")
        return

    video_paths = glob.glob(os.path.join(args.video_root, '**', '*.mp4'), recursive=True)
    if not video_paths:
        print(f"[Error] No videos found in '{args.video_root}'")
        return
    print(f"Found {len(video_paths)} videos.")

    results = {}
    if os.path.exists(args.output_file):
        with open(args.output_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing transcripts.")

    videos_to_process = [path for path in video_paths if os.path.basename(path) not in results]
    if not videos_to_process:
        print("All videos processed.")
        return
    print(f"Processing {len(videos_to_process)} new videos.")

    temp_dir = tempfile.mkdtemp()
    print(f"Temporary audio dir: {temp_dir}")

    try:
        for i in tqdm(range(0, len(videos_to_process), BATCH_SIZE), desc="Processing Batches"):
            batch_video_paths = videos_to_process[i:i + BATCH_SIZE]
            audio_paths_for_batch = []
            path_map = {}

            for video_path in batch_video_paths:
                video_filename = os.path.basename(video_path)
                temp_audio_path = os.path.join(temp_dir, f"{os.path.splitext(video_filename)[0]}.wav")

                if extract_audio_from_video(video_path, temp_audio_path):
                    audio_paths_for_batch.append(temp_audio_path)
                    path_map[temp_audio_path] = video_filename
                else:
                    tqdm.write(f"  [Info] No audio track for '{video_filename}'.")
                    results[video_filename] = ""

            if audio_paths_for_batch:
                try:
                    outputs = asr_pipeline(
                        audio_paths_for_batch,
                        chunk_length_s=30,
                        batch_size=len(audio_paths_for_batch),
                        generate_kwargs={"language": LANGUAGE, "task": "transcribe"}
                    )

                    for audio_path, output in zip(audio_paths_for_batch, outputs):
                        video_filename = path_map[audio_path]
                        transcribed_text = output['text'].strip()
                        results[video_filename] = transcribed_text

                except Exception as e:
                    tqdm.write(f"\n  [ASR Error] Batch failed: {e}")
                    for audio_path in audio_paths_for_batch:
                        video_filename = path_map[audio_path]
                        results[video_filename] = f"Error: ASR pipeline failed - {e}"

            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)

    finally:
        print(f"Cleaning up: {temp_dir}")
        shutil.rmtree(temp_dir)

    print(f"--- Task Complete ---\nSaved to: {args.output_file}")

if __name__ == "__main__":
    main()
import os
import json
import glob
import argparse
import torch
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import sys

# 添加父目录到 path 以便导入 config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEFAULT_VLM_MODEL_PATH, DEFAULT_VIDEO_ROOT, DEFAULT_OUTPUT_DIR

# --- Configuration (Numbers kept exactly as original) ---
VIDEO_FPS_TO_SAMPLE = 1.0

SYSTEM_PROMPT = """You are an expert analyst for egocentric video from AI smart glasses. Your task is to generate a comprehensive, detailed, and objective description of the events in the video clip from a first-person ("I") perspective.
Your description must include:
1.  **Core Actions**: Detail what I am doing, seeing, and interacting with.
2.  **Object Details**: Mention key objects, their state, and any changes in their state (e.g., "I picked up an empty cup," "I am looking at a whiteboard with some diagrams on it").
3.  **Sequence and Intent**: Describe the sequence of actions. Based on the context, infer my likely intention or what the immediate next step might be (e.g., "After picking up the keys, I am walking towards the door, likely to leave the house.").
4.  **Environment and Context**: Briefly describe the surrounding environment (e.g., "in a kitchen," "at an office desk"). Note any significant background elements or people.
5.  **Dialogue or Sounds**: If any speech is implied or visible (e.g., lip movement), mention the topic of conversation (e.g., "It seems we are discussing a puzzle.").
Your output should be a single, coherent paragraph in English. This caption will be used for future question-answering, so be as informative as possible.
"""
USER_PROMPT = "Based on the video, provide a detailed, first-person description of the events as instructed."


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Native Video Captions")
    parser.add_argument("--model_path", type=str, default=DEFAULT_VLM_MODEL_PATH, help="Path to VLM model")
    parser.add_argument("--video_root", type=str, default=DEFAULT_VIDEO_ROOT, help="Root directory of videos")
    parser.add_argument("--output_file", type=str,
                        default=os.path.join(DEFAULT_OUTPUT_DIR, "video_captions_native.json"), help="Output JSON path")
    return parser.parse_args()


def main():
    args = parse_args()
    print("--- Start Video Caption Generation (Native) ---")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Loading model from '{args.model_path}'...")
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(args.model_path, use_fast=True)
        print("Model loaded.")
    except Exception as e:
        print(f"[Error] Model load failed: {e}")
        return

    video_paths = glob.glob(os.path.join(args.video_root, '**', '*.mp4'), recursive=True)
    if not video_paths:
        print(f"[Error] No .mp4 files found in '{args.video_root}'")
        return

    print(f"Found {len(video_paths)} videos.")

    results = {}
    if os.path.exists(args.output_file):
        with open(args.output_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing captions.")

    for video_path in tqdm(video_paths, desc="Processing"):
        video_filename = os.path.basename(video_path)

        if video_filename in results:
            continue

        try:
            messages = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {
                    "role": "user", "content": [
                    {"type": "video", "video": video_path, "fps": VIDEO_FPS_TO_SAMPLE},
                    {"type": "text", "text": USER_PROMPT}
                ]}
            ]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                fps=VIDEO_FPS_TO_SAMPLE,
                return_tensors="pt",
            ).to(device)

            generated_ids = model.generate(**inputs, max_new_tokens=2048)

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            results[video_filename] = output_text[0].strip()

        except Exception as e:
            print(f"\n  [Error] Inference failed for {video_filename}: {e}")
            results[video_filename] = f"Error: VLM generation failed - {e}"

        if len(results) % 10 == 0:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)

    print("Saving final results...")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"Done. Saved to {args.output_file}")


if __name__ == "__main__":
    main()
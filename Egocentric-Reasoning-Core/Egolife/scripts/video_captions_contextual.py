import os
import json
import glob
import pathlib
import argparse
import torch
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from collections import defaultdict
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEFAULT_VLM_MODEL_PATH, DEFAULT_VIDEO_ROOT, DEFAULT_OUTPUT_DIR

# --- Configuration ---
VIDEO_FPS_TO_SAMPLE = 1.0

SYSTEM_PROMPT = """You are an expert analyst for egocentric video from AI smart glasses. Your task is to generate a comprehensive, detailed, and objective description of the events in the video clip from a first-person ("I") perspective.
Your description must include:
1.  **Core Actions**: Detail what I am doing, seeing, and interacting with.
2.  **Object Details**: Mention key objects, their state, and any changes in their state (e.g., "I picked up an empty cup," "I am looking at a whiteboard with some diagrams on it").
3.  **Sequence and Intent**: Describe the sequence of actions. Based on the context, infer my likely intention or what the immediate next step might be (e.g., "After picking up the keys, I am walking towards the door, likely to leave the house.").
4.  **Environment and Context**: Briefly describe the surrounding environment (e.g., "in a kitchen," "at an office desk").
Your output should be a single, coherent paragraph in English. This caption will be used for future question-answering, so be as informative as possible.
"""

USER_PROMPT_NO_CONTEXT = "Based on the video, provide a detailed, first-person description of the events as instructed."

USER_PROMPT_WITH_CONTEXT_TEMPLATE = """Here is a summary of what happened right before this moment:
"{previous_caption}"

Now, based on the new video clip, continue the narrative from a first-person perspective. Describe the new actions and observations, ensuring your description flows naturally from the previous context.
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Contextual Video Captions")
    parser.add_argument("--model_path", type=str, default=DEFAULT_VLM_MODEL_PATH)
    parser.add_argument("--video_root", type=str, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--output_file", type=str, default=os.path.join(DEFAULT_OUTPUT_DIR, "video_captions_contextual.json"))
    return parser.parse_args()

def group_videos_by_day(root_dir: str) -> dict:
    print("Searching and grouping videos...")
    all_videos = glob.glob(os.path.join(root_dir, '**', '*.mp4'), recursive=True)
    if not all_videos:
        print(f"[Error] No .mp4 files found in '{root_dir}'.")
        return {}

    all_videos.sort()
    print(f"Found and sorted {len(all_videos)} videos.")

    grouped_videos = defaultdict(list)
    for video_path in all_videos:
        p = pathlib.Path(video_path)
        day_folder = p.parent.name
        participant_folder = p.parent.parent.name
        group_key = f"{participant_folder}_{day_folder}"
        grouped_videos[group_key].append(video_path)

    print(f"Grouped into {len(grouped_videos)} 'days'.")
    return grouped_videos

def generate_caption(model, processor, device, video_path, previous_caption=None):
    try:
        if previous_caption:
            user_prompt = USER_PROMPT_WITH_CONTEXT_TEMPLATE.format(previous_caption=previous_caption)
        else:
            user_prompt = USER_PROMPT_NO_CONTEXT

        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "video", "video": video_path, "fps": VIDEO_FPS_TO_SAMPLE},
                {"type": "text", "text": user_prompt}
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
        return output_text[0].strip()

    except Exception as e:
        error_message = f"Error: VLM generation failed - {e}"
        print(f"\n  [Error] Inference failed: {os.path.basename(video_path)}, Error: {e}")
        return error_message

def main():
    args = parse_args()
    print("--- Start Contextual Caption Generation ---")

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

    grouped_videos = group_videos_by_day(args.video_root)
    if not grouped_videos:
        return

    results = {}
    if os.path.exists(args.output_file):
        with open(args.output_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing captions.")

    total_videos = sum(len(v) for v in grouped_videos.values())
    pbar = tqdm(total=total_videos, desc="Overall Progress")

    for day_key, video_list in sorted(grouped_videos.items()):
        print(f"\n--- Processing Day: {day_key} ({len(video_list)} videos) ---")
        previous_caption = None

        for video_path in video_list:
            video_filename = os.path.basename(video_path)
            pbar.set_description(f"Processing {video_filename}")

            if video_filename in results:
                print(f"  -> Skipped: {video_filename}")
                previous_caption = results[video_filename]
                pbar.update(1)
                continue

            context_for_next = previous_caption
            if previous_caption and previous_caption.startswith("Error:"):
                context_for_next = None

            caption = generate_caption(
                model, processor, device, video_path,
                previous_caption=context_for_next
            )

            results[video_filename] = caption
            previous_caption = caption

            pbar.update(1)

            if pbar.n % 10 == 0:
                with open(args.output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=4, ensure_ascii=False)

    pbar.close()
    print("\nSaving final results...")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"--- Task Complete ---\nSaved to: {args.output_file}")

if __name__ == "__main__":
    main()
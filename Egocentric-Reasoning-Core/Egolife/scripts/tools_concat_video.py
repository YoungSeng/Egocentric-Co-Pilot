import os
import json
import glob
import subprocess
import argparse
import sys

# --- Configuration (Numbers kept exactly as original) ---
NUM_VIDEOS_TO_PROCESS = 20


def parse_args():
    parser = argparse.ArgumentParser(description="Concat Videos and Extract Captions")
    parser.add_argument("--video_folder", type=str, required=True, help="Path to folder containing videos")
    parser.add_argument("--captions_file", type=str, required=True, help="Path to global captions JSON")
    parser.add_argument("--output_video", type=str, default="concatenated_video.mp4")
    parser.add_argument("--output_json", type=str, default="concatenated_captions.json")
    return parser.parse_args()


def process_with_ffmpeg(args):
    print("--- Start FFmpeg Processing ---")
    print(f"Target Folder: {args.video_folder}")
    print(f"Limit: {NUM_VIDEOS_TO_PROCESS} videos")

    # Load captions
    try:
        with open(args.captions_file, 'r', encoding='utf-8') as f:
            all_captions_data = json.load(f)
    except Exception as e:
        print(f"[Error] Failed to load captions: {e}")
        return

    # Find videos
    try:
        all_video_files = sorted(glob.glob(os.path.join(args.video_folder, '*.mp4')))
        if not all_video_files:
            print(f"[Error] No mp4 files in {args.video_folder}")
            return

        videos_to_process = all_video_files[:NUM_VIDEOS_TO_PROCESS]
        print(f"Selected {len(videos_to_process)} videos.")
    except Exception as e:
        print(f"[Error] Video search failed: {e}")
        return

    # 1. FFmpeg Concat
    file_list_path = "ffmpeg_file_list.txt"
    try:
        with open(file_list_path, 'w') as f:
            for video_path in videos_to_process:
                f.write(f"file '{video_path}'\n")

        print("Running FFmpeg...")
        command = [
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
            '-i', file_list_path, '-c', 'copy', args.output_video
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Video saved to: {args.output_video}")

    except Exception as e:
        print(f"[Error] FFmpeg failed: {e}")
        return
    finally:
        if os.path.exists(file_list_path):
            os.remove(file_list_path)

    # 2. Extract Captions
    extracted_captions = []
    for video_path in videos_to_process:
        fname = os.path.basename(video_path)
        caption = all_captions_data.get(fname, f"Caption not found for {fname}")
        extracted_captions.append(caption)

    try:
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(extracted_captions, f, indent=4, ensure_ascii=False)
        print(f"Captions saved to: {args.output_json}")
    except Exception as e:
        print(f"[Error] Failed to save JSON: {e}")

    print("--- Task Complete ---")


if __name__ == "__main__":
    args = parse_args()
    process_with_ffmpeg(args)
import json
import os
import re
import subprocess
from pathlib import Path
import shutil
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import argparse
from tqdm import tqdm
from datetime import timedelta

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="recipe_multi_step_localization",
                    choices=["recipe_multi_step_localization", "recipe_rough_step_localization",
                             "ingredient_ingredient_adding_localization", "recipe_step_localization",
                             "fine_grained_action_localization", "recipe_prep_localization",
                             "ingredient_ingredients_order", "ingredient_ingredient_weight"]
                    )
parser.add_argument("--multi_video", action='store_true')
parser.add_argument("--wobbox", action='store_true')

parser.add_argument("--video_segment", type=str, default="full",
                    choices=["full", "segment", "interval"])
parser.add_argument("--video_merge_strategy", type=str, default="multi_input",
                    choices=["concat", "multi_input"])
parser.add_argument("--chunk_video", action='store_true',
                    help="是否将视频分块处理")
parser.add_argument("--chunk_duration", type=int, default=10,  # 10
                    help="视频分块时长（分钟）")
# Modified: Default paths changed to relative
parser.add_argument("--hd_epic_vqa_annotations", type=str, default="./dataset/hd-epic-annotations/vqa-benchmark/")
parser.add_argument("--hd_epic_database", type=str, default="./dataset/HD-EPIC/")
parser.add_argument("--inference_percentage", type=int, default=100,
                    help="Percentage of samples to use for inference (0-100). Default is 100.")
parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
parser.add_argument("--print_intermediate", action="store_true",
                    help="Print intermediate question and prediction for each sample.")
parser.add_argument("--task_name", type=str, default="PreprocessedVideos")
parser.add_argument("--voting_ensemble", type=str, default="")
# Added: Argument for temp directory
parser.add_argument("--temp_dir", type=str, default="./tmp/thinking_video_segments")

args = parser.parse_args()

task = args.task
model_name = args.base_model
HD_EPIC_database = args.hd_epic_database
preprocessed_videos_base = os.path.join(HD_EPIC_database, args.task_name)

# Configuration Paths
if args.wobbox:
    input_json_path = os.path.join(preprocessed_videos_base, f"{task}/{task}.json")
else:
    input_json_path = os.path.join(args.hd_epic_vqa_annotations, f"{task}.json")
output_dir = os.path.join(preprocessed_videos_base, f"{task}")
narration_dir = os.path.join(output_dir, "narration")
temp_dir = Path(args.temp_dir)

# Initialize Model and Processor
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_name)

# Create Directories
os.makedirs(narration_dir, exist_ok=True)
temp_dir.mkdir(parents=True, exist_ok=True)

# Load Data
with open(input_json_path, "r") as f:
    data = json.load(f)

# Calculate samples
total_samples = len(data)
clamped_percentage = max(0, min(100, args.inference_percentage))
num_samples_to_process = int(total_samples * (clamped_percentage / 100.0))

print(f"Total samples available: {total_samples}")
print(f"Processing {num_samples_to_process} samples ({clamped_percentage}%) for inference...")


def split_video_into_chunks(input_path, chunk_duration_min=10):
    """Split video into chunks"""
    chunk_duration = chunk_duration_min * 60  # seconds
    duration = get_video_duration(input_path)

    chunks = []
    start = 0
    while start < duration:
        end = min(start + chunk_duration, duration)
        chunk_name = f"{Path(input_path).stem}_chunk_{len(chunks)}.mp4"
        output_path = temp_dir / chunk_name

        # ffmpeg exact cut
        command = [
            'ffmpeg', '-y',
            '-ss', str(start),
            '-i', input_path,
            '-t', str(end - start),
            '-c', 'copy',
            '-c:v', 'libx264',
            str(output_path)
        ]
        subprocess.run(command, check=True, stderr=subprocess.DEVNULL)

        chunks.append({
            "path": output_path,
            "start": start,
            "end": end
        })
        start = end
    return chunks


def process_video_chunks(original_path, args):
    """Process video chunks"""
    if not args.chunk_video:
        return [{"path": Path(original_path), "start": 0, "end": get_video_duration(original_path)}]

    return split_video_into_chunks(original_path, args.chunk_duration)


def parse_time(time_str):
    """Convert time string to seconds"""
    h, m, s = time_str.split(":")
    s, ms = s.split(".")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def extract_segments(choice_str):
    """Extract time segments from choice string"""
    segments = []
    time_matches = re.findall(r"<TIME (.*?) video 1>", choice_str)
    for i in range(0, len(time_matches), 2):
        start = parse_time(time_matches[i])
        end = parse_time(time_matches[i + 1])
        segments.append((start, end))
    return segments


def trim_video(input_path, output_path, start, end):
    """Trim video using ffmpeg"""
    duration = end - start
    command = [
        'ffmpeg',
        '-y',
        '-ss', str(start),
        '-i', input_path,
        '-t', str(duration),
        '-c', 'copy',
        '-c:v', 'libx264',
        output_path
    ]
    subprocess.run(command, check=True, stderr=subprocess.DEVNULL)


def generate_video_narration(video_path):
    """Generate description using Qwen"""
    try:
        messages = [{
            "role": "user",
            "content": [
                {"type": "video", "video": str(video_path)},
                {"type": "text",
                 "text": "Describe in detail the cooking steps that take place in this video; answers need to be concise and clear, directly describing the action itself."}
            ]
        }]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        # Process inputs
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids = [ids[len(input_ids):] for input_ids, ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return output_text.replace('\n', ' ')
    except Exception as e:
        print(f"Failed to generate description: {str(e)}")
        return "Unable to generate description"


def extract_target_time(question_text):
    """Extract target timestamp from question"""
    match = re.search(r"shown in (\d+:\d+:\d+\.\d+)", question_text)
    if not match:
        return None
    return parse_time(match.group(1))


def get_video_duration(input_path):
    """Get total video duration in seconds"""
    result = subprocess.run([
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        input_path
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return float(result.stdout)


def process_video_segment(original_path, question_text, segment_duration):
    """Process video segment and return (processed_path, time_offset)"""
    target_time = extract_target_time(question_text)
    if not target_time:
        return original_path, 0

    # Calculate interval
    video_duration = get_video_duration(original_path)
    start_time = max(target_time - segment_duration, 0)
    end_time = min(target_time + segment_duration, video_duration)
    duration = end_time - start_time

    # Generate temp filename
    output_path = temp_dir / f"{Path(original_path).stem}_trimmed_{start_time:.1f}-{end_time:.1f}.mp4"

    if not output_path.exists():
        trim_video(original_path, output_path, start_time, duration)

    # Calculate relative time offset
    time_offset = target_time - start_time
    return output_path, time_offset


def process_video_interval(original_path, start_time, end_time):
    """Process specific video interval"""
    duration = end_time - start_time
    output_path = temp_dir / f"{Path(original_path).stem}_interval_{start_time:.1f}-{end_time:.1f}.mp4"

    if not output_path.exists():
        trim_video(original_path, output_path, start_time, duration)

    return output_path, duration


def process_multi_videos(q_data, args):
    """Process multi-video input"""
    video_entries = [v for k, v in q_data['inputs'].items() if k.startswith('video ')]
    processed_paths = []
    temp_files = []
    time_offset = 0

    for idx, video_info in enumerate(video_entries):
        video_id = video_info['id']
        dir_part = video_id.split('-')[0]
        # Modified: Use os.path.join for portability
        original_path = os.path.join(args.hd_epic_database, "Videos", dir_part, f"{video_id}.mp4")

        if args.video_segment == "segment":
            path, offset = process_video_segment(original_path, q_data['question'], args.segment_duration)
            if idx == 0:  # Only adjust offset for first video
                time_offset = offset
        elif args.video_segment == "interval":
            if 'start_time' not in video_info or 'end_time' not in video_info:
                raise ValueError("Video info missing time interval")
            path, _ = process_video_interval(
                original_path,
                parse_time(video_info['start_time']),
                parse_time(video_info['end_time'])
            )
        else:
            path = original_path

        processed_paths.append(path)
        if path != original_path:
            temp_files.append(path)

    return processed_paths, temp_files, time_offset


def build_video_content(video_paths, merge_strategy, q_id):
    """Build video content and calculate time offsets"""
    if merge_strategy == "concat" and len(video_paths) > 1:
        output_path = temp_dir / f"concat_{q_id}.mp4"
        concat_path, durations = concatenate_videos(video_paths, output_path)

        # Calculate cumulative offsets
        time_offsets = [sum(durations[:i]) for i in range(len(durations))]
        return [concat_path], [concat_path], time_offsets
    return video_paths, [], [0]


def concatenate_videos(input_paths, output_path):
    """Concatenate multiple videos"""
    durations = [get_video_duration(p) for p in input_paths]

    list_file = temp_dir / "concat_list.txt"
    with open(list_file, 'w') as f:
        for path in input_paths:
            f.write(f"file '{path}'\n")

    command = [
        'ffmpeg',
        '-y',
        '-f', 'concat',
        '-safe', '0',
        '-i', str(list_file),
        '-c', 'copy',
        str(output_path)
    ]
    result = subprocess.run(command, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(f"Video concatenation failed: {result.stderr.decode()}")
    return output_path, durations


def seconds_to_time_str(seconds):
    """Convert seconds to HH:MM:SS.sss"""
    td = timedelta(seconds=seconds)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = td.microseconds // 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def adjust_choices_time(choices, time_offsets):
    """Adjust split times in choices to absolute times"""
    adjusted = []
    for choice in choices:
        prefix, content = re.match(r"([A-E]\.\s*)(.*)", choice).groups()
        time_segments = re.findall(r'\[V(\d+)\]\s*(\d+:\d+:\d+\.\d+)\s*-\s*(\d+:\d+:\d+\.\d+)', content)

        converted_times = []
        for vid_idx, start_str, end_str in time_segments:
            vid_idx = int(vid_idx) - 1
            offset = time_offsets[vid_idx] if vid_idx < len(time_offsets) else 0

            start = parse_time(start_str) + offset
            end = parse_time(end_str) + offset

            converted_times.append(
                f"{seconds_to_time_str(start)} - {seconds_to_time_str(end)}"
            )

        new_content = "; ".join(converted_times)
        adjusted.append(f"{prefix}{new_content}")

    return adjusted


def get_video_chunk_specs(video_path, args):
    """Get video chunk specs"""
    duration = get_video_duration(video_path)
    chunk_size = args.chunk_duration * 60
    overlap = 0

    specs = []
    start = 0.0
    while start < duration:
        end = min(start + chunk_size, duration)
        specs.append({"start": round(start, 2), "end": round(end, 2)})
        start = end - overlap
    return specs


# Main Loop
for q_id, key in tqdm(enumerate(list(data.keys())[0:num_samples_to_process]), desc="Processing questions"):
    entry = data[key]

    try:

        if args.multi_video:
            video_paths, temp_video_files, _ = process_multi_videos(entry, args)

            final_videos, concat_files, time_offsets = build_video_content(
                video_paths, args.video_merge_strategy, q_id
            )

            # Adjust choices
            adjusted_choices = adjust_choices_time(entry.get('choices', []), time_offsets)

            if args.video_merge_strategy == "concat":
                original_video_path = str(final_videos[0])
            if not Path(original_video_path).exists():
                print(f"Video does not exist: {original_video_path}")
                continue

            for choice_idx, choice_str in enumerate(adjusted_choices):
                option_letter = chr(65 + choice_idx)

                # Process each segment
                for seg_idx, segment in enumerate(choice_str[3:].split("; "), 0):
                    start_, end_ = segment.split(" - ")
                    start = parse_time(start_)
                    end = parse_time(end_)
                    try:
                        base_name = f"{key}_{option_letter}_{seg_idx}"
                        temp_video_path = temp_dir / f"{base_name}.mp4"
                        narration_path = os.path.join(narration_dir, f"{base_name}.txt")

                        if os.path.exists(narration_path):
                            continue
                        else:
                            trim_video(original_video_path, temp_video_path, start, end)
                            narration = generate_video_narration(temp_video_path)
                            with open(narration_path, "w") as f:
                                f.write(narration)
                            os.remove(temp_video_path)

                    except Exception as e:
                        print(f"Error processing {key} option {option_letter} segment {seg_idx}: {str(e)}")
                        continue

            for f in temp_video_files + concat_files:
                if Path(f).exists():
                    os.remove(f)

        if args.chunk_video:
            try:
                video_id = entry["inputs"]["video 1"]["id"]
                video_prefix = video_id.split("-")[0]
                # Modified: Relative path construction
                original_video_path = os.path.join(args.hd_epic_database, "Videos", video_prefix, f"{video_id}.mp4")

                if not Path(original_video_path).exists():
                    print(f"Video does not exist: {original_video_path}")
                    continue

                chunk_specs = get_video_chunk_specs(original_video_path, args)

                need_process = False
                expected_files = []

                for spec in chunk_specs:
                    base_name = f"{key}_chunk_{spec['start']:.0f}-{spec['end']:.0f}"
                    narration_path = os.path.join(narration_dir, f"{base_name}.txt")
                    expected_files.append(narration_path)

                    if not os.path.exists(narration_path):
                        need_process = True
                        break

                if not need_process:
                    if args.print_intermediate:
                        print(f"Validating chunks for {video_id}...")
                    for path in expected_files:
                        try:
                            with open(path, "r") as f:
                                if f.read().strip() in ["", "Unable to generate description"]:
                                    need_process = True
                                    break
                        except Exception as e:
                            print(f"File validation failed {path}: {str(e)}")
                            need_process = True
                            break

                if not need_process:
                    if args.print_intermediate:
                        print(f"All chunks for {video_id} are ready")
                    continue

                print(f"Starting chunk processing for {video_id}...")

                video_chunks = process_video_chunks(original_video_path, args)

                for chunk in video_chunks:
                    chunk_path = chunk["path"]
                    start_time = chunk["start"]
                    end_time = chunk["end"]

                    base_name = f"{key}_chunk_{start_time:.0f}-{end_time:.0f}"
                    narration_path = os.path.join(narration_dir, f"{base_name}.txt")

                    if os.path.exists(narration_path):
                        continue

                    try:
                        narration = generate_video_narration(chunk_path)
                        with open(narration_path, "w") as f:
                            f.write(narration)

                        if args.chunk_video and chunk_path.exists():
                            os.remove(chunk_path)

                    except Exception as e:
                        print(f"Error processing chunk {chunk_path}: {str(e)}")
                        continue

            except Exception as e:
                print(f"Error processing {key}: {str(e)}")
                continue

        else:

            video_id = entry["inputs"]["video 1"]["id"]
            video_prefix = video_id.split("-")[0]
            # Modified: Relative path construction
            original_video_path = os.path.join(args.hd_epic_database, "Videos", video_prefix, f"{video_id}.mp4")

            if not Path(original_video_path).exists():
                print(f"Video does not exist: {original_video_path}")
                continue

            for choice_idx, choice_str in enumerate(entry["choices"]):
                option_letter = chr(65 + choice_idx)
                segments = extract_segments(choice_str)

                for seg_idx, (start, end) in enumerate(segments, 0):
                    try:
                        base_name = f"{key}_{option_letter}_{seg_idx}"
                        temp_video_path = temp_dir / f"{base_name}.mp4"
                        narration_path = os.path.join(narration_dir, f"{base_name}.txt")

                        if os.path.exists(narration_path):
                            continue
                        else:
                            trim_video(original_video_path, temp_video_path, start, end)
                            narration = generate_video_narration(temp_video_path)
                            with open(narration_path, "w") as f:
                                f.write(narration)
                            os.remove(temp_video_path)

                    except Exception as e:
                        print(f"Error processing {key} option {option_letter} segment {seg_idx}: {str(e)}")
                        continue

    except Exception as e:
        print(f"Error processing {key}: {str(e)}")
        continue

# Save modified data
output_json_path = os.path.join(output_dir, "reformatted_questions_narration.json")
with open(output_json_path, "w") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)
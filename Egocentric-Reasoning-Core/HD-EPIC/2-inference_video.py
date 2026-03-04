import json
import re
import argparse
import subprocess
from pathlib import Path
from datetime import timedelta, datetime
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import os
from tqdm import tqdm
import cv2
import base64
import numpy as np

# Config Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument("--video_segment", type=str, default="full",
                    choices=["full", "segment", "interval"],
                    help="Use full video, segment, or interval")
parser.add_argument("--segment_duration", type=int, default=10,
                    help="Segment duration in seconds (before/after)")
parser.add_argument("--task", type=str, default="3d_perception_fixture_location",
                    choices=["3d_perception_fixture_interaction_counting",
                             "3d_perception_object_contents_retrieval",
                             "3d_perception_object_location", "fine_grained_action_localization",
                             "fine_grained_action_recognition", "fine_grained_how_recognition",
                             "fine_grained_why_recognition", "gaze_gaze_estimation",
                             "gaze_interaction_anticipation", "ingredient_ingredient_adding_localization",
                             "ingredient_ingredient_retrieval", "ingredient_ingredient_weight",
                             "ingredient_ingredients_order", "nutrition_nutrition_change",
                             "nutrition_video_nutrition_estimation", "object_motion_object_movement_counting",
                             "object_motion_object_movement_itinerary", "recipe_step_recognition",
                             "recipe_rough_step_localization", "recipe_multi_recipe_recognition",
                             "object_motion_stationary_object_localization", "recipe_following_activity_recognition",
                             "recipe_multi_step_localization", "recipe_step_localization", "recipe_prep_localization"])
parser.add_argument("--wobbox", action='store_true')

parser.add_argument("--start_index", type=int, default=0,
                    help="Start index for processing (inclusive, 0-based)")
parser.add_argument("--end_index", type=int, default=None,
                    help="End index for processing (exclusive)")
parser.add_argument("--existing_predictions", type=str, default=None,
                    help="Path to existing predictions to resume")
parser.add_argument("--wovision", action='store_true')
# Modified: Default paths changed to relative
parser.add_argument("--hd_epic_vqa_annotations", type=str, default="./dataset/hd-epic-annotations/vqa-benchmark/")
parser.add_argument("--hd_epic_database", type=str, default="./dataset/HD-EPIC/")
parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
parser.add_argument("--print_intermediate", action="store_true",
                    help="Print intermediate question and prediction for each sample.")
parser.add_argument("--inference_percentage", type=int, default=100,
                    help="Percentage of samples to use for inference (0-100). Default is 100.")
parser.add_argument("--evaluate_accuracy", action='store_true')
parser.add_argument("--task_name", type=str, default="PreprocessedVideos")
parser.add_argument("--voting_ensemble", type=str, default="")
# Added: Temp directory argument
parser.add_argument("--temp_dir", type=str, default="./tmp/qwen_video_segments")

args = parser.parse_args()

task = args.task
model_name = args.base_model
HD_EPIC_database = args.hd_epic_database
preprocessed_videos_base = os.path.join(HD_EPIC_database, args.task_name)

# Config Paths
if args.wobbox:
    questions_path = os.path.join(preprocessed_videos_base, f"{task}/{task}{args.voting_ensemble}.json")
else:
    questions_path = os.path.join(preprocessed_videos_base,
                                  f"{task}/reformatted_questions_with_narration_video{args.voting_ensemble}.json")
output_dir = Path(os.path.join(preprocessed_videos_base, f"{task}/"))
TEMP_VIDEO_DIR = Path(args.temp_dir)
TEMP_VIDEO_DIR.mkdir(parents=True, exist_ok=True)


def load_model():
    """Load Qwen model and processor"""
    if args.wovision:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cpu",
        ).cuda()
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def seconds_to_time_str(seconds):
    """Convert seconds to HH:MM:SS.sss"""
    td = timedelta(seconds=seconds)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = td.microseconds // 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def parse_choice_time(choice_str):
    """Parse time interval from choice string"""
    time_matches = re.findall(r'\d+:\d+:\d+\.\d+', choice_str)
    if len(time_matches) == 2:
        return parse_time(time_matches[0]), parse_time(time_matches[1])
    return None, None


def process_video_interval(original_path, start_time, end_time):
    """Process specific video interval"""
    duration = end_time - start_time
    output_path = TEMP_VIDEO_DIR / f"{Path(original_path).stem}_interval_{start_time:.1f}-{end_time:.1f}.mp4"

    if not output_path.exists():
        trim_video(original_path, output_path, start_time, duration)

    return output_path, duration


def adjust_choices_time(choices, start_time):
    """Adjust choice time intervals"""
    adjusted_choices = []
    for choice in choices:
        parts = choice.split('. ', 1)
        if len(parts) < 2:
            adjusted_choices.append(choice)
            continue

        prefix, content = parts
        t1, t2 = parse_choice_time(content)
        if t1 is None or t2 is None:
            adjusted_choices.append(choice)
            continue

        new_t1 = t1 - start_time
        new_t2 = t2 - start_time
        new_content = f"{seconds_to_time_str(new_t1)} - {seconds_to_time_str(new_t2)}"
        adjusted_choices.append(f"{prefix}. {new_content}")
    return adjusted_choices


def parse_time(time_str):
    """Convert time string to seconds"""
    try:
        h, m, s = time_str.split(':')
        return int(h) * 3600 + int(m) * 60 + float(s)
    except:
        raise ValueError(f"Invalid time format: {time_str}")


def extract_target_time(question_text):
    """Extract target time from question"""
    match = re.search(r"shown in (\d+:\d+:\d+\.\d+)", question_text)
    if not match:
        return None
    return parse_time(match.group(1))


def trim_video(input_path, output_path, start_time, duration):
    """Trim video using ffmpeg"""
    command = [
        'ffmpeg',
        '-y',
        '-ss', str(max(start_time, 0)),
        '-i', input_path,
        '-t', str(duration),
        '-c', 'copy',
        '-c:v', 'libx264',
        output_path
    ]
    result = subprocess.run(command, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg error: {result.stderr.decode()}")


def get_video_duration(input_path):
    """Get video duration in seconds"""
    result = subprocess.run([
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        input_path
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return float(result.stdout)


def process_video_segment(original_path, question_text, segment_duration):
    """Process video segment return (path, offset)"""
    target_time = extract_target_time(question_text)
    if not target_time:
        return original_path, 0

    video_duration = get_video_duration(original_path)
    start_time = max(target_time - segment_duration, 0)
    end_time = min(target_time + segment_duration, video_duration)
    duration = end_time - start_time

    output_path = TEMP_VIDEO_DIR / f"{Path(original_path).stem}_trimmed_{start_time:.1f}-{end_time:.1f}.mp4"

    if not output_path.exists():
        trim_video(original_path, output_path, start_time, duration)

    time_offset = target_time - start_time
    return output_path, time_offset


def adjust_question_time(question_text, new_time):
    """Adjust question time description"""
    adjusted_time = str(timedelta(seconds=new_time)).split('.')[0] + f".{int(new_time % 1 * 1000):03d}"
    return re.sub(
        r"(shown in )(\d+:\d+:\d+\.\d+)",
        f"\\g<1>{adjusted_time}",
        question_text
    )


def main():
    model, processor = load_model()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading questions from {questions_path}...")
    with open(questions_path, 'r') as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} questions.")

    total_samples = len(questions)
    clamped_percentage = max(0, min(100, args.inference_percentage))
    num_samples_to_process = int(total_samples * (clamped_percentage / 100.0))

    print(f"Total samples available: {total_samples}")
    print(f"Processing {num_samples_to_process} samples ({clamped_percentage}%) for inference...")

    existing_predictions = {}
    if args.existing_predictions:
        try:
            with open(args.existing_predictions, 'r') as f:
                existing_predictions = json.load(f)
            print(f"Loaded {len(existing_predictions)} existing predictions")
        except Exception as e:
            print(f"Error loading existing predictions: {str(e)}")

    question_items = list(questions.items())
    start_index = args.start_index
    end_index = args.end_index if args.end_index is not None else len(question_items)

    if start_index < 0 or start_index >= len(question_items):
        print(f"Warning: start_index ({start_index}) out of bounds. Resetting to 0.")
        start_index = 0
    if end_index is not None and (end_index < 0 or end_index > len(question_items)):
        print(f"Warning: end_index ({end_index}) out of bounds. Resetting to end of list.")
        end_index = len(question_items)
    if start_index >= end_index:
        print(f"Warning: start_index ({start_index}) is >= end_index ({end_index}). No questions will be processed.")
        question_items_to_process = []
    else:
        question_items_to_process = question_items[start_index:end_index]

    print(
        f"Processing questions from index {start_index} to {end_index - 1} (total {len(question_items_to_process)} questions)...")

    predictions = existing_predictions.copy()
    intermediate = []
    processed_count = 0

    for q_id, q_data in tqdm(question_items_to_process[0:num_samples_to_process], desc="Processing questions"):
        try:
            if predictions.get(q_id, -1) != -1:
                continue

            video_id = q_data['inputs']['video']['id']
            dir_part = video_id.split('-')[0]
            # Modified: Relative path
            original_video_path = os.path.join(args.hd_epic_database, "Videos", dir_part, f"{video_id}.mp4")

            if not Path(original_video_path).exists():
                raise FileNotFoundError(f"Video {original_video_path} not found")

            duration = float('inf')

            if args.video_segment == "segment":
                processed_path, time_offset = process_video_segment(
                    original_video_path,
                    q_data['question'],
                    args.segment_duration
                )
                video_path = str(processed_path)
                adjusted_question = adjust_question_time(q_data['question'], time_offset)
            elif args.video_segment == "interval":
                video_info = q_data['inputs']['video']
                start_time = parse_time(video_info['start_time'])
                end_time = parse_time(video_info['end_time'])

                video_duration = get_video_duration(original_video_path)
                start_time = max(0, min(start_time, video_duration - 1))
                end_time = min(end_time, video_duration)
                duration = end_time - start_time

                if duration <= 0:
                    raise ValueError("Invalid time interval")

                video_path, _ = process_video_interval(original_video_path, start_time, end_time)
                video_path = str(video_path)
                adjusted_question = q_data['question']

                if 'choices' in q_data:
                    adjusted_choices = adjust_choices_time(q_data['choices'], start_time)
                    adjusted_question += "\n".join(adjusted_choices)
            else:
                video_path = original_video_path
                adjusted_question = q_data['question']

                if 'choices' in q_data:
                    adjusted_question += "\n".join(q_data['choices'])

            if args.wovision:
                messages = [
                    {
                        "role": "system",
                        "content": "You are an expert Cooking analyzer. Answer the multiple choice question by giving ONLY the letter identifying the answer. Example: A or B or C or D or E. You MUST answer even if unsure."
                    },
                    {
                        "role": "user",
                        "content": adjusted_question
                    }]

                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                inputs = processor(text=[text], padding=True, return_tensors="pt").to("cuda")
            else:

                messages = [
                    {
                        "role": "system",
                        "content": "You are an expert video analyzer. Answer the multiple choice question by giving ONLY the letter identifying the answer. Example: A or B or C or D or E. You MUST answer even if unsure."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": video_path, "fps": 1.0},
                            {"type": "text", "text": adjusted_question}
                        ]
                    }]

                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)

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

            intermediate.append(f"{q_id}\nQuestion: {adjusted_question}\nOutput: {output_text}\n{'=' * 50}")

            if args.print_intermediate:
                print(f"{q_id}\nQuestion: {adjusted_question}\nOutput: {output_text}\n{'=' * 50}")

            predictions[q_id] = output_text

            if args.video_segment == "segment" or args.video_segment == "interval":
                os.remove(video_path)

            processed_count += 1

        except Exception as e:
            print(f"Error processing {q_id}: {str(e)}")
            predictions[q_id] = -1
            intermediate.append(f"{q_id}\nError: {str(e)}\n{'=' * 50}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"predictions_{task}_resumed_{timestamp}.json" if args.existing_predictions else f"predictions_{task}.json"

    with open(output_dir / f"predictions_{task}{args.voting_ensemble}.json", 'w') as f:
        json.dump(predictions, f, indent=2)

    print(f"Processing completed. Total processed: {processed_count} new questions")
    print(f"Results saved to: {output_dir / output_filename}")

    with open(output_dir / f"intermediate_results{args.voting_ensemble}.txt", 'w') as f:
        f.write("\n\n".join(intermediate))


if __name__ == "__main__":
    main()
    if args.evaluate_accuracy:
        evaluate_result = os.path.join(preprocessed_videos_base, f"{task}/evaluation_results{args.voting_ensemble}.txt")
        evaluate_cmd = [
            'python', "evaluate_script.py", output_dir / f"predictions_{task}{args.voting_ensemble}.json",
            args.hd_epic_vqa_annotations, evaluate_result, args.voting_ensemble
        ]
        try:
            subprocess.run(evaluate_cmd, check=True, stderr=subprocess.PIPE)

        except subprocess.CalledProcessError as e:
            print(e.stderr.decode())
import json
import re
import argparse
import subprocess
from pathlib import Path
from datetime import timedelta
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import os
from tqdm import tqdm
import cv2
import base64
import numpy as np
import datetime

# 配置参数解析
parser = argparse.ArgumentParser()
parser.add_argument("--video_segment", type=str, default="full",
                   choices=["full", "segment", "interval"],
                   help="使用完整视频、截取片段或指定时间区间")
parser.add_argument("--segment_duration", type=int, default=10,
                   help="截取片段的持续时间（前后各多少秒）")
parser.add_argument("--task", type=str, default="ingredient_exact_ingredient_recognition",
                   choices=["ingredient_exact_ingredient_recognition", "ingredient_ingredient_recognition",
                            "recipe_recipe_recognition", "recipe_prep_localization",
                            "recipe_step_localization"
                           ])
parser.add_argument("--video_merge_strategy", type=str, default="multi_input",
                   choices=["concat", "multi_input"],
                   help="合并视频策略：拼接(concat)或多输入(multi_input)")
parser.add_argument("--wobbox", action='store_true')

parser.add_argument("--start_index", type=int, default=0,
                   help="从问题列表中的哪个索引（包含）开始处理，0-based")
parser.add_argument("--end_index", type=int, default=None,
                   help="到问题列表中的哪个索引（不包含）结束处理。如果为None，则处理到列表末尾")
# CHANGED: Default paths relative
parser.add_argument("--hd_epic_vqa_annotations", type=str, default="./hd-epic-annotations/vqa-benchmark/")
parser.add_argument("--hd_epic_database", type=str, default="./HD-EPIC/")
parser.add_argument("--inference_percentage", type=int, default=100,
                    help="Percentage of samples to use for inference (0-100). Default is 100.")
parser.add_argument("--evaluate_accuracy", action='store_true')
parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
parser.add_argument("--print_intermediate", action="store_true",
                    help="Print intermediate question and prediction for each sample.")
parser.add_argument("--task_name", type=str, default="PreprocessedVideos")
parser.add_argument("--voting_ensemble", type=str, default="")
# CHANGED: Added temp dir argument
parser.add_argument("--temp_dir", type=str, default="/tmp/qwen_video_segments",
                    help="Temporary directory for video processing")

args = parser.parse_args()

task = args.task
model_name = args.base_model
HD_EPIC_database = args.hd_epic_database
preprocessed_videos_base = os.path.join(HD_EPIC_database, args.task_name)

# 配置路径
if args.wobbox:
    questions_path = os.path.join(preprocessed_videos_base, f"{task}/{task}{args.voting_ensemble}.json")
else:
    questions_path = os.path.join(preprocessed_videos_base, f"{task}/reformatted_questions_with_narration_video{args.voting_ensemble}.json")
output_dir = Path(os.path.join(preprocessed_videos_base, f"{task}/"))
TEMP_VIDEO_DIR = Path(args.temp_dir)
TEMP_VIDEO_DIR.mkdir(parents=True, exist_ok=True)


def concatenate_videos(input_paths, output_path):
    """拼接多个视频文件并返回各视频时长信息"""
    durations = [get_video_duration(p) for p in input_paths]

    now = datetime.datetime.now()
    timestamp_str = now.strftime("%Y%m%d_%H%M%S_%f")

    list_file = TEMP_VIDEO_DIR / f"concat_list_{timestamp_str}.txt"
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
        raise RuntimeError(f"拼接视频失败: {result.stderr.decode()}")
    return output_path, durations


def process_multi_videos(q_data, args):
    """处理多视频输入"""
    video_entries = [v for k, v in q_data['inputs'].items() if k.startswith('video ')]
    processed_paths = []
    temp_files = []
    time_offset = 0

    for idx, video_info in enumerate(video_entries):
        video_id = video_info['id']
        dir_part = video_id.split('-')[0]
        # CHANGED: Replaced hardcoded path with os.path.join and args.hd_epic_database
        original_path = os.path.join(args.hd_epic_database, "Videos", dir_part, f"{video_id}.mp4")

        if args.video_segment == "segment":
            path, offset = process_video_segment(original_path, q_data['question'], args.segment_duration)
            if idx == 0:
                time_offset = offset
        elif args.video_segment == "interval":
            if 'start_time' not in video_info or 'end_time' not in video_info:
                raise ValueError("视频信息缺少时间区间")
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


def adjust_question_for_multi_video(question, time_offset):
    """调整多视频问题的提问方式"""
    return re.sub(
        r"(shown in )(\d+:\d+:\d+\.\d+)",
        f"\\g<1>{seconds_to_time_str(time_offset)}",
        question
    ) if time_offset > 0 else question


def build_video_content(video_paths, merge_strategy, q_id):
    """构建视频内容并计算时间偏移"""
    if merge_strategy == "concat" and len(video_paths) > 1:
        output_path = TEMP_VIDEO_DIR / f"concat_{q_id}.mp4"
        concat_path, durations = concatenate_videos(video_paths, output_path)

        time_offsets = [sum(durations[:i]) for i in range(len(durations))]
        return [concat_path], [concat_path], time_offsets
    return video_paths, [], [0]


def load_model():
    """加载Qwen模型和处理器"""
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def seconds_to_time_str(seconds):
    """将秒数转换为HH:MM:SS.sss格式"""
    td = timedelta(seconds=seconds)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = td.microseconds // 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def parse_choice_time(choice_str):
    """从选项字符串中解析时间区间"""
    time_matches = re.findall(r'\d+:\d+:\d+\.\d+', choice_str)
    if len(time_matches) == 2:
        return parse_time(time_matches[0]), parse_time(time_matches[1])
    return None, None


def process_video_interval(original_path, start_time, end_time):
    """处理指定时间区间的视频片段"""
    duration = end_time - start_time
    output_path = TEMP_VIDEO_DIR / f"{Path(original_path).stem}_interval_{start_time:.1f}-{end_time:.1f}.mp4"

    if not output_path.exists():
        trim_video(original_path, output_path, start_time, duration)

    return output_path, duration


def adjust_choices_time(choices, time_offsets):
    """将选项中的分段时间转换为拼接后的绝对时间"""
    adjusted = []
    for choice in choices:
        prefix, content = re.match(r"([A-E]\.\s*)(.*)", choice).groups()

        time_segments = re.findall(r'\[V(\d+)\]\s*(\d+:\d+:\d+\.\d+)\s*-\s*(\d+:\d+:\d+\.\d+)', content)
        if time_segments:
            converted_times = []
            for vid_idx, start_str, end_str in time_segments:
                vid_idx = int(vid_idx) - 1
                offset = time_offsets[vid_idx] if vid_idx < len(time_offsets) else 0

                start = parse_time(start_str) + offset
                end = parse_time(end_str) + offset

                converted_times.append(
                    f"{seconds_to_time_str(start)} - {seconds_to_time_str(end)}"
                )
        else:
            return choices

        new_content = "; ".join(converted_times)
        adjusted.append(f"{prefix}{new_content}")

    return adjusted


def parse_time(time_str):
    """将时间字符串转换为秒数"""
    try:
        h, m, s = time_str.split(':')
        return int(h) * 3600 + int(m) * 60 + float(s)
    except:
        raise ValueError(f"Invalid time format: {time_str}")


def extract_target_time(question_text):
    """从问题中提取目标时间点"""
    match = re.search(r"shown in (\d+:\d+:\d+\.\d+)", question_text)
    if not match:
        return None
    return parse_time(match.group(1))


def trim_video(input_path, output_path, start_time, duration):
    """使用ffmpeg修剪视频"""
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
    """获取视频总时长（秒）"""
    result = subprocess.run([
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        input_path
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return float(result.stdout)


def process_video_segment(original_path, question_text, segment_duration):
    """处理视频片段并返回(处理后的路径, 时间偏移量)"""
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
    """调整问题中的时间描述"""
    adjusted_time = str(timedelta(seconds=new_time)).split('.')[0] + f".{int(new_time%1*1000):03d}"
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

    predictions = {}
    intermediate = []

    for q_id, q_data in tqdm(question_items_to_process[0:num_samples_to_process], desc="Processing questions"):
        try:
            video_paths, temp_video_files, _ = process_multi_videos(q_data, args)

            final_videos, concat_files, time_offsets = build_video_content(
                video_paths, args.video_merge_strategy, q_id
            )

            adjusted_choices = adjust_choices_time(q_data.get('choices', []), time_offsets)
            adjusted_question = q_data['question'] + "\n".join(adjusted_choices)

            content = []
            for video_path in final_videos:
                content.append({"type": "video", "video": str(video_path), "fps": 1.0})
            content.append({"type": "text", "text": adjusted_question})

            messages = [
                {"role": "system", "content": "You are an expert video analyzer. Answer the multiple choice question by giving ONLY the letter identifying the answer. Example: A or B or C or D or E. You MUST answer even if unsure."},
                {"role": "user", "content": content}
            ]

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

            intermediate.append(f"{q_id}\nQuestion: {q_data['question']}\nOutput: {output_text}\n{'=' * 50}")
            if args.print_intermediate:
                print(f"{q_id}\nQuestion: {content}\nOutput: {output_text}\n{'=' * 50}")

            predictions[q_id] = output_text

            for f in temp_video_files + concat_files:
                if Path(f).exists():
                    os.remove(f)

        except Exception as e:
            print(f"Error processing {q_id}: {str(e)}")
            predictions[q_id] = -1
            intermediate.append(f"{q_id}\nError: {str(e)}\n{'=' * 50}")

    with open(output_dir / f"predictions_{task}{args.voting_ensemble}.json", 'w') as f:
        json.dump(predictions, f, indent=2)

    with open(output_dir / f"intermediate_results{args.voting_ensemble}.txt", 'w') as f:
        f.write("\n\n".join(intermediate))


if __name__ == "__main__":
    main()

    if args.evaluate_accuracy:
        evaluate_result = os.path.join(preprocessed_videos_base, f"{task}/evaluation_results{args.voting_ensemble}.txt")
        evaluate_cmd = [
            'python', "evaluate_script.py", str(output_dir / f"predictions_{task}{args.voting_ensemble}.json"), args.hd_epic_vqa_annotations, evaluate_result, args.voting_ensemble
        ]
        try:
            subprocess.run(evaluate_cmd, check=True, stderr=subprocess.PIPE)

        except subprocess.CalledProcessError as e:
            print(e.stderr.decode())
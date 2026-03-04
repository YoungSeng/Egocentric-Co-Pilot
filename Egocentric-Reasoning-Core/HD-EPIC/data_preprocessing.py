import os
import json
import re
import subprocess
from datetime import datetime
import argparse


task_list = ["3d_perception_fixture_location", "3d_perception_object_contents_retrieval",
             "3d_perception_fixture_interaction_counting", "3d_perception_object_location",
             "fine_grained_action_localization", "fine_grained_action_recognition",
             "fine_grained_how_recognition", "fine_grained_why_recognition",
             "gaze_gaze_estimation", "gaze_interaction_anticipation", "ingredient_ingredient_adding_localization",
             "ingredient_ingredient_retrieval", "ingredient_ingredient_weight",
             "nutrition_image_nutrition_estimation", "ingredient_ingredients_order",
             "nutrition_nutrition_change", "nutrition_video_nutrition_estimation",
             "object_motion_object_movement_counting", "object_motion_object_movement_itinerary",
             "recipe_step_recognition", "recipe_rough_step_localization", "recipe_multi_recipe_recognition",
             "object_motion_stationary_object_localization", "recipe_following_activity_recognition",
             "recipe_multi_step_localization", "ingredient_exact_ingredient_recognition",
             "ingredient_ingredient_recognition", "recipe_recipe_recognition", "recipe_prep_localization",
             "recipe_step_localization"
             ]

# 配置参数解析
parser = argparse.ArgumentParser()
parser.add_argument("--hd_epic_vqa_annotations", type=str, default="./dataset/hd-epic-annotations/vqa-benchmark/")
parser.add_argument("--hd_epic_database", type=str, default="./dataset/HD-EPIC/")

parser.add_argument("--task", type=str, default="3d_perception_fixture_location",
                    choices=task_list
                    )
parser.add_argument("--extract_keyframe", action='store_true')
parser.add_argument("--generate_refined_prompt", action='store_true')
parser.add_argument("--wo_refined_prompt", action='store_true')
parser.add_argument("--wo_refined_answer", action='store_true')
parser.add_argument("--refined_question_later", action='store_true')
parser.add_argument("--print_intermediate", action="store_true",
                    help="Print intermediate question and prediction for each sample.")
parser.add_argument("--task_name", type=str, default="PreprocessedVideos")
parser.add_argument("--voting_ensemble", type=str, default="")
args = parser.parse_args()

# 配置路径
HD_EPIC_database = args.hd_epic_database
json_dir = args.hd_epic_vqa_annotations

videos_base = os.path.join(HD_EPIC_database, "Videos")
preprocessed_videos_base = os.path.join(HD_EPIC_database, args.task_name)

# 收集所有JSON文件路径
json_files = [os.path.join(json_dir, f)
              for f in os.listdir(json_dir)
              if f.endswith('.json')]

def convert_time_to_seconds(time_str):
    """将时间字符串转换为秒数"""
    try:
        time_obj = datetime.strptime(time_str, "%H:%M:%S.%f")
        return time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second + time_obj.microsecond / 1e6
    except ValueError:
        return None


def index_to_letter(idx):
    """将数字索引转换为大写字母 (0->A, 1->B, ...)"""
    if 0 <= idx < 26: # Support up to 26 choices
        return chr(ord('A') + idx)
    return str(idx) # Fallback for more than 26 choices or invalid index

def letter_to_index(letter):
    """将大写字母转换为数字索引 (A->0, B->1, ...)"""
    if len(letter) == 1 and 'A' <= letter <= 'Z':
        return ord(letter) - ord('A')
    return -1 # Not a single valid letter A-Z


def process_video_frame(video_path, timestamp, output_path):
    """使用ffmpeg截取视频帧"""
    ffmpeg_cmd = [
        'ffmpeg',
        '-ss', str(timestamp),
        '-i', video_path,
        '-vframes', '1',
        '-q:v', '1',        # 2
        '-y',
        output_path
    ]
    try:
        subprocess.run(ffmpeg_cmd, check=True, stderr=subprocess.PIPE)
        return True, ""
    except subprocess.CalledProcessError as e:
        return False, e.stderr.decode()


for json_file in json_files:
    task_name = os.path.splitext(os.path.basename(json_file))[0]

    if task_name not in task_list or task_name != args.task:
        continue

    print(f"\nProcessing task: {task_name}")

    # 创建输出目录
    task_output_dir = os.path.join(preprocessed_videos_base, task_name)
    os.makedirs(task_output_dir, exist_ok=True)
    if args.extract_keyframe:
        images_dir = os.path.join(task_output_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

    # 初始化日志文件
    log_file = os.path.join(task_output_dir, "processing_log.txt")
    error_log = []
    new_qa_data = {}

    # 读取原始数据
    with open(json_file, "r") as f:
        original_data = json.load(f)

    for qid, qdata in original_data.items():
        log_entry = []
        log_entry.append(f"Processing: {qid}")

        try:
            if args.extract_keyframe:

                # 生成输出路径
                output_image = os.path.join(images_dir, f"{qid}.png")

                if os.path.exists(output_image):
                    if args.print_intermediate:
                        print(f"Skipping {qid}: {output_image} already exists")
                else:
                    # 获取视频ID
                    video_id = qdata["inputs"]["video 1"]["id"]
                    log_entry.append(f"Video ID: {video_id}")

                    # 构建视频路径
                    participant = video_id.split("-")[0]
                    video_path = os.path.join(videos_base, participant, f"{video_id}.mp4")
                    log_entry.append(f"Expected video path: {video_path}")

                    if not os.path.exists(video_path):
                        raise FileNotFoundError("Video file not found")

                    # 解析时间戳
                    time_match = re.search(r"<TIME ([\d:.]+) video 1>", qdata["question"])
                    if not time_match:
                        raise ValueError("No valid timestamp found in question")

                    timestamp_str = time_match.group(1)
                    timestamp = convert_time_to_seconds(timestamp_str)
                    if timestamp is None:
                        raise ValueError(f"Invalid timestamp format: {timestamp_str}")
                    log_entry.append(f"Timestamp: {timestamp_str} -> {timestamp:.3f}s")

                    # 截取视频帧
                    success, error = process_video_frame(video_path, timestamp, output_image)
                    if not success:
                        raise RuntimeError(f"Frame extraction failed: {error}")
                    log_entry.append(f"Frame saved to: {output_image}")

            if task_name == "3d_perception_fixture_location":
                # 构建新问题
                object_match = re.search(r"where is the (.*?) located\?", qdata["question"])
                if not object_match:
                    raise ValueError("Could not extract object name from question")

                object_name = object_match.group(1)
                choices_str = "\n".join([f"{index_to_letter(idx)}: {choice}" for idx, choice in enumerate(qdata["choices"])])

                new_question = (
                    f"This image shows my current viewpoint. Based on this image, "
                    f"please determine the direction of the {object_name} relative to my viewpoint "
                    f"and choose the closest answer from the options below: \n{choices_str}"
                )

                if args.voting_ensemble == "1":
                    new_question = (
                        f"Looking at this image, which represents my viewpoint, "
                        f"where is the {object_name} located relative to my position? "
                        f"Please select the closest option from the following: \n{choices_str}"
                    )
                elif args.voting_ensemble == "2":
                    new_question = (
                        f"Given this image shows my viewpoint, "
                        f"how is the {object_name} positioned in relation to me? "
                        f"Choose the nearest answer from the options below: \n{choices_str}"
                    )
                elif args.voting_ensemble == "3":
                    new_question = (
                        f"Please identify the direction of the {object_name} "
                        f"as seen from the viewpoint depicted in this image. "
                        f"Select the most accurate option from the choices provided: \n{choices_str}"
                    )
                elif args.voting_ensemble == "4":
                    new_question = (
                        f"Using this image of my viewpoint, "
                        f"what is the relative direction of the {object_name}? "
                        f"Pick the best fit from the choices listed below: \n{choices_str}"
                    )

                # 构建新条目
                new_qa_data[qid] = {
                    "inputs": qdata["inputs"],
                    "question": new_question,
                    "correct_idx": qdata["correct_idx"]
                }
            if task_name == "3d_perception_object_contents_retrieval":
                # 匹配操作类型和bbox坐标（直接从问题字符串提取）
                pattern = r"did the person (put in\/on|take from) the item indicated by bounding box <BBOX ([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+)>"
                match = re.search(pattern, qdata["question"], re.IGNORECASE)

                if not match:
                    raise ValueError("Failed to parse contents retrieval question")

                # 提取关键信息
                action_type = match.group(1).lower()  # "put in/on" 或 "take from"
                bbox_coords = [int(float(coord)) for coord in match.groups()[1:]]  # 转换坐标为整数

                # 生成自然语言描述
                action_desc = "placed into" if "put" in action_type else "removed from"
                bbox_note = f"(Bounding Box coordinates: {', '.join(map(str, bbox_coords))})"

                choices_str = "\n".join([f"{index_to_letter(idx)}: {choice}" for idx, choice in enumerate(qdata["choices"])])


                # 构建面向视觉理解的prompt
                new_question = (
                    f"Based on the image showing a household scene, focus on the area marked {bbox_note}. "
                    f"What objects did the person {action_desc} this container? "
                    f"Choose the most accurate option:\n{choices_str}"
                )

                if args.voting_ensemble == "1":
                    new_question = (
                        f"In the household scene image, look at the area marked {bbox_note}. "
                        f"Using the container shown there, what objects did the person {action_desc}? "
                        f"Please choose the most accurate option:\n{choices_str}"
                    )
                if args.voting_ensemble == "2":
                    new_question = (
                        f"Based on the image of the household scene, concentrate on the area marked {bbox_note}. "
                        f"Regarding the action {action_desc} performed by the person with the container in this area, "
                        f"what objects were involved? Select the most accurate option:\n{choices_str}"
                    )
                if args.voting_ensemble == "3":
                    new_question = (
                        f"View the household scene image and the area marked {bbox_note}. "
                        f"The person performed the action {action_desc} using the container shown. "
                        f"Identify the objects that were affected or used in this process. "
                        f"Choose the most accurate option:\n{choices_str}"
                    )
                if args.voting_ensemble == "4":
                    new_question = (
                        f"Focus on the area marked {bbox_note} within this household scene image. "
                        f"Considering the person's action {action_desc} involving the container in this area, "
                        f"which objects were the focus of this action? Choose the most accurate option:\n{choices_str}"
                    )

                # 构建新条目（统一结构）
                new_qa_data[qid] = {
                    "inputs": qdata["inputs"],
                    "question": new_question,
                    "correct_idx": qdata["correct_idx"]
                }
            if task_name == "fine_grained_action_localization":
                # 提取动作名称
                action_match = re.search(r"the action <(.*?)> happen", qdata["question"])
                if not action_match:
                    raise ValueError("Could not extract action name from question")

                action_name = action_match.group(1)

                # 重构问题描述
                new_question = (
                    f"When did the action <{action_name}> occur? Choose the exact time interval from the options below: \n"
                )
                if args.voting_ensemble == "1":
                    new_question = (
                        f"Please identify the exact time period when the action <{action_name}> took place. "
                        f"Select the correct interval from the options provided below: \n"
                    )
                if args.voting_ensemble == "2":
                    new_question = (
                        f"From the options listed below, select the exact time interval during which the action <{action_name}> occurred: \n"
                    )
                if args.voting_ensemble == "3":
                    new_question = (
                        f"What is the exact time interval for the action <{action_name}>? "
                        f"Choose the most accurate option from below: \n"
                    )
                if args.voting_ensemble == "4":
                    new_question = (
                        f"Indicate the exact time interval corresponding to the action <{action_name}>. "
                        f"Choose from the options below: \n"
                    )

                # 处理选项
                processed_choices = []
                for idx, choice in enumerate(qdata["choices"]):
                    # 提取时间区间（兼容两种格式）
                    time_match = re.search(
                        r"(\d{2}:\d{2}:\d{2}\.\d{3}).*?(\d{2}:\d{2}:\d{2}\.\d{3})",
                        choice
                    )
                    if not time_match:
                        raise ValueError(f"Invalid time format in choice: {choice}")

                    # 格式化选项
                    letter = index_to_letter(idx)
                    processed_choice = f"{letter}. {time_match.group(1)} - {time_match.group(2)}"
                    processed_choices.append(processed_choice)

                # 重构输入结构（单数video）
                new_inputs = {
                    "video": {
                        "id": qdata["inputs"]["video 1"]["id"],
                        "start_time": qdata["inputs"]["video 1"]["start_time"],
                        "end_time": qdata["inputs"]["video 1"]["end_time"]
                    }
                }

                # 构建新条目
                new_qa_data[qid] = {
                    "inputs": new_inputs,
                    "question": new_question,
                    "choices": processed_choices,
                    "correct_idx": qdata["correct_idx"]
                }
            if task_name == "fine_grained_action_recognition":

                choices_str = "\n".join(
                    [f"{index_to_letter(idx)}: {choice}" for idx, choice in enumerate(qdata["choices"])])

                new_inputs = {
                    "video": {
                        "id": qdata["inputs"]["video 1"]["id"],
                        "start_time": qdata["inputs"]["video 1"]["start_time"],
                        "end_time": qdata["inputs"]["video 1"]["end_time"]
                    }
                }

                new_question = (
                    f"Which statement most accurately describes the primary action(s) occurring in the video? "
                    f"Choose the best option: \n{choices_str}"
                )
                if args.voting_ensemble == "1":
                    new_question = (
                        f"Based on the video content, what is the main activity or action(s) taking place? "
                        f"Choose the statement below that best identifies it: \n{choices_str}"
                    )
                if args.voting_ensemble == "2":
                    new_question = (
                        f"Which statement below provides the most accurate summary of the primary action(s) shown in the video? "
                        f"Select the best option: \n{choices_str}"
                    )
                if args.voting_ensemble == "3":
                    new_question = (
                        f"Looking at this video, which of the following statements is the most accurate description of the primary action(s)? "
                        f"Choose the best fit: \n{choices_str}"
                    )
                if args.voting_ensemble == "4":
                    new_question = (
                        f"From the choices provided below, select the statement that most accurately represents the primary action(s) seen in the video: \n{choices_str}"
                    )

                new_qa_data[qid] = {
                    "inputs": new_inputs,
                    "question": new_question,
                    # "choices": qdata["choices"],
                    # "correct_idx": qdata["correct_idx"]
                }
            if task_name == "fine_grained_how_recognition":
                # 提取动作名称
                action_match = re.search(r"the action <(.*?)>", qdata["question"])
                if not action_match:
                    raise ValueError("Could not extract action name from question")
                action_name = action_match.group(1)

                # 重构问题描述
                new_question = (
                    f"Choose the most mechanically precise description of how <{action_name}> was performed: \n"
                )
                if args.voting_ensemble == "1":
                    new_question = (
                        f"Which statement below offers the most mechanically precise account of the manner in which <{action_name}> was carried out? "
                        f"Choose the best option: \n"
                    )
                if args.voting_ensemble == "2":
                    new_question = (
                        f"From the options provided, select the description that is most mechanically accurate in explaining how <{action_name}> was executed. "
                        f"Choose the best fit: \n"
                    )
                if args.voting_ensemble == "3":
                    new_question = (
                        f"Which option below provides the most technically detailed description of the performance of the action <{action_name}>? "
                        f"Select the best one: \n"
                    )
                if args.voting_ensemble == "4":
                    new_question = (
                        f"What is the most mechanically precise way to describe how the action <{action_name}> was performed? "
                        f"Choose the best option from below: \n"
                    )

                # 处理选项
                processed_choices = []
                for idx, choice in enumerate(qdata["choices"]):
                    # 标准化描述结构
                    clean_choice = re.sub(r"^(by|using)\s+", "", choice).strip().capitalize()

                    # 提取核心动词短语
                    verb_match = re.search(r"(\b\w+ing\b|\b\w+ed\b)", clean_choice)
                    if verb_match:
                        core_verb = verb_match.group(1)

                    # 构建字母序号选项
                    letter = index_to_letter(idx)
                    processed_choice = f"{letter}. {clean_choice.rstrip('.')}"  # 统一去除结尾标点

                    processed_choices.append(processed_choice)

                new_inputs = {
                    "video": {
                        "id": qdata["inputs"]["video 1"]["id"],
                        "start_time": qdata["inputs"]["video 1"]["start_time"],
                        "end_time": qdata["inputs"]["video 1"]["end_time"]
                    }
                }

                # 构建新条目
                new_qa_data[qid] = {
                    "inputs": new_inputs,
                    "question": new_question,
                    "choices": processed_choices,
                }
            if task_name == "fine_grained_why_recognition":
                # 提取动作名称
                action_match = re.search(r"the action <(.*?)>", qdata["question"])
                if not action_match:
                    raise ValueError("Could not extract action name from question")
                action_name = action_match.group(1)

                # 重构问题描述
                new_question = (
                    f"Choose the most mechanically precise description of why <{action_name}> was performed: \n"
                )
                if args.voting_ensemble == "1":
                    new_question = (
                        f"Which explanation below is the most mechanically precise regarding the reason why the action <{action_name}> was performed? "
                        f"Choose the best option: \n"
                    )
                if args.voting_ensemble == "2":
                    new_question = (
                        f"What was the underlying reason or objective for performing <{action_name}>? "
                        f"Select the statement below that provides the most mechanically precise description of this purpose: \n"
                    )
                if args.voting_ensemble == "3":
                    new_question = (
                        f"Which statement most accurately and mechanically precisely describes the reason for the action <{action_name}>? "
                        f"Choose the best fit: \n"
                    )
                if args.voting_ensemble == "4":
                    new_question = (
                        f"From the options provided, select the most mechanically precise justification or explanation for why the action <{action_name}> was carried out: \n"
                    )

                # 处理选项
                processed_choices = []
                for idx, choice in enumerate(qdata["choices"]):
                    # 标准化描述结构
                    clean_choice = re.sub(r"^(by|using)\s+", "", choice).strip().capitalize()

                    # 提取核心动词短语
                    verb_match = re.search(r"(\b\w+ing\b|\b\w+ed\b)", clean_choice)
                    if verb_match:
                        core_verb = verb_match.group(1)

                    # 构建字母序号选项
                    letter = index_to_letter(idx)
                    processed_choice = f"{letter}. {clean_choice.rstrip('.')}"  # 统一去除结尾标点

                    processed_choices.append(processed_choice)

                new_inputs = {
                    "video": {
                        "id": qdata["inputs"]["video 1"]["id"],
                        "start_time": qdata["inputs"]["video 1"]["start_time"],
                        "end_time": qdata["inputs"]["video 1"]["end_time"]
                    }
                }

                # 构建新条目
                new_qa_data[qid] = {
                    "inputs": new_inputs,
                    "question": new_question,
                    "choices": processed_choices,
                }

            if task_name == "gaze_gaze_estimation":
                # 重构问题描述
                new_question = "Select the most likely focus of visual attention based on head orientation and eye gaze: \n"

                if args.voting_ensemble == "1":
                    new_question = (
                        f"Based on the observed head orientation and eye gaze, what is the most probable focus of visual attention? "
                        f"Choose the best option below: \n"
                    )
                if args.voting_ensemble == "2":
                    new_question = (
                        f"Using the head orientation and eye gaze as clues, identify the most likely object or area of visual attention. "
                        f"Select the best choice: \n"
                    )
                if args.voting_ensemble == "3":
                    new_question = (
                        f"Infer the most likely point of visual attention from the head orientation and eye gaze. "
                        f"Choose the most appropriate option: \n"
                    )
                if args.voting_ensemble == "4":
                    new_question = (
                        f"Considering the head orientation and eye gaze, which option represents the most likely focus of visual attention? "
                        f"Select the best answer: \n"
                    )

                # 处理选项
                processed_choices = []
                for idx, choice in enumerate(qdata["choices"]):
                    # 添加字母序号
                    letter = index_to_letter(idx)
                    processed_choice = f"{letter}. {choice.rstrip('.')}"

                    processed_choices.append(processed_choice)

                new_inputs = {
                    "video": {
                        "id": qdata["inputs"]["video 1"]["id"],
                        "start_time": qdata["inputs"]["video 1"]["start_time"],
                        "end_time": qdata["inputs"]["video 1"]["end_time"]
                    }
                }

                # 构建新条目
                new_qa_data[qid] = {
                    "inputs": new_inputs,
                    "question": new_question,
                    "choices": processed_choices,
                }

            if task_name == "gaze_interaction_anticipation":
                # 重构问题描述
                new_question = "Predict the next physical interaction target based on movement trajectory and visual cues: \n"

                if args.voting_ensemble == "1":
                    new_question = (
                        f"Using the movement trajectory and visual cues, anticipate the most likely object or area the person will physically interact with next. "
                        f"Choose the best option below: \n"
                    )
                if args.voting_ensemble == "2":
                    new_question = (
                        f"From the movement trajectory and visual cues, infer the most probable target of the next physical interaction. "
                        f"Choose the most appropriate option: \n"
                    )
                if args.voting_ensemble == "3":
                    new_question = (
                        f"Considering the movement trajectory and visual cues, what is the most likely target for the next physical interaction? "
                        f"Choose the best fit: \n"
                    )
                if args.voting_ensemble == "4":
                    new_question = (
                        f"Identify the most likely target of the next physical interaction, based on the movement trajectory and visual cues. "
                        f"Select the best choice from the options below: \n"
                    )

                # 处理选项
                processed_choices = []
                for idx, choice in enumerate(qdata["choices"]):
                    # 添加字母序号
                    letter = index_to_letter(idx)
                    processed_choice = f"{letter}. {choice.rstrip('.')}"

                    processed_choices.append(processed_choice)

                new_inputs = {
                    "video": {
                        "id": qdata["inputs"]["video 1"]["id"],
                        "start_time": qdata["inputs"]["video 1"]["start_time"],
                        "end_time": qdata["inputs"]["video 1"]["end_time"]
                    }
                }

                # 构建新条目
                new_qa_data[qid] = {
                    "inputs": new_inputs,
                    "question": new_question,
                    "choices": processed_choices,
                }

            if task_name == "ingredient_ingredient_adding_localization":
                # 提取关键信息
                ingredient_match = re.search(
                    r"ingredient\s+([\w\s\-\(\)]+?)\s+added\s+to\s+recipe\s+([\w\s\-,\(\)]+?)\?",
                    qdata["question"],
                    re.IGNORECASE
                )
                if not ingredient_match:
                    raise ValueError("Could not extract ingredient and recipe name")

                ingredient, recipe = ingredient_match.groups()

                # 重构问题描述
                new_question = (
                    f"Identify the precise time interval when {ingredient} was added during "
                    f"the preparation of {recipe}. Select the exact timeframe: \n"
                )

                if args.voting_ensemble == "1":
                    new_question = (
                        f"During the preparation of {recipe}, determine the precise time interval when {ingredient} was added. "
                        f"Choose the exact timeframe from the options below: \n"
                    )
                if args.voting_ensemble == "2":
                    new_question = (
                        f"Which of the following time intervals most accurately indicates when {ingredient} was added during the preparation of {recipe}? "
                        f"Select the best option: \n"
                    )
                if args.voting_ensemble == "3":
                    new_question = (
                        f"What is the precise timeframe during the preparation of {recipe} at which {ingredient} was introduced? "
                        f"Choose the exact time interval from the options below: \n"
                    )
                if args.voting_ensemble == "4":
                    new_question = (
                        f"Select the precise time interval from the options below that corresponds to the moment {ingredient} was added during the preparation of {recipe}: \n"
                    )

                # 处理选项
                processed_choices = []
                for idx, choice in enumerate(qdata["choices"]):
                    # 提取标准化时间
                    time_match = re.findall(r"\d{2}:\d{2}:\d{2}\.\d{3}", choice)
                    if len(time_match) != 2:
                        raise ValueError(f"Invalid time format in choice: {choice}")

                    # 格式化为紧凑时间区间
                    start = time_match[0][:-4]  # 保留到秒级精度
                    end = time_match[1][:-4]
                    letter = index_to_letter(idx)
                    processed_choice = f"{letter}. {start} - {end}"

                    processed_choices.append(processed_choice)

                # 重构输入结构（单数video）
                new_inputs = {
                    "video": {
                        "id": qdata["inputs"]["video 1"]["id"],
                    }
                }

                # 构建新条目
                new_qa_data[qid] = {
                    "inputs": new_inputs,
                    "question": new_question,
                    "choices": processed_choices,
                }

            if task_name == "ingredient_ingredient_retrieval":
                # 提取时间区间
                time_pattern = r"Between <TIME (\d{2}:\d{2}:\d{2}\.\d{3}) video 1> and <TIME (\d{2}:\d{2}:\d{2}\.\d{3}) video 1>"
                time_match = re.search(time_pattern, qdata["question"])
                if not time_match:
                    raise ValueError("Could not extract time interval from question")

                # 重构输入结构
                new_inputs = {
                    "video": {
                        "id": qdata["inputs"]["video 1"]["id"],
                        "start_time": time_match.group(1),
                        "end_time": time_match.group(2),
                    }
                }

                # 构建新问题
                new_question = "Which ingredients were added to the dish during the specified time interval? Choose the best option: \n"

                if args.voting_ensemble == "1":
                    new_question = (
                        f"Identify which ingredients were added to the dish during the specified time interval. "
                        f"Choose the best option below: \n"
                    )
                if args.voting_ensemble == "2":
                    new_question = (
                        f"During the specified time interval, which ingredients were added to the dish? "
                        f"Select the most accurate option: \n"
                    )
                if args.voting_ensemble == "3":
                    new_question = (
                        f"Considering the specified time interval, what ingredients were added to the dish? "
                        f"Choose the best fit from the options: \n"
                    )
                if args.voting_ensemble == "4":
                    new_question = (
                        f"From the options below, select the one that correctly lists the ingredients added to the dish during the specified time interval: \n"
                    )

                # 处理选项（标准化格式）
                processed_choices = [f"{index_to_letter(idx)}. {choice.capitalize()}"
                                     for idx, choice in enumerate(qdata["choices"])]

                # 构建新条目
                new_qa_data[qid] = {
                    "inputs": new_inputs,
                    "question": new_question,
                    "choices": processed_choices,
                }

            if task_name == "ingredient_ingredient_weight":
                # 提取测量对象
                item_match = re.search(r"weigh of (.*?)\?", qdata["question"])
                if not item_match:
                    raise ValueError("Could not extract measured item from question")
                item_name = item_match.group(1).strip()

                # 重构问题描述
                new_question = (
                    f"What was the precise measurement result for {item_name}? "
                    f"Select the exact numerical value: \n"
                )

                if args.voting_ensemble == "1":
                    new_question = (
                        f"Identify the exact numerical value of the measurement result for {item_name}. "
                        f"Choose from the options below: \n"
                    )
                if args.voting_ensemble == "2":
                    new_question = (
                        f"What is the precise numerical value obtained from measuring {item_name}? "
                        f"Select the exact result from below: \n"
                    )
                if args.voting_ensemble == "3":
                    new_question = (
                        f"For {item_name}, what was the precise numerical measurement? "
                        f"Choose the exact value: \n"
                    )
                if args.voting_ensemble == "4":
                    new_question = (
                        f"Select the precise numerical value that represents the measurement result for {item_name} from the options below: \n"
                    )

                # 标准化选项格式
                processed_choices = []
                for idx, choice in enumerate(qdata["choices"]):
                    # 统一数字和单位格式（确保无空格）
                    clean_choice = re.sub(r"(\d+)\s*([a-zA-Z]+)", r"\1\2", choice.strip())

                    # 添加字母序号
                    letter = index_to_letter(idx)
                    processed_choice = f"{letter}. {clean_choice}"

                    processed_choices.append(processed_choice)

                new_inputs = {
                    "video": {
                        "id": qdata["inputs"]["video 1"]["id"],
                    }
                }

                # 构建新条目
                new_qa_data[qid] = {
                    "inputs": new_inputs,
                    "question": new_question,
                    "choices": processed_choices,
                }

            if task_name == "nutrition_image_nutrition_estimation":
                # 提取营养成分类型（protein/carbs/calories等）
                nutrient_match = re.search(r"showcase higher (\w+)\?", qdata["question"])
                if not nutrient_match:
                    raise ValueError("Could not identify nutrient type in question")

                nutrient = nutrient_match.group(1).lower()

                # 构建专业术语映射
                nutrient_terms = {
                    "protein": "protein content",
                    "carbs": "carbohydrate content",
                    "calories": "caloric value"
                }

                # 获取专业术语（默认为原词）
                nutrient_term = nutrient_terms.get(nutrient, f"{nutrient} content")

                # 重构问题描述
                new_question = (
                    f"Based on images and standard nutritional values, which ingredient has the highest {nutrient_term}? "
                    f"Choose the best option: \n"
                )

                if args.voting_ensemble == "1":
                    new_question = (
                        f"Considering the images and standard nutritional data, identify the ingredient that is richest in {nutrient_term}. "
                        f"Select the best option from below: \n"
                    )
                if args.voting_ensemble == "2":
                    new_question = (
                        f"According to the images and standard nutritional information, which ingredient contains the most {nutrient_term}? "
                        f"Choose the most accurate option: \n"
                    )
                if args.voting_ensemble == "3":
                    new_question = (
                        f"Based on the images and standard nutritional values, which of the following ingredients has the highest level of {nutrient_term}? "
                        f"Choose the best fit: \n"
                    )
                if args.voting_ensemble == "4":
                    new_question = (
                        f"Using the images and standard nutritional values as reference, which ingredient excels in terms of {nutrient_term} content? "
                        f"Select the best choice: \n"
                    )

                # 标准化选项格式
                processed_choices = []
                for idx, choice in enumerate(qdata["choices"]):
                    # 规范食材命名（处理大小写和特殊字符）
                    clean_choice = choice.strip()
                    if not clean_choice:
                        continue

                    # 智能大小写处理（保留品牌名/专有名词）
                    if ' ' in clean_choice or '-' in clean_choice:
                        # 处理复合名词（如"chicken stock cube"）
                        parts = re.split(r'([ -])', clean_choice)
                        standardized = []
                        for part in parts:
                            if part in (' ', '-'):
                                standardized.append(part)
                            elif len(part) > 3:
                                standardized.append(part.capitalize())
                            else:
                                standardized.append(part.lower())
                        standardized_choice = ''.join(standardized)
                    else:
                        # 处理单词（首字母大写）
                        standardized_choice = clean_choice.capitalize()

                    # 添加字母编号
                    letter = index_to_letter(idx)
                    processed_choices.append(f"{letter}. {standardized_choice}")

                # 构建结果
                new_qa_data[qid] = {
                    "inputs": qdata["inputs"],
                    "question": new_question,
                    "choices": processed_choices,
                }
            if task_name == "ingredient_ingredients_order":
                # 重构问题描述（增加时序强调）
                new_question = (
                    "Observing the cooking process in the video, what is the CORRECT chronological order "
                    "of ingredients being added to the dish? Select the exact sequence from first to last: \n"
                )

                if args.voting_ensemble == "1":
                    new_question = (
                        f"Based on the cooking process shown in the video, determine the correct chronological order in which the ingredients were added to the dish. "
                        f"Choose the exact sequence from first to last from the options below: \n"
                    )
                if args.voting_ensemble == "2":
                    new_question = (
                        f"By observing the cooking process in the video, identify the exact chronological sequence in which the ingredients were added to the dish. "
                        f"Select the correct order from the options below: \n"
                    )
                if args.voting_ensemble == "3":
                    new_question = (
                        f"Looking at the cooking video, which of the following options represents the correct chronological order of ingredients being added to the dish? "
                        f"Select the exact sequence from first to last: \n"
                    )
                if args.voting_ensemble == "4":
                    new_question = (
                        f"From the video of the cooking process, what is the exact chronological sequence of ingredient addition to the dish? "
                        f"Choose the correct sequence from the options below: \n"
                    )

                # 处理选项格式
                # processed_choices = []
                # for idx, choice in enumerate(qdata["choices"]):
                #     # 将列表转换为带序号的步骤描述
                #     # steps = [f"{i + 1}. {ingredient}" for i, ingredient in enumerate(choice)]
                #     sequence_str = " → ".join(choice)  # 添加箭头符号表示时序
                #
                #     # 构建带字母序号和视觉分隔符的选项
                #     letter = index_to_letter(idx)
                #     processed_choice = (
                #         f"{letter}. SEQUENCE: {sequence_str}"
                #         # f"    Details: {' | '.join(steps)}"
                #     )
                #     processed_choices.append(processed_choice)

                processed_choices = [
                    f"{index_to_letter(idx)}. {', '.join(choice)}"
                    for idx, choice in enumerate(qdata["choices"])
                ]

                new_inputs = {
                    "video": {
                        "id": qdata["inputs"]["video 1"]["id"],
                    }
                }

                # 构建新条目
                new_qa_data[qid] = {
                    "inputs": new_inputs,
                    "question": new_question,
                    "choices": processed_choices,
                }

            if task_name == "nutrition_nutrition_change":
                # 提取时间区间
                time_pattern = r"From <TIME (\d{2}:\d{2}:\d{2}\.\d{3}) video 1> to <TIME (\d{2}:\d{2}:\d{2}\.\d{3}) video 1>"
                time_match = re.search(time_pattern, qdata["question"])
                recipe_match = re.search(r"dish with recipe (.*?)\?", qdata["question"])

                if not time_match or not recipe_match:
                    raise ValueError("Failed to parse question components")

                recipe_name = recipe_match.group(1).replace(" with ", " with ")  # 保持格式统一

                # 重构输入结构
                new_inputs = {
                    "video": {
                        "id": qdata["inputs"]["video 1"]["id"],
                        "start_time": time_match.group(1),
                        "end_time": time_match.group(2),
                    }
                }

                # 重构问题描述
                new_question = (
                    f"What were the nutrient changes for the {recipe_name} dish? "
                    f"Choose the best option: \n"
                )

                if args.voting_ensemble == "1":
                    new_question = (
                        f"Which statement below most accurately describes the alterations in the nutritional profile of the {recipe_name} dish? "
                        f"Choose the best option: \n"
                    )
                if args.voting_ensemble == "2":
                    new_question = (
                        f"Which option best reflects the changes in the nutrient levels of the {recipe_name} dish? "
                        f"Select the most accurate description: \n"
                    )
                if args.voting_ensemble == "3":
                    new_question = (
                        f"Which description most accurately details the nutrient changes that occurred for the {recipe_name} dish? "
                        f"Choose the best fit: \n"
                    )
                if args.voting_ensemble == "4":
                    new_question = (
                        f"From the options provided, select the statement that best characterizes the nutritional shifts for the {recipe_name} dish: \n"
                    )

                # 处理选项
                processed_choices = []
                for idx, choice in enumerate(qdata["choices"]):
                    # 标准化数值格式
                    clean_choice = re.sub(
                        r"(\w+) changed by ([-\d.]+)",
                        lambda m: f"{m.group(1).title()}: {float(m.group(2)):+.1f}",
                        choice
                    )
                    # 添加字母编号
                    letter = index_to_letter(idx)
                    processed_choice = f"{letter}. {clean_choice}"
                    processed_choices.append(processed_choice)

                # 构建新条目
                new_qa_data[qid] = {
                    "inputs": new_inputs,
                    "question": new_question,
                    "choices": processed_choices,
                }

            if task_name == "nutrition_video_nutrition_estimation":
                # 提取营养类型并标准化术语
                nutrient_map = {
                    "carbs": "net carbohydrates",
                    "fat": "fat content",
                    "protein": "protein content"
                }

                # 匹配营养类型
                nutrient_match = re.search(r"highest (\w+)", qdata["question"])
                if not nutrient_match:
                    raise ValueError("Could not determine nutrient type from question")

                nutrient_key = nutrient_match.group(1).lower()
                nutrient_term = nutrient_map.get(nutrient_key, nutrient_key)  # 获取专业术语

                # 重构问题描述
                new_question = (
                    f"Which single ingredient in this recipe has the highest {nutrient_term} "
                    f"per standard serving? Select the most specific option: \n"
                )

                if args.voting_ensemble == "1":
                    new_question = (
                        f"Among the ingredients in this recipe, identify the single one that is richest in {nutrient_term} per standard serving. "
                        f"Select the most specific option from below: \n"
                    )
                if args.voting_ensemble == "2":
                    new_question = (
                        f"Considering the ingredients in this recipe and their standard servings, which single ingredient contains the highest amount of {nutrient_term}? "
                        f"Select the most accurate option from the choices: \n"
                    )
                if args.voting_ensemble == "3":
                    new_question = (
                        f"Looking at this recipe, which single ingredient demonstrates the highest level of {nutrient_term} per standard serving? "
                        f"Choose the most specific answer: \n"
                    )
                if args.voting_ensemble == "4":
                    new_question = (
                        f"From the ingredients in this recipe, select the single ingredient that has the highest {nutrient_term} content per standard serving. "
                        f"Choose the most specific option provided: \n"
                    )

                # 处理选项
                processed_choices = []
                for idx, choice in enumerate(qdata["choices"]):
                    # 标准化格式
                    clean_choice = choice.strip().lower()

                    # 处理复合描述（保留关键成分）
                    if "paste" in clean_choice or "mix" in clean_choice:
                        clean_choice = re.sub(r",\s*(and\s*)?", " + ", clean_choice)  # 统一连接符

                    # 首字母大写
                    clean_choice = clean_choice.capitalize()

                    # 添加字母标签
                    letter = index_to_letter(idx)
                    processed_choice = f"{letter}. {clean_choice}"
                    processed_choices.append(processed_choice)

                new_inputs = {
                    "video": {
                        "id": qdata["inputs"]["video 1"]["id"],
                    }
                }

                # 构建新条目
                new_qa_data[qid] = {
                    "inputs": new_inputs,
                    "question": new_question,
                    "choices": processed_choices,
                }

            if task_name == "recipe_step_recognition":
                # 提取时间区间
                time_pattern = r"between <TIME (\d{2}:\d{2}:\d{2}\.\d{3}) video 1> and <TIME (\d{2}:\d{2}:\d{2}\.\d{3}) video 1>"
                time_match = re.search(time_pattern, qdata["question"])
                if not time_match:
                    raise ValueError("Could not extract time interval from question")

                # 重构输入结构
                new_inputs = {
                    "video": {
                        "id": qdata["inputs"]["video 1"]["id"],
                        "start_time": time_match.group(1),
                        "end_time": time_match.group(2),
                    }
                }

                # 构建新问题
                new_question = "What specific actions did the participant perform? Choose the best option: \n"

                if args.voting_ensemble == "1":
                    new_question = (
                        f"Identify the specific actions performed by the participant. "
                        f"Choose the best option from below: \n"
                    )
                if args.voting_ensemble == "2":
                    new_question = (
                        f"Which statement best describes the specific actions carried out by the participant? "
                        f"Select the most accurate option: \n"
                    )
                if args.voting_ensemble == "3":
                    new_question = (
                        f"What specific activities did the participant engage in? "
                        f"Choose the best fit from the options: \n"
                    )
                if args.voting_ensemble == "4":
                    new_question = (
                        f"From the options provided, select the one that most accurately lists the specific actions the participant performed: \n"
                    )

                # 处理选项（标准化格式）
                processed_choices = [f"{index_to_letter(idx)}. {choice.capitalize()}"
                                     for idx, choice in enumerate(qdata["choices"])]

                # 构建新条目
                new_qa_data[qid] = {
                    "inputs": new_inputs,
                    "question": new_question,
                    "choices": processed_choices,
                }

            if task_name == "recipe_rough_step_localization":
                # 提取关键步骤名称
                step_match = re.search(r"recipe step (.*?)\?$", qdata["question"])
                if not step_match:
                    raise ValueError("Could not extract cooking step from question")

                # 简化步骤描述（保留核心动作+关键食材）
                step_desc = re.sub(r",?\s*chopped\s+", " ", step_match.group(1))  # 移除重复的chopped描述
                step_desc = re.sub(r"\s+", " ", step_desc).strip()  # 合并多余空格
                step_desc = f"<{step_desc}>"  # 用尖括号强调步骤名称

                # 重构问题描述
                new_question = (
                    f"Identify the time segment corresponding to the cooking step {step_desc}. "
                    f"Select the precise interval when this action occurs: \n"
                )

                if args.voting_ensemble == "1":
                    new_question = (
                        f"Match the cooking step '{step_desc}' with the precise time interval during which it occurs. "
                        f"Select the correct timeframe from the options below: \n"
                    )
                if args.voting_ensemble == "2":
                    new_question = (
                        f"Select the precise time interval below that represents when the cooking step '{step_desc}' occurs: \n"
                    )
                if args.voting_ensemble == "3":
                    new_question = (
                        f"What is the precise time interval during which the cooking step '{step_desc}' occurs? "
                        f"Choose the exact timeframe from the options below: \n"
                    )
                if args.voting_ensemble == "4":
                    new_question = (
                        f"From the options below, select the precise time interval that covers the execution of the cooking step '{step_desc}': \n"
                    )

                # 处理选项
                processed_choices = []
                for idx, choice in enumerate(qdata["choices"]):
                    # 提取时间区间
                    time_match = re.search(
                        r"(\d{2}:\d{2}:\d{2}\.\d{3}).*?(\d{2}:\d{2}:\d{2}\.\d{3})",
                        choice
                    )
                    if not time_match:
                        raise ValueError(f"Invalid time format in choice: {choice}")

                    # 格式化选项
                    letter = index_to_letter(idx)
                    processed_choice = f"{letter}. {time_match.group(1)} → {time_match.group(2)}"  # 使用箭头符号
                    processed_choices.append(processed_choice)

                # 构建标准化输入
                new_inputs = {"video": qdata["inputs"]["video 1"]}  # 统一为单数video

                # 构建新条目
                new_qa_data[qid] = {
                    "inputs": new_inputs,
                    "question": new_question,
                    "choices": processed_choices,
                }

            if task_name == "recipe_multi_recipe_recognition":
                # 重构问题描述
                new_question = (
                    "Based on the observed cooking steps and ingredients shown in the video, "
                    "which one of the following recipes was demonstrated? Select the exact match: \n"
                )

                if args.voting_ensemble == "1":
                    new_question = (
                        f"Based on the cooking steps and ingredients observed in the video, identify the recipe that was demonstrated. "
                        f"Select the exact match from the options below: \n"
                    )
                if args.voting_ensemble == "2":
                    new_question = (
                        f"Which of the following recipes is an exact match for the cooking process (steps and ingredients) demonstrated in the video? "
                        f"Select the correct option: \n"
                    )
                if args.voting_ensemble == "3":
                    new_question = (
                        f"Considering the cooking steps and ingredients visible in the video, what recipe was demonstrated? "
                        f"Select the option that is the exact match: \n"
                    )
                if args.voting_ensemble == "4":
                    new_question = (
                        f"From the options below, select the exact recipe that corresponds to the cooking process (steps and ingredients) observed in the video: \n"
                    )

                # 处理选项（添加关键特征标识）
                processed_choices = []
                for idx, choice in enumerate(qdata["choices"]):
                    letter = index_to_letter(idx)
                    processed_choices.append(f"{letter}. {choice}")

                new_inputs = {"video": qdata["inputs"]["video 1"]}  # 统一为单数video

                # 构建新条目
                new_qa_data[qid] = {
                    "inputs": new_inputs,
                    "question": new_question,
                    "choices": processed_choices,
                }

            if task_name == "recipe_following_activity_recognition":
                # 提取菜谱步骤核心内容
                step_match = re.search(r"recipe step (.*?) in this video\?", qdata["question"])
                if not step_match:
                    raise ValueError("Could not extract recipe step from question")

                # 简化步骤描述（保留核心动词短语）
                recipe_step = re.sub(r" -.*?\.", "", step_match.group(1))  # 移除破折号后的补充说明
                # recipe_step = re.sub(r",.*?\.", "", recipe_step)  # 移除逗号后的细节描述

                # 重构问题
                new_question = (
                    f"When following the recipe step '{recipe_step}', "
                    f"what was the participant's main action? \n"
                )

                if args.voting_ensemble == "1":
                    new_question = (
                        f"Based on the recipe step '{recipe_step}', identify the primary action performed by the participant. "
                        f"Choose the best option below: \n"
                    )
                if args.voting_ensemble == "2":
                    new_question = (
                        f"Which of the following statements best describes the participant's main action while performing the recipe step '{recipe_step}'? "
                        f"Select the best option: \n"
                    )
                if args.voting_ensemble == "3":
                    new_question = (
                        f"Focusing on recipe step '{recipe_step}', what was the main action performed by the participant? "
                        f"Choose the most appropriate option: \n"
                    )
                if args.voting_ensemble == "4":
                    new_question = (
                        f"Which option best characterizes the main action performed by the participant during recipe step '{recipe_step}'? "
                        f"Select the best choice: \n"
                    )

                # 处理选项（标准化动作描述）
                processed_choices = []
                for idx, choice in enumerate(qdata["choices"]):
                    # 添加字母序号
                    processed_choices.append(f"{index_to_letter(idx)}. {choice}")

                # 标准化输入结构
                new_inputs = {
                    "video": {
                        "id": qdata["inputs"]["video 1"]["id"]
                    }
                }

                # 构建新条目
                new_qa_data[qid] = {
                    "inputs": new_inputs,
                    "question": new_question,
                    "choices": processed_choices,
                }

            if task_name == "recipe_multi_step_localization":
                # 重构输入结构（单数video）
                new_inputs = {
                    "video": {"id": qdata["inputs"]["video 1"]["id"]}  # 保留必要元数据
                }

                if args.wo_refined_prompt:
                    processed_choices = []
                    for idx, choice in enumerate(qdata["choices"]):
                        letter = index_to_letter(idx)
                        processed_choice = f"{letter}. {choice}"
                        processed_choices.append(processed_choice)
                    new_qa_data[qid] = {
                        "inputs": new_inputs,
                        "question": qdata["question"] + "\n",
                        "choices": processed_choices,
                    }
                else:
                    # 提取所有步骤描述（匹配双引号内的内容）
                    steps_matches = re.findall(r'\"(.*?)\"', qdata["question"])
                    if not steps_matches or len(steps_matches) < 2:
                        raise ValueError("Could not extract recipe steps from question")

                    # 重构问题描述
                    steps_str = "; ".join([f"{idx + 1}. {step}" for idx, step in enumerate(steps_matches)])
                    # print(steps_str)
                    new_question = (
                        f"Identify the exact time intervals for these recipe steps: {steps_str}\n"
                        "Choose the option where all time sequences match the steps in order. \n"
                    )

                    if args.voting_ensemble == "1":
                        new_question = (
                            f"For the following recipe steps: {steps_str}, identify the exact time intervals for each step. "
                            f"Choose the option where all provided time sequences correspond precisely to the steps listed, in order:\n"
                        )
                    if args.voting_ensemble == "2":
                        new_question = (
                            f"From the options below, select the one that provides the exact time intervals corresponding to the recipe steps: {steps_str}, with the time sequences matching the steps in the correct order:\n"
                        )
                    if args.voting_ensemble == "3":
                        new_question = (
                            f"What are the exact time intervals for the sequence of recipe steps: {steps_str}? "
                            f"Choose the option below where the time intervals are listed in the correct chronological order corresponding to each step:\n"
                        )
                    if args.voting_ensemble == "4":
                        new_question = (
                            f"Associate the exact time intervals with each of the recipe steps listed: {steps_str}. "
                            f"Choose the option where the timeframe for each step is correctly listed in sequential order:\n"
                        )

                    if args.wo_refined_answer:
                        processed_choices = []
                        for idx, choice in enumerate(qdata["choices"]):
                            letter = index_to_letter(idx)
                            processed_choice = f"{letter}. {choice}"
                            processed_choices.append(processed_choice)
                        new_qa_data[qid] = {
                            "inputs": new_inputs,
                            "question": new_question,
                            "choices": processed_choices,
                        }
                    else:
                        # 处理选项
                        processed_choices = []
                        for idx, choice in enumerate(qdata["choices"]):
                            # 提取时间区间（兼容多种格式）
                            time_pattern = r"<TIME\s+(\d{2}:\d{2}:\d{2}\.\d{1,3})\s+video\s+1>\s*to\s*<TIME\s+(\d{2}:\d{2}:\d{2}\.\d{1,3})\s+video\s+1>"
                            time_matches = re.findall(time_pattern, choice)
                            if len(time_matches) != len(steps_matches):
                                raise ValueError(
                                    f"Time intervals count mismatch in choice {idx} of {qid}\n"
                                    f"Expected: {len(steps_matches)} intervals (steps: {steps_matches})\n"
                                    f"Actual: {len(time_matches)} intervals in: {choice}"
                                )

                            # 格式化选项（每步骤独立行）
                            letter = index_to_letter(idx)
                            time_str = "; ".join([f"Step {i + 1}: {start} - {end}"
                                                  for i, (start, end) in enumerate(time_matches)])
                            processed_choice = f"{letter}. {time_str}"
                            processed_choices.append(processed_choice)
                    # 构建新条目
                    new_qa_data[qid] = {
                        "inputs": new_inputs,
                        "question": new_question,
                        "choices": processed_choices,
                    }

            if task_name == "ingredient_exact_ingredient_recognition":
                # 提取成分和菜品名称
                ingredient_match = re.search(r"exact quantity of (.+?) used in (.+)$", qdata["question"])
                if not ingredient_match:
                    raise ValueError("Could not extract ingredient/dish from question")

                ingredient, dish = ingredient_match.groups()

                # 构建专业烹饪场景的问题
                new_question = (
                    f"When preparing <{dish}>, what precise amount of <{ingredient}> should be used? "
                    f"Select the correct measurement: \n"
                )

                if args.voting_ensemble == "1":
                    new_question = (
                        f"When preparing the dish <{dish}>, what is the exact quantity of <{ingredient}> that should be used? "
                        f"Select the correct measurement from the options below: \n"
                    )
                if args.voting_ensemble == "2":
                    new_question = (
                        f"For the preparation of <{dish}>, what is the precise measurement of <{ingredient}> that should be used? "
                        f"Select the correct value from the options below: \n"
                    )
                if args.voting_ensemble == "3":
                    new_question = (
                        f"From the options below, identify the precise amount of <{ingredient}> that should be used when preparing <{dish}>. "
                        f"Select the correct measurement: \n"
                    )
                if args.voting_ensemble == "4":
                    new_question = (
                        f"Precisely how much of <{ingredient}> should be used when making <{dish}>? "
                        f"Choose the correct measurement from the options below: \n"
                    )

                # 规范化选项格式
                processed_choices = [
                    f"{index_to_letter(idx)}. {choice}"  # 保留数值和单位
                    if " " in choice else f"{index_to_letter(idx)}. {choice}"
                    for idx, choice in enumerate(qdata["choices"])
                ]

                # 构建结构化数据
                new_qa_data[qid] = {
                    "inputs": qdata["inputs"],
                    "question": new_question,
                    "choices": processed_choices,
                }

            if task_name == "ingredient_ingredient_recognition":
                # 提取菜谱名称和判断类型（used/not used）
                if "not used" in qdata["question"]:
                    recipe_match = re.search(r"not used in (.*)$", qdata["question"])
                    prompt_type = "exclude"
                else:
                    recipe_match = re.search(r"used in (.*)$", qdata["question"])
                    prompt_type = "include"

                if not recipe_match:
                    raise ValueError("Could not extract recipe name from question")

                recipe_name = recipe_match.group(1).strip()

                # 构建认知强化型prompt
                if prompt_type == "exclude":
                    new_question = (
                        f"Based on the standard recipe for {recipe_name}, which one of the following ingredients "
                        f"is typically EXCLUDED? Select the anomaly. \n"
                    )
                    if args.voting_ensemble == "1":
                        new_question = (
                            f"According to the standard recipe for {recipe_name}, which ingredient is typically not included? "
                            f"Select the one that doesn't belong from the options below: \n"
                        )
                    if args.voting_ensemble == "2":
                        new_question = (
                            f"In the standard preparation of {recipe_name}, which ingredient is typically absent? "
                            f"Choose the one not found in this recipe from the options below: \n"
                        )
                    if args.voting_ensemble == "3":
                        new_question = (
                            f"For the standard recipe of {recipe_name}, which one of the following ingredients is the outlier? "
                            f"Select the ingredient that is not typically part of this recipe: \n"
                        )
                    if args.voting_ensemble == "4":
                        new_question = (
                            f"Based on the standard recipe for {recipe_name}, which ingredient from the options below is usually left out? "
                            f"Choose the ingredient that is typically excluded: \n"
                        )
                else:
                    new_question = (
                        f"Which of these ingredients is ESSENTIAL in authentic {recipe_name} preparation? "
                        f"Choose the required component. \n"
                    )
                    if args.voting_ensemble == "1":
                        new_question = (
                            f"In authentic {recipe_name} preparation, which ingredient is absolutely necessary? "
                            f"Choose the required component from the options below: \n"
                        )
                    if args.voting_ensemble == "2":
                        new_question = (
                            f"For authentic {recipe_name} preparation, which ingredient is critical? "
                            f"Select the indispensable ingredient from the options below: \n"
                        )
                    if args.voting_ensemble == "3":
                        new_question = (
                            f"Which of these ingredients is a key part of authentic {recipe_name} preparation? "
                            f"Choose the essential ingredient from the options below: \n"
                        )
                    if args.voting_ensemble == "4":
                        new_question = (
                            f"Identify the ingredient from the options below that is a mandatory requirement in authentic {recipe_name} preparation. "
                            f"Select the correct essential ingredient: \n"
                        )

                # 结构化选项处理
                processed_choices = []
                for idx, choice in enumerate(qdata["choices"]):
                    # 添加字母序号并标准化格式
                    letter = index_to_letter(idx)
                    processed_choice = f"{letter}. {choice.capitalize()}"

                    processed_choices.append(processed_choice)

                # 构建新条目
                new_qa_data[qid] = {
                    "inputs": qdata["inputs"],
                    "question": new_question,
                    "choices": processed_choices,
                }

            if task_name == "recipe_recipe_recognition":
                new_question = (
                    "Based on the participant's actual cooking steps shown, which recipe did they COMPLETELY DEMONSTRATE? "
                    "Focus on matching key preparation phases. \n"
                )

                if args.voting_ensemble == "1":
                    new_question = (
                        f"Based on the participant's actual cooking steps and focusing on matching the key preparation phases, "
                        f"which of the following recipes was completely demonstrated? Choose the best option: \n"
                    )
                if args.voting_ensemble == "2":
                    new_question = (
                        f"Considering the participant's actions in the video, which recipe had its complete key preparation phases demonstrated? "
                        f"Select the recipe that was fully shown: \n"
                    )
                if args.voting_ensemble == "3":
                    new_question = (
                        f"Which recipe is fully represented by the participant's actual cooking steps shown in the video, focusing on the key preparation phases? "
                        f"Choose the best match: \n"
                    )
                if args.voting_ensemble == "4":
                    new_question = (
                        f"From the options below, select the recipe that corresponds to the complete demonstration, "
                        f"based on the participant's actual cooking steps and matching key preparation phases: \n"
                    )

                processed_choices = []
                for idx, choice in enumerate(qdata["choices"]):
                    # 添加认知特征注释
                    letter = index_to_letter(idx)

                    # 结构化呈现
                    processed_choice = f"{letter}. {choice}"
                    processed_choices.append(processed_choice)

                # 构建强化prompt
                new_qa_data[qid] = {
                    "inputs": qdata["inputs"],
                    "question": new_question,
                    "choices": processed_choices,
                }

            if task_name == "recipe_prep_localization":
                if args.refined_question_later:
                    new_question = qdata["question"]
                else:
                    # 提取菜谱名称和具体步骤
                    step_match = re.search(r"perform prep for (.*?) from recipe (.*?)\?", qdata["question"])
                    if not step_match:
                        raise ValueError("Could not extract cooking step and recipe name")

                    step_description = step_match.group(1).strip()
                    recipe_name = step_match.group(2).strip()

                    # 构建强化理解的prompt
                    new_question = (
                        f"When preparing {recipe_name}, during which time intervals did the participant perform: "
                        f"'{step_description}'? Select all correct segments. \n"
                    )

                    if args.voting_ensemble == "1":
                        new_question = (
                            f"During the preparation of {recipe_name}, what are all the precise time intervals when the participant performed the step '{step_description}'? "
                            f"Select all correct segments from the options below:\n"
                        )
                    if args.voting_ensemble == "2":
                        new_question = (
                            f"From the options below, select all precise time intervals that correspond to the participant performing the recipe step '{step_description}' during the preparation of {recipe_name}: \n"
                        )
                    if args.voting_ensemble == "3":
                        new_question = (
                            f"What are the precise time intervals during which the participant performed the step '{step_description}' while preparing {recipe_name}? "
                            f"Select all correct segments from the options below:\n"
                        )
                    if args.voting_ensemble == "4":
                        new_question = (
                            f"Pinpoint all exact time intervals where the participant performed the recipe step '{step_description}' during the preparation of {recipe_name}. "
                            f"Select all correct segments from the options below:\n"
                        )

                # 处理时间段选项
                processed_choices = []
                for idx, choice in enumerate(qdata["choices"]):
                    # 提取所有时间段和视频信息
                    time_blocks = re.findall(
                        r"<TIME (\d{2}:\d{2}:\d{2}\.\d{3}) video (\d+)> to <TIME (\d{2}:\d{2}:\d{2}\.\d{3}) video (\d+)> \((\w+ \d+)\)",
                        choice
                    )
                    if not time_blocks:
                        raise ValueError(f"选项{idx}的时间格式无效")

                    # 验证视频一致性
                    video_sources = set()
                    formatted_segments = []
                    for start, v1, end, v2, vid in time_blocks:
                        if v1 != v2:
                            raise ValueError(f"选项{idx}中存在跨视频时间段")
                        video_sources.add(f"V{v1}")
                        formatted_segments.append(f"[V{v1}] {start} - {end}")

                    # 构建选项描述
                    letter = index_to_letter(idx)
                    duration_note = ""
                    if len(formatted_segments) > 1:
                        total_sec = sum(
                            (datetime.strptime(end, "%H:%M:%S.%f") -
                             datetime.strptime(start, "%H:%M:%S.%f")
                             ).total_seconds()
                            for start, _, end, _, _ in time_blocks
                        )
                        duration_note = f" ({len(formatted_segments)} clips, {int(total_sec)}s)"

                    video_tag = " ".join(sorted(video_sources)) + ": " if video_sources else ""
                    processed_choice = (
                            f"{letter}. {video_tag}" +
                            "; ".join(formatted_segments) # + duration_note
                    )
                    processed_choices.append(processed_choice)

                # 构建新条目
                new_qa_data[qid] = {
                    "inputs": qdata["inputs"],
                    "question": new_question,
                    "choices": processed_choices,
                }

            if task_name == "recipe_step_localization":
                if args.refined_question_later:
                    new_question = qdata["question"]
                else:
                    # 提取关键信息
                    step_match = re.search(r"step (.*?) from recipe (.*?)\?", qdata["question"])
                    if not step_match:
                        raise ValueError("Could not extract cooking step and recipe name")

                    step_name = step_match.group(1).lower()
                    recipe_name = step_match.group(2)

                    # 构建强化认知的prompt
                    new_question = (
                        f"During the preparation of {recipe_name}, identify the EXACT time range(s) when the participant "
                        f"performed the step: '{step_name}'. Choose the cooking sequence. \n"
                    )
                    if args.voting_ensemble == "1":
                        new_question = (
                            f"Pinpoint the EXACT time range(s) during the preparation of {recipe_name} when the participant performed the step: '{step_name}'. "
                            f"Select the option that provides the correct sequence of time intervals: \n"
                        )
                    if args.voting_ensemble == "2":
                        new_question = (
                            f"Identify the EXACT time range(s) associated with the participant performing the step: '{step_name}' during the preparation of {recipe_name}. "
                            f"Choose the option that represents the correct cooking sequence: \n"
                        )
                    if args.voting_ensemble == "3":
                        new_question = (
                            f"What are the EXACT time range(s) during which the participant performed the step '{step_name}' while preparing {recipe_name}? "
                            f"Select the option below that lists the correct sequence of time intervals: \n"
                        )
                    if args.voting_ensemble == "4":
                        new_question = (
                            f"From the options below, select the one that lists the EXACT time range(s) corresponding to the participant performing the step: '{step_name}' during the preparation of {recipe_name}. "
                            f"Choose the correct cooking sequence: \n"
                        )

                # 处理时间段选项
                processed_choices = []
                for idx, choice in enumerate(qdata["choices"]):
                    # 提取所有时间段和视频信息
                    time_blocks = re.findall(
                        r"<TIME (\d{2}:\d{2}:\d{2}\.\d{3}) video (\d+)> to <TIME (\d{2}:\d{2}:\d{2}\.\d{3}) video (\d+)> \((\w+ \d+)\)",
                        choice
                    )
                    if not time_blocks:
                        raise ValueError(f"选项{idx}的时间格式无效")

                    # 验证视频一致性
                    video_sources = set()
                    formatted_segments = []
                    for start, v1, end, v2, vid in time_blocks:
                        if v1 != v2:
                            raise ValueError(f"选项{idx}中存在跨视频时间段")
                        video_sources.add(f"V{v1}")
                        formatted_segments.append(f"[V{v1}] {start} - {end}")

                    # 构建选项描述
                    letter = index_to_letter(idx)
                    duration_note = ""
                    if len(formatted_segments) > 1:
                        total_sec = sum(
                            (datetime.strptime(end, "%H:%M:%S.%f") -
                             datetime.strptime(start, "%H:%M:%S.%f")
                             ).total_seconds()
                            for start, _, end, _, _ in time_blocks
                        )
                        duration_note = f" ({len(formatted_segments)} clips, {int(total_sec)}s)"

                    video_tag = " ".join(sorted(video_sources)) + ": " if video_sources else ""
                    processed_choice = (
                            f"{letter}. {video_tag}" +
                            "; ".join(formatted_segments)  # + duration_note
                    )
                    processed_choices.append(processed_choice)

                # 构建新条目
                new_qa_data[qid] = {
                    "inputs": qdata["inputs"],
                    "question": new_question,
                    "choices": processed_choices,
                }

        except Exception as e:
            error_msg = f"Error processing {qid}: {str(e)}"
            print(f"  {error_msg}")
            error_log.append(error_msg)
            continue

        # 写入成功日志
        with open(log_file, "a") as f:
            f.write("\n".join(log_entry) + "\n\n")

    if args.generate_refined_prompt:
        # 保存新QA数据
        new_qa_path = os.path.join(task_output_dir, f"{task_name}{args.voting_ensemble}.json")
        with open(new_qa_path, "w") as f:
            json.dump(new_qa_data, f, indent=4)

    # 保存错误日志
    if error_log:
        error_path = os.path.join(task_output_dir, "errors.txt")
        with open(error_path, "w") as f:
            f.write("\n".join(error_log))

print("\nProcessing completed. Check logs for details.")
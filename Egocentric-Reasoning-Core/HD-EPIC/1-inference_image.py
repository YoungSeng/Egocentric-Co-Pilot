import json
import os
import re
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch
from qwen_vl_utils import process_vision_info
import argparse
from tqdm import tqdm
import subprocess

parser = argparse.ArgumentParser()
# CHANGED: Default paths relative
parser.add_argument("--hd_epic_database", type=str, default="./HD-EPIC/")
parser.add_argument("--task", type=str, default="3d_perception_fixture_location",
                    choices=["3d_perception_fixture_location"])
parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
parser.add_argument("--inference_percentage", type=int, default=100,
                    help="Percentage of samples to use for inference (0-100). Default is 100.")
parser.add_argument("--evaluate_accuracy", action='store_true')
parser.add_argument("--hd_epic_vqa_annotations", type=str, default="./hd-epic-annotations/vqa-benchmark/")
parser.add_argument("--print_intermediate", action="store_true",
                    help="Print intermediate question and prediction for each sample.")
parser.add_argument("--task_name", type=str, default="PreprocessedVideos")
parser.add_argument("--voting_ensemble", type=str, default="")

args = parser.parse_args()

task = args.task
model_name = args.base_model
inference_percentage = args.inference_percentage

print('\n', "Task:", task, "Model:", model_name, "Inference Percentage:", inference_percentage, '\n')
HD_EPIC_database = args.hd_epic_database
preprocessed_videos_base = os.path.join(HD_EPIC_database, args.task_name)

# 配置路径
input_json = os.path.join(preprocessed_videos_base, f"{task}/{task}{args.voting_ensemble}.json")
image_base = os.path.join(preprocessed_videos_base, f"{task}/images")
output_json = os.path.join(preprocessed_videos_base, f"{task}/predictions_{task}{args.voting_ensemble}.json")

# 加载模型和处理器（启用flash attention优化）
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_name)

# 处理所有条目
with open(input_json, 'r') as f:
    data = json.load(f)

# 计算需要推理的样本数量
total_samples = len(data)
clamped_percentage = max(0, min(100, inference_percentage))
num_samples_to_process = int(total_samples * (clamped_percentage / 100.0))

print(f"Total samples available: {total_samples}")
print(f"Processing {num_samples_to_process} samples ({clamped_percentage}%) for inference...")

results = {}
all_keys = list(data.keys())
intermediate = []

for key in tqdm(all_keys[0:num_samples_to_process], desc="Inferencing"):
    entry = data[key]

    # 构建图片路径
    image_path = os.path.join(image_base, f"{key}.png")

    # 构建对话消息
    messages = [
        {
            "role": "system",
            "content": "You are an expert image analyzer. Answer the multiple choice question by giving ONLY the number identifying the answer. Example: A or B or C or D or E. You MUST answer even if unsure."
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": entry["question"]}
            ]
        }
    ]

    # 预处理输入
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,  # 直接使用图片路径
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    results[key] = output_text

    intermediate.append(f"{key}\nQuestion: {entry['question']}\nOutput: {output_text}\n{'=' * 50}")
    if args.print_intermediate:
        print(entry["question"], output_text)

# 保存结果
with open(output_json, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Predictions saved to {output_json}")

with open(os.path.join(preprocessed_videos_base, f"{task}/intermediate_results{args.voting_ensemble}.txt"), 'w') as f:
    f.write("\n\n".join(intermediate))

if args.evaluate_accuracy:
    evaluate_result = os.path.join(preprocessed_videos_base, f"{task}/evaluation_results{args.voting_ensemble}.txt")
    evaluate_cmd = [
        'python', "evaluate_script.py", output_json, args.hd_epic_vqa_annotations, evaluate_result, args.voting_ensemble
    ]
    try:
        subprocess.run(evaluate_cmd, check=True, stderr=subprocess.PIPE)

    except subprocess.CalledProcessError as e:
        print(e.stderr.decode())
import json
import os
import re
from PIL import Image
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import argparse
import subprocess
from tqdm import tqdm

# Config Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="3d_perception_fixture_location",
                    choices=["3d_perception_fixture_interaction_counting", "3d_perception_object_contents_retrieval",
                             "3d_perception_object_location", "object_motion_object_movement_counting",
                             "object_motion_object_movement_itinerary", "object_motion_stationary_object_localization"]
                    )
# Modified: Default paths changed to relative
parser.add_argument("--hd_epic_vqa_annotations", type=str, default="./dataset/hd-epic-annotations/vqa-benchmark/")
parser.add_argument("--hd_epic_database", type=str, default="./dataset/HD-EPIC/")
parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
parser.add_argument("--print_intermediate", action="store_true",
                    help="Print intermediate question and prediction for each sample.")
parser.add_argument("--inference_percentage", type=int, default=100,
                    help="Percentage of samples to use for inference (0-100). Default is 100.")
parser.add_argument("--task_name", type=str, default="PreprocessedVideos")
parser.add_argument("--voting_ensemble", type=str, default="")

args = parser.parse_args()

model_name = args.base_model

# Initialize Model and Processor
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_name)

task = args.task
json_dir = args.hd_epic_vqa_annotations
HD_EPIC_database = args.hd_epic_database
# Config Paths
input_json_path = os.path.join(json_dir, f"{task}.json")
preprocessed_videos_base = os.path.join(HD_EPIC_database, args.task_name)
image_base_dir = os.path.join(preprocessed_videos_base, f"{task}/images")
output_dir = os.path.join(os.path.join(preprocessed_videos_base, f"{task}"))
narration_dir = os.path.join(output_dir, "narration")
os.makedirs(narration_dir, exist_ok=True)

print('\n', "Task:", task, "Model:", model_name, "Inference Percentage:", args.inference_percentage, '\n')

# Load Data
with open(input_json_path, "r") as f:
    data = json.load(f)

# Calculate Samples
total_samples = len(data)
clamped_percentage = max(0, min(100, args.inference_percentage))
num_samples_to_process = int(total_samples * (clamped_percentage / 100.0))

print(f"Total samples available: {total_samples}")
print(f"Processing {num_samples_to_process} samples ({clamped_percentage}%) for inference...")

def extract_bbox_coordinates(question):
    """Extract BBOX coordinates"""
    match = re.search(r"<BBOX (.*?)>", question)
    if not match:
        return None
    return list(map(float, match.group(1).split()))


def generate_narration(image_path, bbox_coords):
    """Generate object description using Qwen"""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
    except Exception as e:
        print(f"Error opening image: {e}")
        return None

    x1_norm, y1_norm, x2_norm, y2_norm = bbox_coords

    # Build Prompt
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text",
             "text": f"<box>{x1_norm},{y1_norm},{x2_norm},{y2_norm}</box> What is this object? Answer with simple noun only."}
        ]
    }]

    # Process Input
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    vision_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=vision_inputs,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    # Generate Response
    generated_ids = model.generate(**inputs, max_new_tokens=20)
    response = processor.batch_decode(generated_ids[:, inputs.input_ids.shape[1]:],
                                      skip_special_tokens=True)[0].strip().lower()

    if args.print_intermediate:
        print(response)
    return response

all_keys = list(data.keys())

# Main Loop
for key in tqdm(all_keys[0:num_samples_to_process], desc="Inferencing"):
    entry = data[key]
    narration_path = os.path.join(narration_dir, f"{key}.txt")

    if os.path.exists(narration_path):
        with open(narration_path, "r") as f:
            narration = f.read().strip()
        if args.print_intermediate:
            print(f"Loaded existing narration for {key}: {narration}")
    else:
        image_path = os.path.join(image_base_dir, f"{key}.png")
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue

        bbox_coords = extract_bbox_coordinates(entry["question"])
        if not bbox_coords:
            print(f"No BBOX found in: {key}")
            continue

        narration = generate_narration(image_path, bbox_coords)
        if not narration:
            continue

        with open(os.path.join(narration_dir, f"{key}.txt"), "w") as f:
            f.write(narration)

    entry["question"] = re.sub(
        r"(item|object) (indicated|identified) by( bounding box)? <BBOX [^>]+>",
        f"{narration}",
        entry["question"]
    )

    entry["question"] = re.sub(
        r"the (object|item) <BBOX [^>]+>",
        f"{narration}",
        entry["question"]
    )

    if args.print_intermediate:
        print(entry["question"])

output_path = os.path.join(output_dir, "reformatted_questions_narration.json")
with open(output_path, "w") as f:
    json.dump(data, f, indent=4)

print("Processing completed!")
import json
import os
import re
import subprocess
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch
from qwen_vl_utils import process_vision_info
import argparse
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="nutrition_image_nutrition_estimation",
                    choices=["nutrition_image_nutrition_estimation"]
                    )
# Modified: Default paths changed to relative
parser.add_argument("--hd_epic_vqa_annotations", type=str, default="./dataset/hd-epic-annotations/vqa-benchmark/")
parser.add_argument("--hd_epic_database", type=str, default="./dataset/HD-EPIC/")
parser.add_argument("--inference_percentage", type=int, default=100,
                    help="Percentage of samples to use for inference (0-100). Default is 100.")
parser.add_argument("--evaluate_accuracy", action='store_true')
parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
parser.add_argument("--print_intermediate", action="store_true",
                    help="Print intermediate question and prediction for each sample.")
parser.add_argument("--task_name", type=str, default="PreprocessedVideos")
parser.add_argument("--voting_ensemble", type=str, default="")

args = parser.parse_args()

# Config Task
task = args.task
model_name = args.base_model
HD_EPIC_database = args.hd_epic_database
preprocessed_videos_base = os.path.join(HD_EPIC_database, args.task_name)

# Config Paths
input_json = os.path.join(preprocessed_videos_base, f"{task}/{task}{args.voting_ensemble}.json")
video_base = os.path.join(HD_EPIC_database, "Videos")
image_base = os.path.join(preprocessed_videos_base, f"{task}/images")
output_json = os.path.join(preprocessed_videos_base, f"{task}/predictions_{task}{args.voting_ensemble}.json")

# Create Directories
os.makedirs(image_base, exist_ok=True)


def process_video_frame(video_path, timestamp, output_path):
    """Extract video frame using ffmpeg"""
    ffmpeg_cmd = [
        'ffmpeg',
        '-ss', str(timestamp),
        '-i', video_path,
        '-vframes', '1',
        '-q:v', '1',
        '-y',
        output_path
    ]
    try:
        subprocess.run(ffmpeg_cmd, check=True, stderr=subprocess.PIPE)
        return True, ""
    except subprocess.CalledProcessError as e:
        return False, e.stderr.decode()


# Load Model
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_name)

# Process Entries
with open(input_json, 'r') as f:
    data = json.load(f)

# Calculate Samples
total_samples = len(data)
clamped_percentage = max(0, min(100, args.inference_percentage))
num_samples_to_process = int(total_samples * (clamped_percentage / 100.0))

print(f"Total samples available: {total_samples}")
print(f"Processing {num_samples_to_process} samples ({clamped_percentage}%) for inference...")

results = {}
intermediate = []

for q_id, key in tqdm(enumerate(list(data.keys())[0:num_samples_to_process]), desc="Processing questions"):
    entry = data[key]
    image_contents = []

    try:
        # Process each input image
        for img_key in sorted(entry["inputs"].keys()):
            img_info = entry["inputs"][img_key]

            # Build video path
            user_id = img_info["id"].split('-')[0]
            video_name = f"{img_info['id']}.mp4"
            video_path = os.path.join(video_base, user_id, video_name)

            # Build output image path
            frame_filename = f"{key}_{img_key.replace(' ', '_')}.png"
            output_path = os.path.join(image_base, frame_filename)

            if not os.path.exists(output_path):
                if not process_video_frame(video_path, img_info["time"], output_path):
                    raise RuntimeError(f"Failed to extract frame from {video_path} at {img_info['time']}")

            image_contents.append({"type": "image", "image": f"{output_path}"})

        final_question = entry["question"] + "\n".join(entry["choices"])
        # Build Qwen messages
        messages = [
            {
                "role": "system",
                "content": "You are an expert nutrition analyzer. Answer the multiple choice question by giving ONLY the letter identifying the answer (e.g. 'A'). You MUST answer even if unsure."
            },
            {
                "role": "user",
                "content": image_contents + [
                    {"type": "text", "text": final_question}
                ]
            }
        ]

        # Preprocess Inputs
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to("cuda")

        # Inference
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        results[key] = output_text

    except Exception as e:
        print(f"Error processing {key}: {str(e)}")
        results[key] = "ERROR"

    intermediate.append(f"{key}\nQuestion: {final_question}\nOutput: {output_text}\n{'=' * 50}")
    if args.print_intermediate:
        print(f"Processed {key}: Question: {final_question}  Answer: {results[key]}")

# Save Results
with open(output_json, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Predictions saved to {output_json}")

with open(os.path.join(preprocessed_videos_base, f"{task}/intermediate_results{args.voting_ensemble}.txt"), 'w') as f:
    f.write("\n\n".join(intermediate))

if args.evaluate_accuracy:
    evaluate_result = os.path.join(preprocessed_videos_base, f"{task}/evaluation_results{args.voting_ensemble}.txt")
    evaluate_cmd = [
        'python', "evaluate_script.py",output_json, args.hd_epic_vqa_annotations,
        evaluate_result, args.voting_ensemble
    ]
    try:
        subprocess.run(evaluate_cmd, check=True, stderr=subprocess.PIPE)

    except subprocess.CalledProcessError as e:
        print(e.stderr.decode())
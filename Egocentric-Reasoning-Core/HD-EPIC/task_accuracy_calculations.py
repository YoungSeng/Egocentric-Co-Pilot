import json
import os
import re
from collections import defaultdict
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--hd_epic_database", type=str, required=True, help="Path to HD-EPIC database root.")
parser.add_argument("--task_name", type=str, default="PreprocessedVideos", help="Task name folder.")
parser.add_argument("--output_json", type=str, default="predictions.json")
parser.add_argument("--output_txt", type=str, default="results.txt")

args = parser.parse_args()

# ... (task_to_category definition remains identical) ...
task_to_category = {
    "Recipe Recognition": "Recipe",
    "Multi_Recipe Recognition": "Recipe",
    "Multi_Step Localization": "Recipe",
    "Step Localization": "Recipe",
    "Prep Localization": "Recipe",
    "Step Recognition": "Recipe",
    "Rough Step Localization": "Recipe",
    "Following Activity Recognition": "Recipe",
    "Ingredient Retrieval": "Ingredient",
    "Ingredient Weight": "Ingredient",
    "Ingredients Order": "Ingredient",
    "Ingredient Adding Localization": "Ingredient",
    "Ingredient Recognition": "Ingredient",
    "Exact Ingredient Recognition": "Ingredient",
    "Image Nutrition Estimation": "Nutrition",
    "Nutrition Change": "Nutrition",
    "Video Nutrition Estimation": "Nutrition",
    "Action Recognition": "Fine_grained",
    "How Recognition": "Fine_grained",
    "Why Recognition": "Fine_grained",
    "Action Localization": "Fine_grained",
    "Fixture Location": "3D_perception",
    "Object Location": "3D_perception",
    "Object Contents Retrieval": "3D_perception",
    "Fixture Interaction Counting": "3D_perception",
    "Object Movement Itinerary": "Object_motion",
    "Object Movement Counting": "Object_motion",
    "Stationary Object Localization": "Object_motion",
    "Gaze Estimation": "Gaze",
    "Interaction Anticipation": "Gaze"
}

tasks = list(task_to_category.keys())
HD_EPIC_database = args.hd_epic_database
base_path = os.path.join(HD_EPIC_database, args.task_name)

def letter_to_num(letter):
    return {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}.get(letter.upper(), 0)

def parse_evaluation_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    entries = []
    current_entry = []

    for line in content.split('\n'):
        line = line.strip()
        if line.startswith("QID:"):
            if current_entry:
                entries.append(" ".join(current_entry))
            current_entry = [line]
        elif line.startswith("---"):
            if current_entry:
                entries.append(" ".join(current_entry))
                current_entry = []
        else:
            if current_entry and line:
                current_entry.append(line)

    if current_entry:
        entries.append(" ".join(current_entry))

    results = []
    for entry in entries:
        if not entry.startswith("QID:"):
            continue

        try:
            qid_match = re.search(r'QID:\s*([^,]+)', entry)
            pred_match = re.search(r'Predicted:\s*([^,]+)', entry)
            gt_match = re.search(r'GT:\s*(\d+)', entry)
            correct_match = re.search(r'Correct:\s*(True|False)', entry)

            if not all([qid_match, pred_match, gt_match, correct_match]):
                continue

            qid = qid_match.group(1).strip()
            predicted = pred_match.group(1).strip()
            gt = int(gt_match.group(1))
            correct = correct_match.group(1) == "True"

            letter_match = re.match(r'^([A-E])[\.\s,;]?', predicted)
            pred_letter = letter_match.group(1) if letter_match else ''

            results.append({
                "qid": qid,
                "pred": pred_letter,
                "correct": correct,
                "gt": gt
            })
        except Exception as e:
            print(f"Parsing failed: {entry[:50]}... Error: {str(e)}")

    return results

predictions = {}
task_accuracies = {}
category_accuracies = defaultdict(list)

for task in tasks:
    task_dir = task.lower().replace(" ", "_")
    category = task_to_category[task]
    dir_name = f"{category.lower()}_{task_dir}"
    result_path = os.path.join(base_path, dir_name, "evaluation_results.txt")

    if not os.path.exists(result_path):
        print(f"File not found: {result_path}")
        continue

    entries = parse_evaluation_file(result_path)
    if not entries:
        print(f"No valid entries found: {task}")
        continue

    correct = sum(1 for e in entries if e["correct"])
    total = len(entries)
    acc = (correct / total * 100) if total > 0 else 0.0
    task_accuracies[task] = acc

    for e in entries:
        predictions[e["qid"]] = letter_to_num(e["pred"])

    category_accuracies[category].append(acc)

category_avg = {cat: sum(accs) / len(accs) for cat, accs in category_accuracies.items()}

categories_for_total = [
    "Recipe", "Ingredient", "Nutrition", "Fine_grained",
    "3D_perception", "Object_motion", "Gaze"
]
category_values = [category_avg.get(cat, 0) for cat in categories_for_total]
total_avg = sum(category_values) / len(category_values) if category_values else 0

with open(args.output_json, 'w') as f:
    json.dump(predictions, f, indent=2)

with open(args.output_txt, 'w') as f:
    f.write("=== Task Accuracies ===\n")
    for task, acc in task_accuracies.items():
        f.write(f"{task}: {acc:.2f}%\n")

    f.write("\n=== Category Averages ===\n")
    for cat in categories_for_total:
        avg = category_avg.get(cat, 0)
        f.write(f"{cat}: {avg:.2f}%\n")

    f.write(f"\n=== Total Average ===\n{total_avg:.2f}%")

print(f"Calculations complete. Results saved to {args.output_txt}")
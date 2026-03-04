import json
import os
import re
from collections import defaultdict, Counter
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--hd_epic_database", type=str, required=True, help="Path to HD-EPIC database root.")
parser.add_argument("--task_name", type=str, default="PreprocessedVideos", help="Task name folder.")
parser.add_argument("--multi_run_tasks", nargs='*', type=str, default=[],
                    help="List of task names that have multiple evaluation runs.")
parser.add_argument("--num_runs", type=int, default=5,
                    help="Total number of runs for multi-run tasks (e.g., 5 for runs 0, 1, 2, 3, 4).")
parser.add_argument("--output_predictions_file", type=str, default="predictions.json")
parser.add_argument("--output_results_file", type=str, default="results.txt")

args = parser.parse_args()

# 定义任务到类别的映射
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
MULTI_RUN_TASK_NAMES = tasks  # Default to all tasks being potential multi-run candidates


def letter_to_num(letter):
    return {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}.get(str(letter).upper(), -1)


def parse_evaluation_file(file_path):
    if not os.path.exists(file_path):
        return []
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
    for entry_str in entries:
        if not entry_str.startswith("QID:"):
            continue
        try:
            qid_match = re.search(r'QID:\s*([^,]+)', entry_str)
            pred_match = re.search(r'Predicted:\s*([^,]+)', entry_str)
            gt_match = re.search(r'GT:\s*(\d+)', entry_str)

            if not all([qid_match, pred_match, gt_match]):
                continue

            qid = qid_match.group(1).strip()
            predicted_raw = pred_match.group(1).strip()
            gt = int(gt_match.group(1))

            letter_match = re.match(r'^([A-E])', predicted_raw.upper())
            pred_letter = letter_match.group(1) if letter_match else ''

            is_correct_re_evaluated = (letter_to_num(pred_letter) == gt) if pred_letter else False

            results.append({
                "qid": qid,
                "pred_letter": pred_letter,
                "correct_re_evaluated": is_correct_re_evaluated,
                "gt": gt
            })
        except Exception as e:
            print(f"Parsing failed for entry: {entry_str[:70]}...\nError: {str(e)}")
    return results


def get_voted_prediction(predictions_list):
    valid_predictions = [p for p in predictions_list if p and p in ['A', 'B', 'C', 'D', 'E']]
    if not valid_predictions:
        return ''

    counts = Counter(valid_predictions)
    max_count = 0
    for letter_option in ['A', 'B', 'C', 'D', 'E']:
        if counts[letter_option] > max_count:
            max_count = counts[letter_option]

    tied_winners = [letter_option for letter_option in ['A', 'B', 'C', 'D', 'E'] if counts[letter_option] == max_count]
    return tied_winners[0] if tied_winners else ''


# ... (Main processing logic remains largely identical, referencing base_path derived from args) ...

final_predictions_for_json = {}
task_accuracies = {}
multi_run_intermediate_results = defaultdict(dict)
category_accuracies = defaultdict(list)

print(f"Starting Task Processing. Multi-run tasks enabled: {bool(args.num_runs > 0)}")
print(f"Base Data Path: {base_path}")

for task in tasks:
    task_dir_slug = task.lower().replace(" ", "_")
    category = task_to_category[task]
    dir_name = f"{category.lower()}_{task_dir_slug}"
    task_full_path_dir = os.path.join(base_path, dir_name)

    # 逻辑：如果是多轮评估模式
    if task in MULTI_RUN_TASK_NAMES and args.num_runs > 0:
        print(f"\nProcessing Multi-run Task: {task} ({args.num_runs} runs)")
        qid_data_for_voting = defaultdict(lambda: {'preds': [''] * args.num_runs, 'gt': None})

        task_has_any_data = False
        for run_loop_idx in range(args.num_runs):
            if run_loop_idx == 0:
                result_file_name = "evaluation_results.txt"
            else:
                result_file_name = f"evaluation_results{run_loop_idx}.txt"

            result_path = os.path.join(task_full_path_dir, result_file_name)
            run_display_name = f"run_{run_loop_idx}"

            if not os.path.exists(result_path):
                multi_run_intermediate_results[task][run_display_name] = "N/A (Not Found)"
                continue

            entries_this_run = parse_evaluation_file(result_path)
            if not entries_this_run:
                multi_run_intermediate_results[task][run_display_name] = 0.0
                continue

            task_has_any_data = True
            num_correct_this_run = sum(1 for entry in entries_this_run if entry["correct_re_evaluated"])

            for entry in entries_this_run:
                qid_data_for_voting[entry["qid"]]['preds'][run_loop_idx] = entry["pred_letter"]
                if qid_data_for_voting[entry["qid"]]['gt'] is None:
                    qid_data_for_voting[entry["qid"]]['gt'] = entry["gt"]
                elif qid_data_for_voting[entry["qid"]]['gt'] != entry["gt"]:
                    print(f"Warning: GT mismatch for QID {entry['qid']} in task {task}")

            acc_this_run = (num_correct_this_run / len(entries_this_run) * 100) if entries_this_run else 0.0
            multi_run_intermediate_results[task][run_display_name] = acc_this_run

        if not qid_data_for_voting:
            task_accuracies[task] = 0.0
            multi_run_intermediate_results[task]["voted"] = 0.0
            category_accuracies[category].append(0.0)
            continue

        voted_correct_count = 0
        qids_with_gt_for_voting = 0

        for qid, data in qid_data_for_voting.items():
            if data['gt'] is None:
                final_predictions_for_json[qid] = letter_to_num(get_voted_prediction(data['preds']))
                continue

            qids_with_gt_for_voting += 1
            voted_pred_letter = get_voted_prediction(data['preds'])
            final_predictions_for_json[qid] = letter_to_num(voted_pred_letter)

            if voted_pred_letter and letter_to_num(voted_pred_letter) == data['gt']:
                voted_correct_count += 1

        voted_acc = (voted_correct_count / qids_with_gt_for_voting * 100) if qids_with_gt_for_voting > 0 else 0.0
        task_accuracies[task] = voted_acc
        multi_run_intermediate_results[task]["voted"] = voted_acc
        category_accuracies[category].append(voted_acc)
        print(f"Task: {task}, Voted Acc: {voted_acc:.2f}%")

    else:
        # 单轮逻辑
        result_path = os.path.join(task_full_path_dir, "evaluation_results.txt")
        if not os.path.exists(result_path):
            print(f"Single-run Task {task} file not found: {result_path}")
            task_accuracies[task] = 0.0
            category_accuracies[category].append(0.0)
            continue

        entries = parse_evaluation_file(result_path)
        if not entries:
            task_accuracies[task] = 0.0
            category_accuracies[category].append(0.0)
            continue

        correct_count = sum(1 for e in entries if e["correct_re_evaluated"])
        total_count = len(entries)
        acc = (correct_count / total_count * 100) if total_count > 0 else 0.0
        task_accuracies[task] = acc
        category_accuracies[category].append(acc)

        for e in entries:
            final_predictions_for_json[e["qid"]] = letter_to_num(e["pred_letter"])

# 统计与输出
category_avg = {}
for cat, accs in category_accuracies.items():
    category_avg[cat] = sum(accs) / len(accs) if accs else 0.0

categories_for_total = ["Recipe", "Ingredient", "Nutrition", "Fine_grained", "3D_perception", "Object_motion", "Gaze"]
category_values = [category_avg.get(cat, 0) for cat in categories_for_total if cat in category_avg]
total_avg = sum(category_values) / len(category_values) if category_values else 0.0

with open(args.output_predictions_file, 'w') as f:
    json.dump(final_predictions_for_json, f, indent=2)
print(f"\nPredictions saved to {args.output_predictions_file}")

with open(args.output_results_file, 'w') as f:
    f.write("=== Task Accuracies ===\n")
    sorted_tasks = sorted(tasks, key=lambda t: (task_to_category[t], t))

    for task in sorted_tasks:
        acc = task_accuracies.get(task, 0.0)
        if task in MULTI_RUN_TASK_NAMES and args.num_runs > 0:
            f.write(f"{task} (Multi-run):\n")
            intermediate = multi_run_intermediate_results.get(task, {})
            for i in range(args.num_runs):
                run_key = f"run_{i}"
                run_acc = intermediate.get(run_key, "N/A")
                f.write(f"  Run {i}: {run_acc}\n")
            f.write(f"  Voted: {acc:.2f}%\n")
        else:
            f.write(f"{task}: {acc:.2f}%\n")

    f.write("\n=== Category Averages ===\n")
    for cat in categories_for_total:
        avg = category_avg.get(cat, 0.0)
        f.write(f"{cat}: {avg:.2f}%\n")

    f.write(f"\n=== Total Average ===\n{total_avg:.2f}%")

print(f"Results saved to {args.output_results_file}")
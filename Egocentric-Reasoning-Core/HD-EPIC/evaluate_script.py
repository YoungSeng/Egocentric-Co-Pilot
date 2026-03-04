import argparse
import json
import os
import re
from shutil import copyfile


# ... (Logic functions: letter_to_index, extract_prediction, process_predictions, evaluate, save_backup remain IDENTICAL) ...
# 为了节省篇幅，这里假设中间的逻辑函数与原文件完全一致。
# 关键修改在底部的 main block。

def letter_to_index(letter):
    """将大写字母转换为数字索引 (A->0, B->1, ...)"""
    if isinstance(letter, str) and len(letter) == 1 and 'A' <= letter.upper() <= 'Z':
        return ord(letter.upper()) - ord('A')
    return -1


def extract_prediction(text):
    patterns = [
        r'Answer:\s*([A-E])', r'\*\*([A-E])\*\*', r'\*\*([A-E])\.',
        r'([A-E])\. ', r'\n\n([A-E])', r'Option ([A-E]) '
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[0].upper()
    return None


def process_predictions(raw_predictions):
    processed = {}
    for qid, pred in raw_predictions.items():
        if isinstance(pred, int):
            processed[qid] = pred
        elif isinstance(pred, str) and len(pred) == 1 and pred.upper() in 'ABCDE':
            processed[qid] = pred
        else:
            extracted = extract_prediction(str(pred))
            processed[qid] = extracted if extracted else -1
    return processed


def evaluate(predictions, ground_truth):
    total = 0
    correct = 0
    results = []
    for qid, pred in predictions.items():
        gt = ground_truth.get(qid, {})
        true_idx = gt.get('correct_idx', 'N/A')
        is_correct = False
        try:
            pred_idx = letter_to_index(pred) if isinstance(pred, str) else int(pred)
        except:
            pred_idx = -1
        try:
            true_idx_num = int(true_idx)
        except:
            true_idx_num = -1

        if isinstance(true_idx, int) and (pred_idx == true_idx_num):
            correct += 1
            is_correct = True

        results.append(
            {"question_id": qid, "predicted_index": pred, "correct_index_gt": true_idx, "is_correct": is_correct})
        total += 1

    accuracy = (correct / total * 100) if total > 0 else 0
    return {"accuracy": accuracy, "correct": correct, "total": total, "results": results}


def save_backup(prediction_file):
    base_dir = os.path.dirname(prediction_file)
    file_name = os.path.basename(prediction_file)
    if file_name.startswith("predictions_") and file_name.endswith(".json"):
        task_name = file_name[len("predictions_"):-len(".json")]
        backup_file = os.path.join(base_dir, f"predictions_{task_name}_source.json")
    else:
        backup_file = prediction_file.replace(".json", "_source.json")
    if not os.path.exists(backup_file) and os.path.exists(prediction_file):
        copyfile(prediction_file, backup_file)
    return backup_file


def main(prediction_file, gt_base_dir, output_file, voting_ensemble):
    # 1. Backup
    base_dir = os.path.dirname(prediction_file)
    file_name = os.path.basename(prediction_file)

    if file_name.startswith("predictions_") and file_name.endswith(".json") and "source" not in file_name:
        task_name = file_name[len("predictions_"):-len(".json")]
        backup_file = os.path.join(base_dir, f"predictions_{task_name}_source.json")
    else:
        raise ValueError("Invalid filename format")

    if os.path.exists(backup_file):
        os.remove(backup_file)
        backup_path = save_backup(prediction_file)
        with open(backup_path, 'r') as f:
            raw_preds = json.load(f)
    else:
        backup_path = save_backup(prediction_file)
        print(f"Backup created at: {backup_path}")
        with open(prediction_file, 'r') as f:
            raw_preds = json.load(f)

    clean_preds = process_predictions(raw_preds)

    # 3. Save processed
    with open(prediction_file, 'w') as f:
        json.dump(clean_preds, f, indent=2)

    # 4. Load GT
    task_name_extracted = os.path.basename(prediction_file)[len("predictions_"):-len(".json")]
    if voting_ensemble != "":
        gt_file = os.path.join(gt_base_dir, f"{task_name_extracted[:-1]}.json")
    else:
        gt_file = os.path.join(gt_base_dir, f"{task_name_extracted}.json")

    with open(gt_file, 'r') as f:
        ground_truth = json.load(f)

    # 5. Evaluate
    raw_eval = evaluate(raw_preds, ground_truth)
    clean_eval = evaluate(clean_preds, ground_truth)

    # 6. Report
    with open(output_file, 'w') as f:
        f.write("--- Per-Sample Results ---\n")
        for res in clean_eval['results']:
            f.write(
                f"QID: {res['question_id']},\tPredicted: {res['predicted_index']},\tGT: {res['correct_index_gt']},\tCorrect: {res['is_correct']}\n")
        f.write("\n--- Summary ---\n")
        f.write(f"Total Predictions: {clean_eval['total']}\n")
        f.write(f"Correct Predictions: {clean_eval['correct']}\n")
        f.write(f"Overall Accuracy: {clean_eval['accuracy']:.2f}%\n")
        f.write("\n--- Processing Summary ---\n")
        f.write(f"Original Accuracy: {raw_eval['accuracy']:.2f}%\n")
        f.write(f"Processed Accuracy: {clean_eval['accuracy']:.2f}%\n")

    print(f"Evaluation results saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate prediction file against ground truth.')
    parser.add_argument('prediction_file', help='Path to prediction JSON file')
    parser.add_argument('gt_base_dir', help='Path to ground truth directory')
    parser.add_argument('output_file', help='Path to output results file')
    parser.add_argument("voting_ensemble", type=str, default="", nargs='?')

    args = parser.parse_args()

    main(
        args.prediction_file,
        args.gt_base_dir,
        args.output_file,
        args.voting_ensemble
    )

"""
Example Usage:
python evaluate_script.py \
  "/path/to/predictions/predictions_task_name.json" \
  "/path/to/annotations/" \
  "/path/to/output/evaluation_results.txt"
"""
import argparse
import os
import sys

def write_to_output(output_file, message):
    """Helper function to write messages to the output file."""
    if output_file:
        output_file.write(message + "\n")

def process_evaluation_file(filepath, output_file):
    """
    Processes a single evaluation_results.txt file.
    Checks if predicted values are within {A, B, C, D, E} and writes invalid ones to output_file.
    """
    allowed_predictions = {'A', 'B', 'C', 'D', 'E'}
    print(f"--- Processing file: {filepath} ---", file=sys.stdout)

    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()

                if "QID:" in line and "Predicted:" in line:
                    try:
                        parts = line.split(',')
                        qid = None
                        predicted = None

                        for part in parts:
                            if "QID:" in part:
                                qid_parts = part.split(':', 1)
                                if len(qid_parts) > 1:
                                    qid = qid_parts[1].strip()
                            elif "Predicted:" in part:
                                pred_parts = part.split(':', 1)
                                if len(pred_parts) > 1:
                                    predicted = pred_parts[1].strip()

                        if qid is not None and predicted is not None:
                            if predicted not in allowed_predictions:
                                write_to_output(output_file, "  Invalid Prediction Found:")
                                write_to_output(output_file, f"    File: {filepath}")
                                write_to_output(output_file, f"    QID: {qid}")
                                write_to_output(output_file, f"    Predicted: {predicted}")
                                write_to_output(output_file, "-" * 20)

                    except Exception as e:
                        write_to_output(output_file, f"  Error parsing line in {filepath}: {line}")
                        write_to_output(output_file, f"  Error details: {e}")
                        print(f"  Error parsing line in {filepath}: {line}", file=sys.stderr)
                        print(f"  Error details: {e}", file=sys.stderr)

    except FileNotFoundError:
        write_to_output(output_file, f"  Warning: File not found: {filepath}")
        print(f"  Warning: File not found: {filepath}", file=sys.stderr)
    except Exception as e:
        write_to_output(output_file, f"  Error reading file {filepath}: {e}")
        print(f"  Error reading file {filepath}: {e}", file=sys.stderr)

    print(f"--- Finished processing file: {os.path.basename(filepath)} ---", file=sys.stdout)

def main():
    parser = argparse.ArgumentParser(description="Process evaluation results files in task subdirectories and save invalid predictions to a file.")
    # CHANGED: Default path is now relative
    parser.add_argument("--hd_epic_database", type=str,
                        default="./HD-EPIC/",
                        help="Base directory for the HD-EPIC dataset.")
    parser.add_argument("--task_name", type=str,
                        default="PreprocessedVideos",
                        help="Name of the main task directory within hd_epic_database.")
    parser.add_argument("--output_file", type=str,
                        default="invalid_predictions.txt",
                        help="Path to the output file where invalid predictions will be saved.")

    args = parser.parse_args()

    base_dir = args.hd_epic_database
    task_subdir_name = args.task_name
    output_file_path = args.output_file

    full_task_path = os.path.join(base_dir, task_subdir_name)

    if not os.path.exists(full_task_path):
        print(f"Error: Task directory not found: {full_task_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(full_task_path):
        print(f"Error: Task path is not a directory: {full_task_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning for subdirectories containing 'evaluation_results.txt' within: {full_task_path}", file=sys.stdout)
    print(f"Saving invalid predictions to: {output_file_path}", file=sys.stdout)
    print("-" * 40, file=sys.stdout)

    found_files = False

    try:
        with open(output_file_path, 'w') as outfile:
            for item_name in os.listdir(full_task_path):
                item_path = os.path.join(full_task_path, item_name)

                if os.path.isdir(item_path):
                    evaluation_file_path = os.path.join(item_path, "evaluation_results.txt")

                    if os.path.exists(evaluation_file_path) and os.path.isfile(evaluation_file_path):
                        found_files = True
                        process_evaluation_file(evaluation_file_path, outfile)
                        print("-" * 40, file=sys.stdout)

    except IOError as e:
        print(f"Error: Could not write to output file {output_file_path}: {e}", file=sys.stderr)
        sys.exit(1)

    if not found_files:
        print(f"No 'evaluation_results.txt' files found in any subdirectory of {full_task_path}.", file=sys.stdout)

if __name__ == "__main__":
    main()
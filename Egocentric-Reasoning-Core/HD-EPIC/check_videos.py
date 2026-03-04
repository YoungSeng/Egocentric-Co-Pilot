import os
import pathlib
import argparse


def check_files_for_content(directory_path: str, target_content: str):
    """
    Check all files in a directory. If content matches target_content, print the path.
    """
    dir_path = pathlib.Path(directory_path)

    if not dir_path.exists():
        print(f"Error: Directory not found: {directory_path}")
        return
    if not dir_path.is_dir():
        print(f"Error: Path is not a directory: {directory_path}")
        return

    print(f"Checking directory: {directory_path} for content: '{target_content}'")

    found_count = 0
    for item in dir_path.iterdir():
        if item.is_file():
            try:
                with open(item, 'r', encoding='utf-8') as f:
                    content = f.read().strip()

                if content == target_content:
                    print(f"Match found: {item}")
                    found_count += 1
            except Exception as e:
                print(f"Error reading file {item}: {e}")

    if found_count == 0:
        print(f"No files found with content: '{target_content}'")
    else:
        print(f"Check complete. Found {found_count} matching files.")


def main():
    parser = argparse.ArgumentParser(description="Utility script to check file contents in a directory.")
    parser.add_argument("--dir", type=str, required=True, help="Directory to check.")
    parser.add_argument("--content", type=str, default="无法生成描述", help="Content string to search for.")

    args = parser.parse_args()

    check_files_for_content(args.dir, args.content)


if __name__ == "__main__":
    main()
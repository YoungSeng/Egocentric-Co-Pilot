import os
import subprocess
import argparse


def convert_video_to_mp4(input_path, output_path):
    """
    Converts a video file to MP4 format using ffmpeg.
    The command is adapted for full video conversion based on the user's template.
    """
    # For full video conversion, start from the beginning.
    start_time = 0

    command = [
        'ffmpeg',
        '-y',  # Overwrite output files without asking
        '-ss', str(start_time),  # Start processing from this time
        '-i', input_path,  # Input file
        '-c', 'copy',  # Copy all streams by default (audio, subtitles, etc.)
        '-c:v', 'libx264',  # Re-encode video stream with libx264 (H.264)
        output_path  # Output file
    ]

    print(f"Attempting to convert: {input_path}")
    print(f"To: {output_path}")
    # print(f"Executing command: {' '.join(command)}")

    try:
        # Check if ffmpeg is available first
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)

        process = subprocess.run(command, capture_output=True, text=True, check=True)
        print(f"Successfully converted {input_path} to {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error converting {input_path}:")
        print(f"  Return code: {e.returncode}")
        print(f"  Stderr:\n{e.stderr}")
        return False
    except FileNotFoundError:
        print("Error: ffmpeg command not found. Please ensure ffmpeg is installed and in your system's PATH.")
        return None


def main():
    parser = argparse.ArgumentParser(description="Convert videos to MP4 format using ffmpeg.")
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory containing raw videos.")
    parser.add_argument("--extensions", nargs='+', default=['.mkv', '.webm'],
                        help="List of file extensions to convert.")

    args = parser.parse_args()

    root_video_dir = args.root_dir
    valid_extensions_to_convert = args.extensions

    converted_count = 0
    skipped_count = 0
    error_count = 0

    if not os.path.isdir(root_video_dir):
        print(f"Error: Root directory '{root_video_dir}' not found.")
        return

    print(f"Starting video conversion in: {root_video_dir}")
    print(f"Looking for files with extensions: {', '.join(valid_extensions_to_convert)}")

    for dirpath, _, filenames in os.walk(root_video_dir):
        for filename in filenames:
            base, ext = os.path.splitext(filename)
            ext_lower = ext.lower()

            if ext_lower in valid_extensions_to_convert:
                input_file_path = os.path.join(dirpath, filename)
                output_filename = base + ".mp4"
                output_file_path = os.path.join(dirpath, output_filename)

                print("-" * 30)
                if os.path.exists(output_file_path):
                    print(f"Output file {output_file_path} already exists. Skipping.")
                    skipped_count += 1
                    continue

                conversion_result = convert_video_to_mp4(input_file_path, output_file_path)
                if conversion_result is True:
                    converted_count += 1
                elif conversion_result is False:
                    error_count += 1
                elif conversion_result is None:  # ffmpeg not found
                    print("Aborting due to ffmpeg not being found.")
                    return  # Stop the script entirely

    print("\n" + "=" * 30)
    print("Conversion Summary:")
    print(f"  Successfully converted: {converted_count}")
    print(f"  Skipped (already exist): {skipped_count}")
    print(f"  Errors during conversion: {error_count}")
    print("=" * 30)


if __name__ == "__main__":
    main()
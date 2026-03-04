import json
import os
import subprocess
import argparse
import sys

def process_youcookii(args):
    """
    Process YouCookII dataset to extract video segments and descriptions.
    """
    limit = args.limit
    annotations_file = args.annotations_file
    raw_videos_base_dir = args.raw_videos_dir
    processed_data_base_dir = args.output_dir

    print(f"开始处理YouCookII数据集，限制处理 {limit} 个视频 (设置为None或负数则处理所有)。")
    print(f"读取标注文件: {annotations_file}")

    try:
        with open(annotations_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 标注文件未找到，请确认路径是否正确: {annotations_file}")
        return
    except json.JSONDecodeError:
        print(f"错误: 无法解析JSON文件，请确认文件格式是否正确: {annotations_file}")
        return

    if 'database' not in data:
        print("错误: JSON文件中找不到'database'键。")
        return

    database = data['database']
    video_ids = list(database.keys())
    print(f"总共找到 {len(video_ids)} 个视频条目需要处理。")

    processed_count = 0

    os.makedirs(processed_data_base_dir, exist_ok=True)

    for video_id in video_ids:
        if limit is not None and processed_count >= limit:
            print(f"已达到处理数量限制 ({limit} 个视频)，停止处理。")
            break

        video_info = database[video_id]
        subset = video_info.get('subset') # 'training' or 'validation'
        recipe_type = video_info.get('recipe_type') # e.g., '113'
        annotations = video_info.get('annotations', [])

        if not subset or not recipe_type:
            print(f"警告: 视频 {video_id} 缺少 'subset' 或 'recipe_type' 信息，跳过。")
            continue

        # Original structure: RAW_VIDEOS_BASE_DIR / subset / recipe_type / video_id.mp4
        raw_video_path = os.path.join(raw_videos_base_dir, subset, recipe_type, f"{video_id}.mp4")

        output_video_dir = os.path.join(processed_data_base_dir, video_id)
        os.makedirs(output_video_dir, exist_ok=True)

        print(f"\n处理视频: {video_id} ( {processed_count + 1} / {len(video_ids)} )")
        print(f"  原始视频路径尝试: {raw_video_path}")

        if not os.path.exists(raw_video_path):
            print(f"  警告: 原始视频文件未找到，跳过该视频: {raw_video_path}")
            continue

        if not annotations:
            print(f"  警告: 视频 {video_id} 没有标注信息，跳过视频裁剪和文本保存。")
            processed_count += 1
            continue

        sentences = []

        for i, annotation in enumerate(annotations):
            segment = annotation.get('segment') # [start_time, end_time]
            sentence = annotation.get('sentence')
            segment_id = annotation.get('id', i)

            if segment is None or sentence is None or len(segment) != 2:
                print(f"  警告: 视频 {video_id} 的标注 {i} 格式不正确 ({annotation})，跳过该标注。")
                continue

            start_time, end_time = segment
            start_time_str = str(start_time)
            end_time_str = str(end_time)

            output_segment_path = os.path.join(output_video_dir, f"segment_{segment_id}.mp4")

            ffmpeg_command = [
                'ffmpeg',
                '-i', raw_video_path,
                '-ss', start_time_str,
                '-to', end_time_str,
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-y',
                '-loglevel', 'warning',
                output_segment_path
            ]

            print(f"    裁剪片段 {segment_id} ({start_time_str}s - {end_time_str}s) 到 {output_segment_path}")

            try:
                subprocess.run(ffmpeg_command, check=True)
                sentences.append(sentence)
                print(f"    裁剪成功.")

            except FileNotFoundError:
                print(f"    错误: ffmpeg命令未找到。请确认ffmpeg已安装并添加到系统PATH中。")
                sys.exit("FFmpeg not found. Please install it.")
            except subprocess.CalledProcessError as e:
                print(f"    错误: 执行ffmpeg命令失败，无法处理片段 {segment_id}: {e}")
                continue
            except Exception as e:
                 print(f"    发生未知错误处理片段 {segment_id}: {e}")
                 continue

        if sentences:
            sentences_file_path = os.path.join(output_video_dir, "sentences.txt")
            try:
                with open(sentences_file_path, 'w', encoding='utf-8') as f:
                    for sentence in sentences:
                        f.write(sentence + '\n')
                print(f"  所有句子已保存到: {sentences_file_path}")
            except Exception as e:
                print(f"  错误: 无法写入句子文件 {sentences_file_path}: {e}")
        else:
             print(f"  视频 {video_id} 没有成功处理的标注，未生成 sentences.txt 文件。")

        processed_count += 1
        print(f"  视频 {video_id} 处理完成.")

    print("\n数据集处理完毕。")
    print(f"总共处理了 {processed_count} 个视频。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="处理YouCookII数据集，提取视频片段和文本。")
    parser.add_argument(
        '--limit',
        type=int,
        default=1,
        help="限制处理的视频数量。设置为None或负数则处理所有视频。"
    )
    # CHANGED: Added arguments for paths with relative defaults
    parser.add_argument('--annotations_file', type=str,
                        default="./YouCookII/annotations/youcookii_annotations_trainval.json",
                        help="Path to the JSON annotations file.")
    parser.add_argument('--raw_videos_dir', type=str,
                        default="./YouCookII/raw_videos/",
                        help="Base directory containing raw videos.")
    parser.add_argument('--output_dir', type=str,
                        default="./dataset/processed/YouCookII/",
                        help="Output directory for processed data.")

    args = parser.parse_args()

    # If limit < 0, set to None
    if args.limit < 0:
        args.limit = None

    process_youcookii(args)
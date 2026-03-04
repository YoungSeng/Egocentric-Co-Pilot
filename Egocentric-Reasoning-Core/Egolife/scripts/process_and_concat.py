# import os
# import json
# import glob
# from tqdm import tqdm
# from moviepy import VideoFileClip, concatenate_videoclips
# import collections
#
# # --- 1. 配置 ---
#
# # 包含所有视频组的根目录 (例如, 内含 P01, P02, ... 等文件夹)
# # 这是你提供的路径，请确认是否正确
# VIDEO_ROOT_DIR = "/mnt/data_3/home_aiglasses/EgoLife/videos/"
#
# # 你已经生成的、包含每个短视频caption的JSON文件路径
# EXISTING_CAPTIONS_JSON = "./video_captions_native.json"
#
# # --- 输出配置 ---
#
# # 拼接后的大视频文件的保存目录
# OUTPUT_VIDEO_DIR = "./concatenated_videos/"
#
# # 提取出的前10分钟caption的新JSON文件路径
# OUTPUT_CAPTIONS_JSON = "./aggregated_captions_first_10_min.json"
#
# # 目标拼接时长（分钟）
# TARGET_DURATION_MINUTES = 10
# TARGET_DURATION_SECONDS = TARGET_DURATION_MINUTES * 60
#
#
# # --- 2. 主逻辑 ---
#
# def process_videos_and_captions():
#     """
#     主函数，用于拼接视频并聚合对应的caption。
#     """
#     print("--- 开始处理视频拼接和Caption聚合任务 ---")
#
#     # --- 准备工作 ---
#     # 创建输出目录
#     os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)
#     print(f"拼接视频将保存到: {OUTPUT_VIDEO_DIR}")
#
#     # 加载已有的captions
#     try:
#         with open(EXISTING_CAPTIONS_JSON, 'r', encoding='utf-8') as f:
#             existing_captions_data = json.load(f)
#         print(f"成功从 '{EXISTING_CAPTIONS_JSON}' 加载了 {len(existing_captions_data)} 条caption。")
#     except FileNotFoundError:
#         print(f"[严重错误] Caption文件未找到: {EXISTING_CAPTIONS_JSON}")
#         print("请确保你已经运行了VLM脚本并生成了此文件。")
#         return
#     except json.JSONDecodeError:
#         print(f"[严重错误] Caption文件 '{EXISTING_CAPTIONS_JSON}' 格式错误，无法解析。")
#         return
#
#     # --- 识别视频分组 ---
#     # 我们假设每个直接在 VIDEO_ROOT_DIR下的子目录都是一个独立的视频组 (如 P01, P02)
#     try:
#         # 【修改点1】: 增加 not d.startswith('.') 来过滤掉隐藏目录
#         group_dirs = [d for d in os.listdir(VIDEO_ROOT_DIR) if
#                       os.path.isdir(os.path.join(VIDEO_ROOT_DIR, d)) and not d.startswith('.')]
#     except FileNotFoundError:
#         print(f"[严重错误] 视频根目录未找到: {VIDEO_ROOT_DIR}")
#         return
#
#     if not group_dirs:
#         print(f"[错误] 在 '{VIDEO_ROOT_DIR}' 中没有找到任何视频分组目录。")
#         return
#
#     print(f"找到 {len(group_dirs)} 个视频分组: {', '.join(group_dirs)}")
#
#     # --- 循环处理每个分组 ---
#     aggregated_captions_result = collections.OrderedDict()
#
#     for group_id in tqdm(group_dirs, desc="处理视频分组"):
#         group_video_dir = os.path.join(VIDEO_ROOT_DIR, group_id)
#
#         # 【修改点2】: 使用 '**' 和 recursive=True 进行递归搜索
#         video_files = sorted(glob.glob(os.path.join(group_video_dir, '**', '*.mp4'), recursive=True))
#
#         if not video_files:
#             print(f"\n[警告] 分组 '{group_id}' 中没有找到.mp4文件，跳过。")
#             continue
#
#         # --- 筛选前10分钟的视频片段 ---
#         clips_to_concat = []
#         captions_for_this_group = []
#         current_duration = 0.0
#
#         print(f"\n正在为分组 '{group_id}' 筛选前 {TARGET_DURATION_MINUTES} 分钟的视频...")
#
#         for video_path in video_files:
#             if current_duration >= TARGET_DURATION_SECONDS:
#                 break  # 已达到目标时长
#
#             try:
#                 # 获取视频时长
#                 with VideoFileClip(video_path) as clip:
#                     clip_duration = clip.duration
#
#                 # 添加到待处理列表
#                 clips_to_concat.append(video_path)
#                 current_duration += clip_duration
#
#                 # 提取对应的caption
#                 video_filename = os.path.basename(video_path)
#                 caption = existing_captions_data.get(video_filename)
#                 if caption:
#                     captions_for_this_group.append(caption)
#                 else:
#                     print(f"  [警告] 在JSON文件中未找到 '{video_filename}' 的caption。")
#                     captions_for_this_group.append(f"Caption not found for {video_filename}")
#
#             except Exception as e:
#                 print(f"  [错误] 处理文件 '{video_path}' 失败: {e}")
#
#         if not clips_to_concat:
#             print(f"分组 '{group_id}' 中没有可处理的视频，跳过。")
#             continue
#
#         print(f"为 '{group_id}' 选择了 {len(clips_to_concat)} 个视频片段，总时长约 {current_duration:.2f} 秒。")
#
#         # --- 1. 拼接视频 ---
#         try:
#             print(f"正在拼接 '{group_id}' 的视频...")
#             # 从路径列表加载VideoFileClip对象
#             video_clip_objects = [VideoFileClip(path) for path in clips_to_concat]
#
#             # 拼接
#             final_clip = concatenate_videoclips(video_clip_objects, method="compose")
#
#             # 定义输出路径
#             output_video_path = os.path.join(OUTPUT_VIDEO_DIR, f"{group_id}_first_{TARGET_DURATION_MINUTES}min.mp4")
#
#             # 写入文件
#             final_clip.write_videofile(output_video_path, codec="libx264", audio_codec="aac")
#
#             # 释放资源
#             final_clip.close()
#             for clip in video_clip_objects:
#                 clip.close()
#
#             print(f"视频拼接成功，已保存至: {output_video_path}")
#
#         except Exception as e:
#             print(f"\n  [严重错误] 拼接分组 '{group_id}' 的视频失败: {e}")
#
#         # --- 2. 保存聚合的Caption ---
#         aggregated_captions_result[group_id] = captions_for_this_group
#
#     # --- 最终保存所有聚合的Caption ---
#     print("\n所有分组处理完毕，正在保存聚合的Caption...")
#     try:
#         with open(OUTPUT_CAPTIONS_JSON, 'w', encoding='utf-8') as f:
#             json.dump(aggregated_captions_result, f, indent=4, ensure_ascii=False)
#         print(f"所有聚合的caption已保存到: {OUTPUT_CAPTIONS_JSON}")
#     except Exception as e:
#         print(f"[严重错误] 保存最终caption JSON文件失败: {e}")
#
#     print("\n--- 任务完成 ---")
#
#
# if __name__ == "__main__":
#     process_videos_and_captions()


'''
import os
import json
import glob
from moviepy import VideoFileClip, concatenate_videoclips

# --- 1. 配置 ---

# 【固定】要处理的视频文件夹路径
VIDEO_FOLDER_PATH = "/mnt/data_3/home_aiglasses/EgoLife/videos/A1_JAKE/DAY1/"

# 【固定】包含所有caption的JSON文件路径
EXISTING_CAPTIONS_JSON = "/home/sicheng/Desktop/mycode/EgoLife/video_captions_native.json"

# 【固定】要处理的视频数量
NUM_VIDEOS_TO_PROCESS = 20

# --- 输出配置 ---

# 拼接后的大视频文件保存路径
OUTPUT_VIDEO_PATH = "./A1_JAKE_DAY1_first_20_clips.mp4"

# 提取出的caption的新JSON文件路径
OUTPUT_CAPTIONS_JSON = "./A1_JAKE_DAY1_first_20_captions.json"


# --- 2. 主逻辑 ---

def process_specific_folder():
    """
    处理指定文件夹中的前N个视频，拼接并提取caption。
    """
    print("--- 开始处理特定文件夹的视频和Caption ---")
    print(f"目标文件夹: {VIDEO_FOLDER_PATH}")
    print(f"要处理的视频数量: {NUM_VIDEOS_TO_PROCESS}")

    # --- 加载已有的captions ---
    try:
        with open(EXISTING_CAPTIONS_JSON, 'r', encoding='utf-8') as f:
            all_captions_data = json.load(f)
        print(f"成功从 '{EXISTING_CAPTIONS_JSON}' 加载了 {len(all_captions_data)} 条caption。")
    except FileNotFoundError:
        print(f"[严重错误] Caption文件未找到: {EXISTING_CAPTIONS_JSON}")
        return
    except json.JSONDecodeError:
        print(f"[严重错误] Caption文件格式错误，无法解析。")
        return

    # --- 查找并筛选视频文件 ---
    try:
        # 查找所有mp4文件，并按文件名排序以保证时序
        all_video_files = sorted(glob.glob(os.path.join(VIDEO_FOLDER_PATH, '*.mp4')))

        if not all_video_files:
            print(f"[错误] 在 '{VIDEO_FOLDER_PATH}' 中没有找到任何.mp4文件。")
            return

        # 截取前 N 个视频
        videos_to_process = all_video_files[:NUM_VIDEOS_TO_PROCESS]

        if len(videos_to_process) < NUM_VIDEOS_TO_PROCESS:
            print(
                f"[警告] 文件夹中只有 {len(videos_to_process)} 个视频，少于目标的 {NUM_VIDEOS_TO_PROCESS} 个。将处理所有找到的视频。")

        print(f"已选定 {len(videos_to_process)} 个视频进行处理。")

    except FileNotFoundError:
        print(f"[严重错误] 视频文件夹未找到: {VIDEO_FOLDER_PATH}")
        return

    # --- 1. 拼接视频 ---
    try:
        print("正在拼接视频...")
        # 从路径列表加载VideoFileClip对象
        video_clip_objects = [VideoFileClip(path) for path in videos_to_process]

        # 拼接
        final_clip = concatenate_videoclips(video_clip_objects, method="compose")

        # 写入文件
        final_clip.write_videofile(OUTPUT_VIDEO_PATH, codec="libx264", audio_codec="aac", logger='bar')

        # 释放资源
        final_clip.close()
        for clip in video_clip_objects:
            clip.close()

        print(f"\n视频拼接成功，已保存至: {OUTPUT_VIDEO_PATH}")

    except Exception as e:
        print(f"\n[严重错误] 拼接视频失败: {e}")
        return  # 如果视频拼接失败，后续也没必要进行了

    # --- 2. 提取并保存对应的Caption ---
    extracted_captions = []
    print("正在提取对应的captions...")
    for video_path in videos_to_process:
        video_filename = os.path.basename(video_path)
        caption = all_captions_data.get(video_filename)

        if caption:
            extracted_captions.append(caption)
        else:
            print(f"  [警告] 在JSON文件中未找到 '{video_filename}' 的caption。")
            # 添加一个占位符，以保持caption列表和视频列表长度一致
            extracted_captions.append(f"Caption not found for {video_filename}")

    # 将提取的caption保存到新的JSON文件中
    try:
        with open(OUTPUT_CAPTIONS_JSON, 'w', encoding='utf-8') as f:
            # 直接将caption列表存入json
            json.dump(extracted_captions, f, indent=4, ensure_ascii=False)
        print(f"提取的 {len(extracted_captions)} 条caption已保存到: {OUTPUT_CAPTIONS_JSON}")
    except Exception as e:
        print(f"[严重错误] 保存caption JSON文件失败: {e}")

    print("\n--- 任务完成 ---")


if __name__ == "__main__":
    process_specific_folder()
'''

import os
import json
import glob
import subprocess

# --- 1. 配置 ---

# 【固定】要处理的视频文件夹路径
VIDEO_FOLDER_PATH = "/mnt/data_3/home_aiglasses/EgoLife/videos/A1_JAKE/DAY1/"

# 【固定】包含所有caption的JSON文件路径
EXISTING_CAPTIONS_JSON = "/home/sicheng/Desktop/mycode/EgoLife/video_captions_native.json"

# 【固定】要处理的视频数量
NUM_VIDEOS_TO_PROCESS = 20

# --- 输出配置 ---

# 拼接后的大视频文件保存路径
OUTPUT_VIDEO_PATH = "./A1_JAKE_DAY1_first_20_clips.mp4"

# 提取出的caption的新JSON文件路径
OUTPUT_CAPTIONS_JSON = "./A1_JAKE_DAY1_first_20_captions.json"


# --- 2. 主逻辑 ---

def process_with_ffmpeg():
    """
    使用 ffmpeg 高效拼接指定文件夹中的前N个视频，并提取caption。
    """
    print("--- 开始使用 FFmpeg 高效处理视频和Caption ---")
    print(f"目标文件夹: {VIDEO_FOLDER_PATH}")
    print(f"要处理的视频数量: {NUM_VIDEOS_TO_PROCESS}")

    # --- 加载已有的captions (这部分不变) ---
    try:
        with open(EXISTING_CAPTIONS_JSON, 'r', encoding='utf-8') as f:
            all_captions_data = json.load(f)
        print(f"成功从 '{EXISTING_CAPTIONS_JSON}' 加载了 {len(all_captions_data)} 条caption。")
    except FileNotFoundError:
        print(f"[严重错误] Caption文件未找到: {EXISTING_CAPTIONS_JSON}")
        return
    except json.JSONDecodeError:
        print(f"[严重错误] Caption文件格式错误，无法解析。")
        return

    # --- 查找并筛选视频文件 ---
    try:
        all_video_files = sorted(glob.glob(os.path.join(VIDEO_FOLDER_PATH, '*.mp4')))
        if not all_video_files:
            print(f"[错误] 在 '{VIDEO_FOLDER_PATH}' 中没有找到任何.mp4文件。")
            return

        videos_to_process = all_video_files[:NUM_VIDEOS_TO_PROCESS]
        if len(videos_to_process) < NUM_VIDEOS_TO_PROCESS:
            print(f"[警告] 文件夹中只有 {len(videos_to_process)} 个视频，将处理所有找到的视频。")

        print(f"已选定 {len(videos_to_process)} 个视频进行处理。")
    except FileNotFoundError:
        print(f"[严重错误] 视频文件夹未找到: {VIDEO_FOLDER_PATH}")
        return

    # --- 1. 使用 FFmpeg 进行视频拼接 ---

    # 临时文件名，用于存放待拼接的文件列表
    file_list_path = "ffmpeg_file_list.txt"

    try:
        print("正在准备 FFmpeg 文件列表...")
        # 为ffmpeg创建一个文件列表，格式为: file '/path/to/video1.mp4'
        with open(file_list_path, 'w') as f:
            for video_path in videos_to_process:
                # 注意：如果路径中包含特殊字符或空格，需要转义。
                # Python的 repr() 函数可以很好地处理这个问题，但ffmpeg可能不认\。
                # 最安全的方式是确保路径本身不包含单引号。
                f.write(f"file '{video_path}'\n")

        print("正在执行 FFmpeg 拼接命令...")
        # 构建ffmpeg命令
        # -f concat: 使用concat demuxer
        # -safe 0: 允许使用绝对路径 (如果你的文件列表里是绝对路径)
        # -i: 输入文件列表
        # -c copy: 直接复制流，不重新编码，速度极快
        # -y: 如果输出文件已存在，则覆盖
        command = [
            'ffmpeg',
            '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', file_list_path,
            '-c', 'copy',
            OUTPUT_VIDEO_PATH
        ]

        # 执行命令
        # 使用 subprocess.run，它会等待命令执行完毕
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        # 如果需要看到ffmpeg的实时输出，可以去掉 capture_output=True
        # result = subprocess.run(command, check=True)

        print(f"FFmpeg 拼接成功，已保存至: {OUTPUT_VIDEO_PATH}")
        # print("FFmpeg 输出:\n", result.stdout) # 如果需要看ffmpeg的输出可以取消这行注释

    except FileNotFoundError:
        print("\n[严重错误] 'ffmpeg' 命令未找到。请确保 FFmpeg 已安装并存在于系统的PATH中。")
        print("在Ubuntu/Debian上, 你可以运行: sudo apt-get install ffmpeg")
        return
    except subprocess.CalledProcessError as e:
        # 如果ffmpeg命令执行失败 (返回非零退出码)
        print("\n[严重错误] FFmpeg 执行失败。")
        print(f"返回码: {e.returncode}")
        print("FFmpeg 错误输出 (stderr):\n", e.stderr)
        return
    except Exception as e:
        print(f"\n[严重错误] 拼接视频时发生未知错误: {e}")
        return
    finally:
        # 无论成功与否，都清理掉临时文件
        if os.path.exists(file_list_path):
            os.remove(file_list_path)
            print(f"已删除临时文件: {file_list_path}")

    # --- 2. 提取并保存对应的Caption (这部分不变) ---
    extracted_captions = []
    print("正在提取对应的captions...")
    for video_path in videos_to_process:
        video_filename = os.path.basename(video_path)
        caption = all_captions_data.get(video_filename)
        if caption:
            extracted_captions.append(caption)
        else:
            print(f"  [警告] 在JSON文件中未找到 '{video_filename}' 的caption。")
            extracted_captions.append(f"Caption not found for {video_filename}")

    try:
        with open(OUTPUT_CAPTIONS_JSON, 'w', encoding='utf-8') as f:
            json.dump(extracted_captions, f, indent=4, ensure_ascii=False)
        print(f"提取的 {len(extracted_captions)} 条caption已保存到: {OUTPUT_CAPTIONS_JSON}")
    except Exception as e:
        print(f"[严重错误] 保存caption JSON文件失败: {e}")

    print("\n--- 任务完成 ---")


if __name__ == "__main__":
    process_with_ffmpeg()
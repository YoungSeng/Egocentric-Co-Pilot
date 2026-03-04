import os
import sys
import time
import datetime
import re
import subprocess
from collections import defaultdict

# 第三方库
import cv2
import numpy as np
import requests
from ultralytics import YOLO, FastSAM

# 本地模块
# 请确保 vision_core.py 与本脚本在同一目录下
from vision_core import (
    ChessboardMemory, mask_data_to_contour, detect_and_analyze_board,
    generate_chessboard, cls_mapping, filter_pieces_on_board,
    board_to_vector_numpy, vector_to_board_numpy, vis_board,
    board_to_fen, get_chinese_move_notation
)

# --- 全局路径配置 ---
# 获取当前脚本所在的绝对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 模型路径配置
MODEL_DIR = os.path.join(BASE_DIR, "pre-trained-models")
PIECE_MODEL_PATH = os.path.join(MODEL_DIR, "piece.pt")
BOARD_MODEL_PATH = os.path.join(MODEL_DIR, "board.pt")

# 引擎路径配置
ENGINE_PATH = os.path.join(BASE_DIR, "ELEEYE.EXE")

# 日志与临时文件配置
LOG_FILENAME = os.path.join(BASE_DIR, "chess_log.txt")
TEMP_COMMAND_FILE = os.path.join(BASE_DIR, "test_commands.txt")
TEMP_BOARD_IMG = os.path.join(BASE_DIR, "board_refine.jpg")


# --------------------

def filter_moves_by_highest_rank(result_string):
    """
    从结果字符串中提取具有最高 rank 的最多前3个走法，并按指定格式返回。
    """
    moves = result_string.split('|')
    highest_rank = -1  # 初始化最高 rank 为一个较小的值

    # 找到最高 rank
    for move_data in moves:
        parts = move_data.split(',')
        move_dict = {}
        for part in parts:
            if ':' in part:
                key, value = part.split(':', 1)
                move_dict[key.strip()] = value.strip()
        if 'rank' in move_dict:
            try:
                rank = int(move_dict['rank'])
                highest_rank = max(highest_rank, rank)
            except ValueError:
                continue

    filtered_moves = []
    count = 0

    # 筛选最高 rank 的走法
    for move_data in moves:
        parts = move_data.split(',')
        move_dict = {}
        for part in parts:
            if ':' in part:
                key, value = part.split(':', 1)
                move_dict[key.strip()] = value.strip()

        if 'rank' in move_dict:
            try:
                rank = int(move_dict['rank'])
                score = int(move_dict['score'])
                if rank == highest_rank:
                    filtered_moves.append(move_dict)
                    count += 1
                    # if count >= 2:
                    #     break  # 最多返回3个走法
            except ValueError:
                continue

    output_strings = []
    for move_dict in filtered_moves:
        output_strings.append(
            f"move:{move_dict.get('move', '')},"
            f"score:{move_dict.get('score', '')},"
            # f"rank:{move_dict.get('rank', '')},"
            # f"note:{move_dict.get('note', '')},"
            f"winrate:{move_dict.get('winrate', '')}"
        )

    return output_strings


def get_best_move(board_fen, side_to_move=None):
    """
    获取最佳着法
    """
    # 云库 API 的基本 URL
    base_url = "http://www.chessdb.cn/chessdb.php"

    # 设置参数
    params = {
        "action": "queryall",  # 获取最佳着法
        "board": board_fen,  # 当前棋局 FEN 表示
        # "w": side_to_move,   # 哪一方的回合
    }

    try:
        # 发送请求到云库 API
        response = requests.get(base_url, params=params)

        # 检查返回结果
        if response.status_code == 200:
            result = response.text
            print(result)
            if "nobestmove" in result:
                return "没有最佳着法"
            elif "move" in result:
                filtered_result_highest_rank = filter_moves_by_highest_rank(result)
                return filtered_result_highest_rank
            elif "unknown" in result:
                return "该局面未被收录，请尝试其他走法"
            elif "invalid board" in result:
                return "未知错误"
            else:
                return "其他错误"
        else:
            return "请求失败"
    except Exception as e:
        print(f"API Request Error: {e}")
        return "请求失败"


class Chinese_Chessboard_New():
    def __init__(self, memory_length=60, stability_threshold_ratio=0.75):
        # 使用配置好的路径加载模型
        if not os.path.exists(PIECE_MODEL_PATH) or not os.path.exists(BOARD_MODEL_PATH):
            print(f"警告: 模型文件不存在，请检查路径:\n{PIECE_MODEL_PATH}\n{BOARD_MODEL_PATH}")

        self.model_piece = YOLO(PIECE_MODEL_PATH)  # train6 | train7
        self.model_board = YOLO(BOARD_MODEL_PATH)

        self.chessboard_memory = ChessboardMemory(memory_length=memory_length,
                                                  stability_threshold_ratio=stability_threshold_ratio)

        # 初始化存储所有棋局状态的列表
        self.tmp_board = None
        self.all_game_states = []

        frame_width = 640
        frame_height = 480
        self.frame_size = (frame_width, frame_height)
        self.last_change_time = time.time()  # 初始化上次状态变化时间为 None
        self.chess_strategy_translate = None
        self.fen_board = None
        self.board_change = True
        self.log_filename = LOG_FILENAME  # 定义日志文件名

    def add_picture(self, orig_img):
        try:
            results_board = self.model_board(orig_img, save=False, save_txt=False, show=False)
            cls_board = results_board[0].boxes.cls.cpu().numpy()

            if len(cls_board) < 4:
                raise ValueError("未检测到棋盘四个角点")
            else:
                xywh_board = results_board[0].boxes.xywh.cpu().numpy()
                class_xyxy_mapping = defaultdict(list)
                for i in range(len(cls_board)):
                    cls_index = int(cls_board[i])  # 将类别索引转换为整数，用作字典的键
                    xyxy_coords = xywh_board[i][:2]  # 获取对应的 xyxy 坐标
                    class_xyxy_mapping[cls_index].append(xyxy_coords)

                board_contour = np.array([class_xyxy_mapping[0][0], class_xyxy_mapping[1][0], class_xyxy_mapping[3][0],
                                          class_xyxy_mapping[2][0]])  # 20250225

                results = self.model_piece(orig_img, save=False, save_txt=False, show=False)  # True

                orig_img_ = orig_img.copy()

                for i, result in enumerate(results):
                    cls = result.boxes.cls.cpu().numpy()
                    xyxy = result.boxes.xyxy.cpu().numpy()
                    # 中间结果保存
                    intermediate_results = []
                    # 循环遍历每个检测结果
                    for i, cls_id in enumerate(cls):
                        if cls_id == 0:
                            continue  # 跳过棋盘（qipan）

                        # 获取边界框坐标
                        x1, y1, x2, y2 = xyxy[i]

                        # 计算中心点
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2

                        # 获取棋子名称
                        piece_name = cls_mapping[int(cls_id)]

                        # 保存中间结果
                        intermediate_results.append({
                            'piece_name': piece_name,
                            'center': (center_x, center_y),
                            'bbox': (x1, y1, x2, y2)
                        })

                intermediate_results = filter_pieces_on_board(board_contour, intermediate_results, None)

                board_detection_result_img, completed_h_peaks, completed_v_peaks, lines_result_img, warped_1, _, transformed_points, warped_3 \
                    = detect_and_analyze_board(
                    orig_img, board_contour=board_contour, video_writer_lines=True,
                    frame_size=self.frame_size, video_writer_warped_board=True,
                    video_writer_warped_board_second=True, intermediate_results=intermediate_results)

                board_tmp, board = generate_chessboard(completed_h_peaks, completed_v_peaks, transformed_points)

                vector_board_numpy = board_to_vector_numpy(board)

                # 将当前帧的向量化棋盘添加到记忆中
                self.chessboard_memory.add_frame(vector_board_numpy)

                # 获取平滑后的棋盘状态
                smoothed_board_numpy = self.chessboard_memory.get_smoothed_board()

                board_refine = None
                # self.board_change = True

                if smoothed_board_numpy is not None:
                    # 将平滑后的向量化棋盘转换回二维棋盘
                    if self.tmp_board is None:
                        self.tmp_board = smoothed_board_numpy
                        print("棋局状态发生变化！")
                        smoothed_board = vector_to_board_numpy(smoothed_board_numpy)
                        # print(smoothed_board)
                        board_refine = vis_board(smoothed_board)
                        cv2.imwrite(TEMP_BOARD_IMG, board_refine)
                    else:
                        if not np.array_equal(self.tmp_board, smoothed_board_numpy):
                            print("棋局状态发生变化！")
                            self.tmp_board = smoothed_board_numpy
                            smoothed_board = vector_to_board_numpy(smoothed_board_numpy)
                            print(smoothed_board)
                            # 将新的棋局状态添加到 all_game_states 列表中
                            self.all_game_states.append(smoothed_board)
                            print(f"已保存新的棋局状态，当前共保存 {len(self.all_game_states)} 个状态。")

                            current_time = time.time()
                            if (current_time - self.last_change_time >= 2):
                                fen_data = board_to_fen(smoothed_board)
                                print(fen_data)

                                if self.chess_strategy_translate == ['马五进三 (score:29999, winrate:极高（必赢）)']:
                                    if fen_data != "4kab2/1N2a1N2/9/9/9/9/6n1r/4B4/4A4/2BA1K3":
                                        if fen_data == "4kab2/1N2a4/9/6N2/9/9/6n1r/4B4/4A4/2BA1K3":
                                            chess_strategy = "推荐走法是马五进三，你的走法是马五退三，当前劣势很大"
                                        elif fen_data == "4kab2/1N2a4/9/9/5N3/9/6n1r/4B4/4A4/2BA1K3":
                                            chess_strategy = "你的走法是马五退四，当前劣势很大"
                                        else:
                                            chess_strategy = "你的走法无效，请尝试其他走法"
                                    else:
                                        chess_strategy = "你已经获得胜利！"
                                    self.chess_strategy_translate = [chess_strategy]
                                    self.last_change_time = current_time
                                    board_refine = vis_board(smoothed_board, self.chess_strategy_translate)
                                    cv2.imwrite(TEMP_BOARD_IMG, board_refine)
                                    return orig_img_, board_refine, self.chess_strategy_translate, True

                                # --- 开始保存到文件的逻辑 ---
                                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                                strategy_to_save = str(self.chess_strategy_translate)

                                log_entry = f"--- Log Entry: {timestamp} ---\n"
                                log_entry += f"FEN Data: {fen_data}\n"
                                log_entry += f"Last Strategy: {strategy_to_save}\n"
                                log_entry += f"Last FEN Data: {self.fen_board}\n"
                                log_entry += "-----------------------------------\n\n"

                                try:
                                    with open(self.log_filename, 'a', encoding='utf-8') as f:
                                        f.write(log_entry)
                                    print(f"Data successfully appended to {self.log_filename}")
                                except IOError as e:
                                    print(f"Error writing to file {self.log_filename}: {e}")
                                except Exception as e:
                                    print(f"An unexpected error occurred during saving: {e}")
                                # --- 保存到文件的逻辑结束 ---

                                chess_strategy = get_best_move(fen_data + ' w')

                                if chess_strategy == "未知错误":
                                    raise ValueError("未检测到所有棋子")
                                elif chess_strategy == "该局面未被收录，请尝试其他走法":
                                    self.chess_strategy_translate = ["该局面未被收录，请尝试其他走法"]
                                    uci_commands = f"ucci\nposition fen {fen_data} r\ngo depth 5\nquit\nquit"

                                    # 写入临时命令文件
                                    with open(TEMP_COMMAND_FILE, "w") as f:
                                        f.write(uci_commands)
                                    print(
                                        f"FEN saved to {TEMP_COMMAND_FILE} because chess_strategy is '{chess_strategy}'")

                                    # -------- 添加运行 ELEEYE.EXE 并处理输出的代码开始 --------
                                    extracted_bestmove = None
                                    extracted_score = None
                                    move_list = []

                                    try:
                                        # 检查引擎是否存在
                                        if not os.path.exists(ENGINE_PATH):
                                            print(f"Error: Engine not found at {ENGINE_PATH}")
                                        else:
                                            with open(TEMP_COMMAND_FILE, 'r') as input_file:
                                                process = subprocess.Popen(ENGINE_PATH,
                                                                           stdin=input_file,
                                                                           stdout=subprocess.PIPE,
                                                                           stderr=subprocess.PIPE,
                                                                           text=True,
                                                                           bufsize=1)

                                                # 实时读取并过滤输出
                                                while True:
                                                    output_line = process.stdout.readline()
                                                    if output_line == '' and process.poll() is not None:
                                                        break

                                                    output_line = output_line.strip()

                                                    if output_line.startswith('info depth') and 'score' in output_line:
                                                        score_match = re.search(r'score [cp]?(-?\d+)', output_line)
                                                        if score_match:
                                                            extracted_score = score_match.group(1)

                                                    elif output_line.startswith('bestmove'):
                                                        move_parts = output_line.split()
                                                        if len(move_parts) > 1:
                                                            extracted_bestmove = move_parts[1]
                                                            if extracted_bestmove and extracted_score is not None:
                                                                move_str = f'move:{extracted_bestmove},score:{extracted_score},rank:-,winrate:-'
                                                                move_list.append(move_str)
                                                                print(move_str)
                                                                self.chess_strategy_translate = get_chinese_move_notation(
                                                                    move_list, smoothed_board)
                                                                print("chess_strategy_translate:",
                                                                      self.chess_strategy_translate)
                                                                break

                                                process.communicate()

                                    except Exception as e:
                                        print(f"Error running ELEEYE.EXE: {e}")

                                    # -------- 添加运行 ELEEYE.EXE 并处理输出的代码结束 --------

                                else:
                                    for i in range(len(chess_strategy)):
                                        strategy_item = chess_strategy[i]
                                        parts = strategy_item.split(',')
                                        score_part = ""
                                        winrate_part = ""
                                        move_part = parts[0]

                                        for part in parts[1:]:
                                            if "score:" in part:
                                                score_part = part.strip()
                                            elif "winrate:" in part:
                                                winrate_part = part.strip()

                                        score_value = 0
                                        if score_part:
                                            try:
                                                score_value = int(score_part.split(':')[1])
                                            except:
                                                pass

                                        winrate_value = ""
                                        if winrate_part:
                                            winrate_value = winrate_part.split(':')[1]

                                        if score_value > 10000 and not winrate_value:
                                            winrate_part = 'winrate:极高（必赢）'
                                            if winrate_part not in strategy_item:
                                                chess_strategy[i] = f"{move_part},{score_part},{winrate_part}"

                                    self.chess_strategy_translate = get_chinese_move_notation(chess_strategy,
                                                                                              smoothed_board)
                                    print("chess_strategy_translate:", self.chess_strategy_translate)

                                self.last_change_time = current_time
                                board_change = True
                                self.fen_board = fen_data

                            board_refine = vis_board(smoothed_board, self.chess_strategy_translate)
                            cv2.imwrite(TEMP_BOARD_IMG, board_refine)

            return orig_img_, board_refine, self.chess_strategy_translate, board_change

        except Exception as e:
            print(f"Error in add_picture: {e}")
            return None, None, None, False


def crop_image_to_1920x1080(image_np, initial_crop_ratio=0.6):
    """
    使用 OpenCV 灵活裁剪图片：先按比例裁剪四周，再裁剪为 16:9 比例，最后 resize 到 1920*1080。
    """
    img = image_np
    height, width, channels = img.shape
    print(f"原始图片尺寸: {width}x{height}")

    # 0. 初始裁剪 (按比例缩小四周)
    if 0 < initial_crop_ratio < 1:
        initial_width = int(width * initial_crop_ratio)
        initial_height = int(height * initial_crop_ratio)
        if initial_width > 0 and initial_height > 0 and (initial_width < width or initial_height < height):
            left_initial = (width - initial_width) // 2
            top_initial = (height - initial_height) // 2
            right_initial = left_initial + initial_width
            bottom_initial = top_initial + initial_height
            img = img[top_initial:bottom_initial, left_initial:right_initial]
            height, width, channels = img.shape
            print(f"已进行初始比例裁剪 ({initial_crop_ratio})，裁剪后尺寸: {width}x{height}")
        else:
            print("初始裁剪比例无效或无需裁剪，跳过初始裁剪。")

    # 1. 裁剪到 16:9 比例
    aspect_ratio = width / height
    target_aspect_ratio = 16 / 9

    if abs(aspect_ratio - target_aspect_ratio) > 0.01:
        if aspect_ratio > target_aspect_ratio:
            # 图片更宽，需要裁剪左右
            new_width = int(height * target_aspect_ratio)
            crop_width = width - new_width
            left = crop_width // 2
            top = 0
            right = left + new_width
            bottom = height
            print(f"图片比例为 {aspect_ratio:.2f}:1，宽于 16:9，裁剪左右两侧。")
        else:
            # 图片更高，需要裁剪上下
            new_height = int(width / target_aspect_ratio)
            crop_height = height - new_height
            top = crop_height // 2
            left = 0
            right = width
            bottom = top + new_height
            print(f"图片比例为 {aspect_ratio:.2f}:1，高于 16:9，裁剪上下两侧。")
        cropped_16_9_img = img[top:bottom, left:right]
        print(f"已裁剪为 16:9 比例，裁剪后尺寸: {cropped_16_9_img.shape[1]}x{cropped_16_9_img.shape[0]}")
    else:
        cropped_16_9_img = img
        print(f"图片已经是 16:9 比例或非常接近，无需裁剪比例。")

    # 2. Resize 到 1920*1080
    resized_img = cv2.resize(cropped_16_9_img, (1920, 1080), interpolation=cv2.INTER_LANCZOS4)
    return resized_img


if __name__ == '__main__':
    """
    示例用法：
    请确保 'pre-trained-models' 文件夹在当前目录下。
    请确保 'ELEEYE.EXE' 在当前目录下（如果需要使用引擎分析）。
    """

    # 示例配置（使用相对路径）
    # image_root = "./datasets/ChineseChess/source_v5/test/VID_TEST/"
    # my_Chinese_Chessboard = Chinese_Chessboard_New(memory_length=4, stability_threshold_ratio=0.75)
    # process_interval = 6

    # 测试单个图片
    # frame_path = "./received_frames/test_frame.jpg"
    # if os.path.exists(frame_path):
    #     frame = cv2.imread(frame_path)
    #     orig_img_ = my_Chinese_Chessboard.add_picture(frame)
    # else:
    #     print(f"测试图片不存在: {frame_path}")

    # 简单的逻辑测试
    test_move = ['move:f0f1,score:-248,rank:-,winrate:-']
    test_board = [
        ['.', '.', '.', '.', '将', '士', '象', '.', '.'],
        ['.', '马', '.', '.', '士', '.', '.', '.', '.'],
        ['.', '.', '.', '.', '象', '.', '.', '.', '.'],
        ['.', '.', '.', '.', '.', '.', '.', '.', '.'],
        ['.', '.', '.', '.', '.', '.', '.', '.', '.'],
        ['.', '.', '.', '.', '.', '.', '.', '.', '.'],
        ['.', '.', '.', '.', '.', '.', '.', '.', '俥'],
        ['.', '.', '.', '.', '相', '.', '傌', '.', '.'],
        ['.', '.', '.', '.', '仕', '.', '.', '.', '.'],
        ['.', '.', '相', '仕', '.', '帅', '.', '.', '.']
    ]

    chess_strategy_translate = get_chinese_move_notation(test_move, test_board)
    print("测试走法转换结果:", chess_strategy_translate)
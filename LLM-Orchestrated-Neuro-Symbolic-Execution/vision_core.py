import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import os
import math
import ssl
from collections import defaultdict, OrderedDict, deque
from PIL import Image, ImageDraw, ImageFont
import torch
import pdb
import io
from ultralytics import YOLO, FastSAM
import matplotlib.patches as patches
from matplotlib import font_manager
import matplotlib

# 设置 SSL 上下文，防止下载模型时报错
ssl._create_default_https_context = ssl._create_unverified_context
# 切换到 Agg 后端以支持无头模式运行
matplotlib.use('Agg')

# ==========================================
# 全局参数配置 (请勿随意修改数字)
# ==========================================
# 线条的十字架长度和粗细 (1920 * 1080)
cross_length = 35  # 十字架线条的长度
line_thickness = 3  # 线条粗细
square_size = 40
vertical_missing_parameter = 30
distance_threshold = 20
max_cnt_threshold = 0.85
kernel = 25  # 45

# 类别映射关系
cls_mapping = {
    1: "red_bing",
    2: "red_pao",
    3: "red_ju",
    4: "red_ma",
    5: "red_xiang",
    6: "red_shi",
    7: "red_shuai",
    8: "black_ju",
    9: "black_ma",
    10: "black_xiang",
    11: "black_shi",
    12: "black_jiang",
    13: "black_zu",
    14: "black_pao"
}

# 全局变量用于存储线条检测结果
horizontal_lines = None
vertical_lines = None


# ==========================================
# 类定义
# ==========================================

class ChessboardMemory:
    def __init__(self, memory_length, stability_threshold_ratio=0.75):
        """
        初始化棋盘记忆。
        Args:
            memory_length (int): 记忆帧的长度（超参数）。
        """
        self.memory_length = memory_length
        self.memory = deque(maxlen=memory_length)  # 使用 deque 实现固定长度队列
        self.stability_threshold_ratio = stability_threshold_ratio
        self.previous_smoothed_board = None  # 用于存储上一个平滑后的棋盘状态

    def add_frame(self, vector_board):
        """
        添加新的棋盘状态帧到记忆中。
        Args:
            vector_board (numpy.ndarray): (10, 9, 15) 的 NumPy 数组。
        """
        self.memory.append(vector_board)

    def get_smoothed_board(self):
        """
        获取平滑后的棋盘状态。
        """
        if not self.memory:
            return None

        summed_board = np.zeros_like(self.memory[0], dtype=float)
        for frame in self.memory:
            summed_board += frame

        stability_threshold = self.stability_threshold_ratio * self.memory_length

        # 初始化 smoothed_board
        if self.previous_smoothed_board is not None:
            smoothed_board = np.copy(self.previous_smoothed_board)
        else:
            smoothed_board = np.zeros_like(self.memory[0], dtype=int)
            empty_piece_index = 14
            smoothed_board[:, :, empty_piece_index] = 1

        for r_idx in range(summed_board.shape[0]):
            for c_idx in range(summed_board.shape[1]):
                position_vector = summed_board[r_idx, c_idx, :]
                max_count = np.max(position_vector)
                max_dimension_index = np.argmax(position_vector)

                if max_count >= stability_threshold:
                    smoothed_board[r_idx, c_idx, :] = 0
                    smoothed_board[r_idx, c_idx, max_dimension_index] = 1

        self.previous_smoothed_board = smoothed_board
        return smoothed_board


# ==========================================
# 工具函数
# ==========================================

def find_adaptive_h_peaks_percentile(horizontal_sum, max_peak_percentage=0.20):
    """使用百分位数自适应查找横向峰值"""
    target_peak_count = int(len(horizontal_sum) * max_peak_percentage)
    if target_peak_count == 0:
        target_peak_count = 1

    for percentile in range(99, 50, -5):
        threshold = np.percentile(horizontal_sum, percentile)
        current_h_peaks = np.where(horizontal_sum >= threshold)[0]
        if len(current_h_peaks) <= target_peak_count:
            return current_h_peaks
    return np.array([])


def filter_pieces_on_board(board_contour, intermediate_results, image=None, output_path="visualized_pieces.jpg"):
    """判断棋子是否在棋盘上"""
    filtered_results = []

    for piece_info in intermediate_results:
        bbox = piece_info['bbox']
        xmin, ymin, xmax, ymax = bbox

        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)

        bbox_points = np.array([
            [xmin, ymin],
            [xmax, ymin],
            [xmax, ymax],
            [xmin, ymax]
        ], dtype=np.float32)

        is_inside = False
        for point in bbox_points:
            if cv2.pointPolygonTest(board_contour, (point[0], point[1]), False) >= 0:
                is_inside = True
                break

        if is_inside:
            filtered_results.append(piece_info)

    return filtered_results


def get_chinese_move_notation(chess_strategy, board, current_color="red"):
    """将走棋策略转换为中文描述性走法"""
    col_cn = ["一", "二", "三", "四", "五", "六", "七", "八", "九"]
    step_cn = {1: "一", 2: "二", 3: "三", 4: "四", 5: "五", 6: "六", 7: "七", 8: "八", 9: "九"}

    def get_file_notation(col, color):
        if color == "red":
            file_num = 9 - col
            return col_cn[file_num - 1]
        else:
            return str(col + 1)

    def get_step_notation(step, color):
        return step_cn.get(step, str(step)) if color == "red" else str(step)

    def is_forward(color, start_rank_digit, end_rank_digit):
        if color == "red":
            return int(end_rank_digit) > int(start_rank_digit)
        else:
            return int(end_rank_digit) < int(start_rank_digit)

    if current_color == "red":
        piece_mapping = {'兵': '兵', '车': '车', '马': '马', '相': '象', '仕': '仕', '帅': '帅', '炮': '炮'}
        vertical_movers = {"兵", "车", "炮", "帅"}
        diagonal_movers = {"马", "象", "仕"}
    else:
        piece_mapping = {'卒': '卒', '俥': '俥', '傌': '傌', '象': '象', '士': '士', '将': '将', '砲': '砲'}
        vertical_movers = {"卒", "俥", "砲", "将"}
        diagonal_movers = {"傌", "象", "士"}

    descriptive_moves = []
    for move_str in chess_strategy:
        parts = move_str.split(',')
        coord_part = parts[0]
        move_coords = coord_part.split(':')[1]
        start_coord = move_coords[:2]
        end_coord = move_coords[2:]

        start_file_letter, start_rank_digit = start_coord[0], start_coord[1]
        end_file_letter, end_rank_digit = end_coord[0], end_coord[1]

        start_col = ord(start_file_letter) - ord('a')
        end_col = ord(end_file_letter) - ord('a')

        start_row_index = 9 - int(start_rank_digit)

        piece = board[start_row_index][start_col]
        piece_name = piece_mapping.get(piece, piece)
        start_file = get_file_notation(start_col, current_color)

        if start_rank_digit == end_rank_digit:
            move_indicator = "平"
            target = get_file_notation(end_col, current_color)
        else:
            forward = is_forward(current_color, start_rank_digit, end_rank_digit)
            move_indicator = "进" if forward else "退"
            if piece_name in vertical_movers or piece_name not in diagonal_movers:
                step = abs(int(end_rank_digit) - int(start_rank_digit))
                target = get_step_notation(step, current_color)
            else:
                target = get_file_notation(end_col, current_color)

        same_file_rows = []
        for r in range(10):
            if board[r][start_col] == piece:
                same_file_rows.append(r)
        prefix = ""
        if len(same_file_rows) > 1:
            if current_color == "red":
                sorted_rows = sorted(same_file_rows)
            else:
                sorted_rows = sorted(same_file_rows, reverse=True)
            idx = sorted_rows.index(start_row_index)
            if len(sorted_rows) == 2:
                prefix = ["前", "后"][idx]
            elif len(sorted_rows) >= 3:
                labels = ["前", "中", "后"]
                prefix = labels[idx] if idx < len(labels) else ""

        extra = ""
        score = winrate = None
        for part in parts:
            if part.startswith("score:"):
                score = part.split(":")[1]
            elif part.startswith("winrate:"):
                winrate = part.split(":")[1]
        if score is not None and winrate is not None:
            extra = f" (score:{score}, winrate:{winrate})"

        move_notation = f"{prefix}{piece_name}{start_file}{move_indicator}{target}{extra}"
        descriptive_moves.append(move_notation)

    return descriptive_moves


def board_to_fen(board):
    """将棋盘 board 数据转换为 FEN 格式"""
    piece_to_fen_map = {
        '俥': 'r', '傌': 'n', '象': 'b', '士': 'a', '将': 'k', '砲': 'c', '卒': 'p',
        '车': 'R', '马': 'N', '相': 'B', '仕': 'A', '帅': 'K', '炮': 'C', '兵': 'P'
    }
    fen_string = ""
    for row_index, row in enumerate(board):
        empty_count = 0
        for piece in row:
            if piece == '.':
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_string += str(empty_count)
                    empty_count = 0
                fen_string += piece_to_fen_map.get(piece, piece)
        if empty_count > 0:
            fen_string += str(empty_count)
        if row_index < len(board) - 1:
            fen_string += "/"
    return fen_string


def fen_to_board(fen_string):
    """将 FEN 格式字符串转换为 board 数据"""
    fen_to_piece_map = {
        'r': '俥', 'n': '傌', 'b': '象', 'a': '士', 'k': '将', 'c': '砲', 'p': '卒',
        'R': '车', 'N': '马', 'B': '相', 'A': '仕', 'K': '帅', 'C': '炮', 'P': '兵'
    }
    board = []
    rows = fen_string.split('/')
    for row_fen in rows:
        row_board = []
        for char in row_fen:
            if char.isdigit():
                row_board.extend(['.'] * int(char))
            else:
                row_board.append(fen_to_piece_map.get(char, char))
        board.append(row_board)
    return board


def board_to_vector_numpy(board):
    """将棋盘 (二维列表) 转换为 NumPy 向量表示"""
    piece_to_index = {
        '车': 0, '马': 1, '相': 2, '仕': 3, '帅': 4, '炮': 5, '兵': 6,
        '俥': 7, '傌': 8, '象': 9, '士': 10, '将': 11, '砲': 12, '卒': 13,
        '.': 14
    }
    vector_board = np.zeros((10, 9, 15), dtype=int)

    for r_idx, row in enumerate(board):
        for c_idx, piece in enumerate(row):
            index = piece_to_index.get(piece)
            if index is not None:
                vector_board[r_idx, c_idx, index] = 1

    return vector_board


def vector_to_board_numpy(vector_board):
    """将 NumPy 向量表示的棋盘转换回二维列表的棋盘"""
    index_to_piece = {
        0: '车', 1: '马', 2: '相', 3: '仕', 4: '帅', 5: '炮', 6: '兵',
        7: '俥', 8: '傌', 9: '象', 10: '士', 11: '将', 12: '砲', 13: '卒',
        14: '.'
    }
    board = [['' for _ in range(9)] for _ in range(10)]

    for r_idx in range(vector_board.shape[0]):
        for c_idx in range(vector_board.shape[1]):
            vector = vector_board[r_idx, c_idx, :]
            index = np.argmax(vector)
            piece = index_to_piece[index]
            board[r_idx][c_idx] = piece

    return board


def generate_chessboard(completed_h_peaks, completed_v_peaks, transformed_points):
    """生成带有棋子位置的中国象棋棋盘"""
    board = [['.' for _ in range(9)] for _ in range(10)]
    h_peaks = np.array(completed_h_peaks)
    v_peaks = np.array(completed_v_peaks)

    for item in transformed_points:
        cx, cy = item['transformed_center']
        row_idx = np.argmin(np.abs(h_peaks - cy))
        row = 9 - row_idx
        col = np.argmin(np.abs(v_peaks - cx))

        if 0 <= row < 10 and 0 <= col < 9:
            piece_name = item['piece_name'].replace("balck", "black")
            symbol_map = {
                'black_ju': '俥', 'black_ma': '傌', 'black_xiang': '象',
                'black_shi': '士', 'black_jiang': '将', 'black_pao': '砲',
                'black_zu': '卒',
                'red_ju': '车', 'red_ma': '马', 'red_xiang': '相',
                'red_shi': '仕', 'red_shuai': '帅', 'red_pao': '炮',
                'red_bing': '兵'
            }
            board[row][col] = symbol_map.get(piece_name, piece_name)

    board.reverse()

    output = []
    for i in range(10):
        output.append(f"{9 - i:2d} {' '.join(board[i])}")

    columns = '    ' + ' '.join(['ａ', 'ｂ', 'ｃ', 'ｄ', 'ｅ', 'ｆ', 'ｇ', 'ｈ', 'ｉ'])
    output.append(columns)

    return '\n'.join(output), board


def vis_board(board, best_moves=None):
    """可视化棋盘"""
    # 注意：需确保 FONT_PATH 指向正确的字体文件，否则可能乱码
    global FONT_PATH
    try:
        font_path = FONT_PATH
    except NameError:
        font_path = "SIMHEI.TTF"  # Default fallback

    prop = font_manager.FontProperties(fname=font_path)

    background_colors = {
        '俥': 'black', '傌': 'black', '象': 'black', '士': 'black', '将': 'black', '砲': 'black', '卒': 'black',
        '车': 'red', '马': 'red', '相': 'red', '仕': 'red', '帅': 'red', '炮': 'red', '兵': 'red'
    }

    piece_colors = {
        '俥': 'white', '傌': 'white', '象': 'white', '士': 'white', '将': 'white', '砲': 'white', '卒': 'white',
        '车': 'white', '马': 'white', '相': 'white', '仕': 'white', '帅': 'white', '炮': 'white', '兵': 'white'
    }

    piece_map = {
        '俥': '俥', '傌': '傌', '象': '象', '士': '士', '将': '将', '砲': '砲', '卒': '卒',
        '车': '车', '马': '马', '相': '相', '仕': '仕', '帅': '帅', '炮': '炮', '兵': '兵',
        '.': ''
    }

    fig, ax = plt.subplots(figsize=(8, 9))
    ax.set_facecolor('#F0C78A')
    ax.set_xlim(-1, 9)
    ax.set_ylim(-1, 10)

    for i in range(10):
        if i == 4:
            continue
        ax.plot([0, 8], [i, i], color='black', linewidth=1)
    ax.plot([0, 8], [4, 4], color='black', linewidth=1)
    ax.plot([0, 8], [5, 5], color='black', linewidth=1)

    for j in range(9):
        if j == 0 or j == 8:
            ax.plot([j, j], [0, 9], color='black', linewidth=1)
        else:
            ax.plot([j, j], [0, 4], color='black', linewidth=1)
            ax.plot([j, j], [5, 9], color='black', linewidth=1)

    ax.plot([3, 5], [0, 2], color='black', linewidth=1)
    ax.plot([3, 5], [2, 0], color='black', linewidth=1)
    ax.plot([3, 5], [7, 9], color='black', linewidth=1)
    ax.plot([3, 5], [9, 7], color='black', linewidth=1)

    row_numbers = ['九', '八', '七', '六', '五', '四', '三', '二', '一']
    for j, number in enumerate(row_numbers):
        ax.text(j, 9.75, number, ha='center', va='center', fontsize=16, color='black', fontproperties=prop)

    ax.text(2, 4.5, "楚", ha='center', va='center', fontsize=24, color='black', fontproperties=prop)
    ax.text(3, 4.5, "河", ha='center', va='center', fontsize=24, color='black', fontproperties=prop)
    ax.text(5, 4.5, "汉", ha='center', va='center', fontsize=24, color='black', fontproperties=prop)
    ax.text(6, 4.5, "界", ha='center', va='center', fontsize=24, color='black', fontproperties=prop)

    for row in range(10):
        for col in range(9):
            piece = board[row][col]
            if piece != '.':
                circle = patches.Circle((col, row), radius=0.45, color=background_colors[piece], ec='black',
                                        linewidth=1, zorder=2)
                ax.add_patch(circle)
                ax.text(col, row, piece_map[piece], ha='center', va='center', fontsize=24, color=piece_colors[piece],
                        zorder=3, fontproperties=prop)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.invert_yaxis()

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    image_np = cv2.imdecode(np.frombuffer(buf.read(), np.uint8), cv2.IMREAD_COLOR)
    buf.close()
    plt.close(fig)

    image_bgr = image_np
    board_image_height, board_image_width, _ = image_bgr.shape

    if best_moves:
        text_box_width = 500
        text_box = np.zeros((board_image_height, text_box_width, 3), dtype=np.uint8)
        text_box[:] = (255, 255, 255)

        text_box_pil = Image.fromarray(cv2.cvtColor(text_box, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(text_box_pil)
        font_size = 20
        try:
            chinese_font = ImageFont.truetype(font_path, font_size)
        except IOError:
            print(f"字体文件 {font_path} 未找到，使用默认字体。")
            chinese_font = ImageFont.load_default()

        text_color = (0, 0, 0)
        line_height = font_size + 2
        text_x = 10
        text_y = 10 + font_size

        draw.text((text_x, text_y), "最佳走法建议:", font=chinese_font, fill=text_color)
        text_y += line_height + 5

        for move_text in best_moves:
            draw.text((text_x, text_y), "- " + move_text, font=chinese_font, fill=text_color)
            text_y += line_height

        text_box_np = cv2.cvtColor(np.array(text_box_pil), cv2.COLOR_RGB2BGR)
        image_bgr = np.concatenate((image_bgr, text_box_np), axis=1)

    return image_bgr


def filter_fold_lines(horizontal_lines, peaks, image_height, min_ratio=0.6, center_range=0.2):
    peaks = np.array(peaks)
    if len(peaks) <= 1:
        return peaks

    center_min = image_height * (0.5 - center_range / 2) + 20
    center_max = image_height * (0.5 + center_range / 2) + 20

    line_distances = np.diff(peaks)
    median_distance = np.median(line_distances)
    lines_to_remove = []

    for i in range(len(peaks)):
        peak = peaks[i]
        if center_min <= peak <= center_max:
            line_length = np.sum(horizontal_lines[peak, :])
            max_possible_length = horizontal_lines.shape[1]
            length_ratio = line_length / max_possible_length

            if i > 0 and i < len(peaks) - 1:
                prev_distance = peaks[i] - peaks[i - 1]
                next_distance = peaks[i + 1] - peaks[i]

                if (length_ratio > min_ratio and
                        prev_distance < median_distance * 0.7 and
                        next_distance < median_distance * 0.7):
                    lines_to_remove.extend([i])

    kept_indices = list(set(range(len(peaks))) - set(lines_to_remove))
    kept_indices = sorted(kept_indices)
    filtered_peaks = peaks[kept_indices]

    return filtered_peaks.tolist()


def merge_close_lines(lines, axis='horizontal', distance_threshold=10):
    if len(lines) == 0:
        return []
    lines = np.array(lines).reshape(-1, 1)
    clustering = DBSCAN(eps=distance_threshold, min_samples=1).fit(lines)
    labels = clustering.labels_
    unique_labels = np.unique(labels)
    merged_lines = []
    for label in unique_labels:
        cluster = lines[labels == label]
        merged_line = int(np.mean(cluster))
        merged_lines.append(merged_line)
    merged_lines = sorted(merged_lines)
    return merged_lines


def interpolate_missing_lines(lines, expected_count, axis_length):
    if expected_count == 10:
        lines = sorted(filter(lambda x: 22 <= x <= axis_length - 22, lines))
        if len(lines) == 11 and lines[0] <= 38: return lines[1:]
    elif expected_count == 9:
        lines = sorted(
            filter(lambda x: vertical_missing_parameter <= x <= axis_length - vertical_missing_parameter, lines))

    if not lines:
        return _generate_theoretical_lines(expected_count, axis_length)

    refined = _dynamic_interpolation_optimized(lines, expected_count, axis_length)
    return sorted(list(set(refined)))[:expected_count]


def _dynamic_interpolation_optimized(initial_lines, expected_count, axis_length):
    lines = initial_lines.copy()
    lines = [int(round(x)) for x in lines]
    avg_spacing = _calculate_integer_spacing(lines)
    lines = _fill_large_gaps(lines, avg_spacing, expected_count)
    lines = _expand_with_check(lines, avg_spacing, expected_count, axis_length)

    if len(lines) < expected_count:
        lines = _force_uniform_distribution(lines, expected_count, axis_length)

    return lines


def _calculate_integer_spacing(lines):
    if len(lines) < 2:
        return max(1, (lines[0] if lines else 50))
    spacings = [lines[i + 1] - lines[i] for i in range(len(lines) - 1)]
    return int(np.median(spacings))


def _expand_with_check(lines, spacing, target_count, axis_length, edge_margin=25):
    while len(lines) < target_count:
        new_pos = lines[-1] + spacing
        if new_pos > axis_length - edge_margin:
            break
        if not _is_position_exists(new_pos, lines):
            lines.append(int(new_pos))
        else:
            break

    while len(lines) < target_count:
        new_pos = lines[0] - spacing
        if new_pos < edge_margin:
            break
        if not _is_position_exists(new_pos, lines):
            lines.insert(0, int(new_pos))
        else:
            break
    return lines


def _fill_large_gaps(lines, spacing, target_count):
    while len(lines) < target_count:
        max_gap = 0
        insert_idx = -1
        for i in range(len(lines) - 1):
            current_gap = lines[i + 1] - lines[i]
            if current_gap > max_gap and current_gap > 1.3 * spacing:
                max_gap = current_gap
                insert_idx = i

        if insert_idx == -1:
            break
        new_pos = lines[insert_idx] + spacing
        if not _is_position_exists(new_pos, lines):
            lines.insert(insert_idx + 1, int(new_pos))
        else:
            break
    return lines


def _is_position_exists(pos, existing_lines, tolerance=30):
    return any(abs(x - pos) <= tolerance for x in existing_lines)


def _force_uniform_distribution(lines, target_count, axis_length):
    start = max(lines[0], 25)
    end = min(lines[-1], axis_length - 25)
    return [int(start + i * (end - start) / (target_count - 1)) for i in range(target_count)]


def _generate_theoretical_lines(count, axis_length):
    start = 25
    end = axis_length - 25
    return [int(start + i * (end - start) / (count - 1)) for i in range(count)]


def find_valid_lines(lines, axis='horizontal', min_length_ratio=0.5, target_size=100):
    valid_lines = []
    # 注意：horizontal_lines 和 vertical_lines 需要在调用前通过 global 或参数传递
    # 原始代码直接使用了全局变量
    global horizontal_lines, vertical_lines

    for line in lines:
        if axis == 'horizontal':
            row = horizontal_lines[line, :]
            contours, _ = cv2.findContours(row.reshape(1, -1, 1), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            max_width = 0
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w > max_width:
                    max_width = w
            if max_width >= target_size * min_length_ratio:
                valid_lines.append(line)
        else:
            column = vertical_lines[:, line]
            contours, _ = cv2.findContours(column.reshape(-1, 1, 1), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            max_height = 0
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if h > max_height:
                    max_height = h
            if max_height >= target_size * min_length_ratio:
                valid_lines.append(line)
    return valid_lines


def line_to_eq(line):
    x1, y1, x2, y2 = line
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    return np.array([A, B, C])


def intersection(line1, line2):
    A1, B1, C1 = line1
    A2, B2, C2 = line2
    determinant = A1 * B2 - A2 * B1
    if determinant == 0:
        return None
    else:
        x = (B1 * C2 - B2 * C1) / determinant
        y = (C1 * A2 - C2 * A1) / determinant
        return (int(x), int(y))


def extend_line(line, image_width, image_height):
    x1, y1, x2, y2 = line
    dx = x2 - x1
    dy = y2 - y1

    if dx == 0:
        y1_new = 0
        y2_new = image_height
        return [x1, y1_new, x2, y2_new]
    if dy == 0:
        x1_new = 0
        x2_new = image_width
        return [x1_new, y1, x2_new, y2]

    slope = dy / dx
    intercept = y1 - slope * x1

    if abs(slope) <= 1:
        x1_new = 0
        y1_new = int(slope * x1_new + intercept)
        x2_new = image_width
        y2_new = int(slope * x2_new + intercept)
    else:
        y1_new = 0
        x1_new = int((y1_new - intercept) / slope)
        y2_new = image_height
        x2_new = int((y2_new - intercept) / slope)

    return [x1_new, y1_new, x2_new, y2_new]


def visualize_keypoint_matches(template, img, kp1, kp2, good_matches, max_width=1200, max_height=800):
    h1, w1 = template.shape[:2]
    h2, w2 = img.shape[:2]
    height = max(h1, h2)
    width = w1 + w2

    out_img = np.zeros((height, width, 3), dtype=np.uint8)
    out_img[:h1, :w1, :] = template
    out_img[:h2, w1:w1 + w2, :] = img

    for m in good_matches:
        pt1 = (int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1]))
        pt2 = (int(kp2[m.trainIdx].pt[0] + w1), int(kp2[m.trainIdx].pt[1]))
        cv2.circle(out_img, pt1, 5, (0, 255, 0), -1)
        cv2.circle(out_img, pt2, 5, (0, 255, 0), -1)
        cv2.line(out_img, pt1, pt2, (255, 0, 0), 2)

    if width > max_width or height > max_height:
        scale_w = max_width / width
        scale_h = max_height / height
        scale = min(scale_w, scale_h)
        new_width = int(width * scale)
        new_height = int(height * scale)
        out_img = cv2.resize(out_img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return out_img


def visualize_homography(img, template, H):
    print("Homography Matrix H:\n", H)
    h_t, w_t = template.shape[:2]
    corners_t = np.float32([
        [0, 0],
        [w_t - 1, 0],
        [w_t - 1, h_t - 1],
        [0, h_t - 1]
    ]).reshape(-1, 1, 2)

    corners_in_img = cv2.perspectiveTransform(corners_t, H)
    corners_in_img = np.int32(corners_in_img)
    vis_img = img.copy()
    cv2.polylines(vis_img, [corners_in_img], True, (0, 255, 0), 3)

    h_i, w_i = img.shape[:2]
    warped_template = cv2.warpPerspective(template, H, (w_i, h_i))

    return vis_img, warped_template


def load_and_preprocess_template(template_path, max_size=1000, rotate_and_flip=True):
    template_img = cv2.imread(template_path)
    if template_img is None:
        raise ValueError(f"无法读取模板图像 {template_path}")

    h, w = template_img.shape[:2]
    scale = max_size / float(max(h, w))
    if scale < 1.0:
        template_img = cv2.resize(template_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    templates = [template_img]
    if rotate_and_flip:
        rot_180 = cv2.rotate(template_img, cv2.ROTATE_180)
        templates.append(rot_180)

    return templates


def detect_and_analyze_board(
        image_path,
        template_path='./myChineseChess_wo_background.png',
        debug_resize_max_width=800,
        min_area_ratio=0.05,
        max_area_ratio=0.95,
        min_aspect_ratio=0.8,
        max_aspect_ratio=1.25,
        board_contour=None,
        frame_size=None,
        video_writer_lines=None,
        video_writer_warped_board=None,
        video_writer_warped_board_second=None,
        intermediate_results=None,
):
    global horizontal_lines, vertical_lines

    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像 {image_path}")
    except:
        img = image_path

    if True:
        if board_contour is None:
            h_img, w_img = img.shape[:2]
            image_area = w_img * h_img
            max_input_size = 1920
            scale_input = max_input_size / float(max(h_img, w_img))
            input_resize_img = img
            if scale_input < 1.0:
                input_resize_img = cv2.resize(img, None, fx=scale_input, fy=scale_input, interpolation=cv2.INTER_AREA)

            gray_img = cv2.cvtColor(input_resize_img, cv2.COLOR_BGR2GRAY)
            candidate_templates = load_and_preprocess_template(template_path, max_size=1920, rotate_and_flip=True)

            akaze = cv2.AKAZE_create()
            detector = akaze

            kp2, des2 = detector.detectAndCompute(gray_img, None)
            if des2 is None or len(des2) == 0:
                raise ValueError("无法在输入图中检测到足够特征点。")

            best_match_result = None
            best_num_inliers = 0

            for idx, tpl in enumerate(candidate_templates):
                gray_tpl = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
                kp1, des1 = detector.detectAndCompute(gray_tpl, None)
                if des1 is None or len(des1) == 0:
                    continue

                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
                matches = bf.knnMatch(des1, des2, k=2)

                ratio_threshold = 0.75
                good_matches = []
                for m, n in matches:
                    if m.distance < ratio_threshold * n.distance:
                        good_matches.append(m)

                MIN_MATCH_COUNT = 15
                if len(good_matches) >= MIN_MATCH_COUNT:
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 20.0)

                    if H is None:
                        continue

                    inliers = mask.ravel().sum()

                    if inliers > best_num_inliers:
                        h_t, w_t = gray_tpl.shape[:2]
                        corners_t = np.float32([
                            [0, 0],
                            [w_t - 1, 0],
                            [w_t - 1, h_t - 1],
                            [0, h_t - 1]
                        ]).reshape(-1, 1, 2)
                        corners_in_img = cv2.perspectiveTransform(corners_t, H)
                        candidate_contour = np.int32(corners_in_img)

                        if (candidate_contour < 0).any():
                            continue

                        best_num_inliers = inliers
                        best_match_result = {
                            'contour': candidate_contour,
                            'H': H,
                            'inliers': inliers,
                            'template_idx': idx
                        }
                else:
                    raise ValueError("未能在轮廓检测中找到符合后处理规则的棋盘四边形。222")

            if not best_match_result:
                raise ValueError("未能匹配到棋盘，请检查输入图像或模板质量。")

            board_contour = best_match_result['contour']
            if scale_input < 1.0:
                inv_scale = 1.0 / scale_input
                board_contour = (board_contour * inv_scale).astype(np.int32)

            if not _is_valid_contour(board_contour, image_area,
                                     min_area_ratio, max_area_ratio,
                                     min_aspect_ratio, max_aspect_ratio):
                raise ValueError("找到匹配，但不符合大小或长宽比的限制规则。")

            vis_img = img.copy()
            cv2.polylines(vis_img, [board_contour], True, (0, 255, 0), 3)

            x, y, w, h = cv2.boundingRect(board_contour)
            cropped_board = img[y:y + h, x:x + w]
            cv2.imwrite("cropped_board.jpg", cropped_board)

            board_corners = board_contour.reshape(4, 2)
            padding = 0.05
            MINLineLength = 50
            MAXLineGap = 800

        else:
            print("Get board contour!")
            board_corners = board_contour
            padding = 0.0
            MINLineLength = 100
            MAXLineGap = 800

        rect = np.zeros((4, 2), dtype="float32")

        s = board_corners.sum(axis=1)
        rect[0] = board_corners[np.argmin(s)]
        rect[2] = board_corners[np.argmax(s)]
        diff = np.diff(board_corners, axis=1)
        rect[1] = board_corners[np.argmin(diff)]
        rect[3] = board_corners[np.argmax(diff)]

        widthA = np.linalg.norm(rect[2] - rect[1])
        widthB = np.linalg.norm(rect[3] - rect[0])
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.linalg.norm(rect[1] - rect[0])
        heightB = np.linalg.norm(rect[2] - rect[3])
        maxHeight = max(int(heightA), int(heightB))

        width = maxWidth
        height = maxHeight

        target_width = int(width * (1 + 0))
        target_height = int(height * (1 + 0))

        dst = np.array([
            [0, 0],
            [width - 0, 0],
            [width - 0, height - 0],
            [0, height - 0]
        ], dtype="float32")

        M1 = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, M1, (width, height))

        if video_writer_warped_board:
            warped_1 = cv2.resize(warped, frame_size)
        else:
            cv2.imwrite('warped_board.jpg', warped)

        if video_writer_warped_board_second:
            M_total = M1
            for item in intermediate_results:
                cx, cy = item['center']
                center_pts = np.array([[[cx, cy]]], dtype=np.float32)
                transformed_center = cv2.perspectiveTransform(center_pts, M_total)[0][0]
                item['transformed_center'] = tuple(transformed_center)

            warped_result = warped.copy()
            for item in intermediate_results:
                piece_name = item['piece_name']
                cx, cy = item['transformed_center']
                cv2.circle(warped_result, (int(cx), int(cy)), 5, (0, 255, 0), -1)
                cv2.putText(warped_result, piece_name, (int(cx), int(cy) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

            warped_3 = cv2.resize(warped_result, frame_size)
        else:
            cv2.imwrite('warped_board_.jpg', warped)

        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        warped_binary = cv2.adaptiveThreshold(
            warped_gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2
        )

        edge_threshold = 22

        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel))

        horizontal_lines = cv2.morphologyEx(warped_binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        vertical_lines = cv2.morphologyEx(warped_binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

        for item in intermediate_results:
            cx, cy = item['transformed_center']
            cx, cy = int(cx), int(cy)

            if 0 <= cy < horizontal_lines.shape[0] and 0 <= cx < vertical_lines.shape[1]:
                start_row = max(0, cy - square_size)
                end_row = min(horizontal_lines.shape[0], cy + square_size + 1)
                start_col = max(0, cx - square_size)
                end_col = min(vertical_lines.shape[1], cx + square_size + 1)

                horizontal_lines[start_row:end_row, start_col:end_col] = 0
                vertical_lines[start_row:end_row, start_col:end_col] = 0

                vertical_lines[cy - cross_length:cy + cross_length + 1,
                cx - line_thickness:cx + line_thickness + 1] = 255

                horizontal_lines[cy - line_thickness:cy + line_thickness + 1,
                cx - cross_length:cx + cross_length + 1] = 255

        horizontal_sum = np.sum(horizontal_lines, axis=1)
        vertical_sum = np.sum(vertical_lines, axis=0)

        h_peaks = np.where((horizontal_sum > np.max(horizontal_sum) * 0.15) & (
                    horizontal_sum < np.max(horizontal_sum) * max_cnt_threshold))[0]
        v_peaks = np.where(vertical_sum > np.max(vertical_sum) * 0.15)[0]

        h_peaks = h_peaks[(h_peaks > 30) & (h_peaks < target_height - 30)]
        v_peaks = v_peaks[(v_peaks > edge_threshold) & (v_peaks < target_width - edge_threshold)]

        merged_h_peaks = merge_close_lines(h_peaks, axis='horizontal', distance_threshold=distance_threshold)
        merged_v_peaks = merge_close_lines(v_peaks, axis='vertical', distance_threshold=20)

        lines_result_img = None
        if video_writer_lines:
            combined_lines = horizontal_lines | vertical_lines
            if combined_lines.dtype != np.uint8:
                combined_lines = combined_lines.astype(np.uint8)
            if len(combined_lines.shape) == 2:
                combined_lines = cv2.cvtColor(combined_lines, cv2.COLOR_GRAY2BGR)
            lines_result_img = cv2.resize(combined_lines, frame_size)
        else:
            cv2.imwrite("Lines.jpg", horizontal_lines | vertical_lines)

        completed_h_peaks = interpolate_missing_lines(merged_h_peaks, expected_count=10, axis_length=target_height)
        completed_v_peaks = interpolate_missing_lines(merged_v_peaks, expected_count=9, axis_length=target_width)

        completed_h_peaks = [int(y) for y in completed_h_peaks]
        completed_v_peaks = [int(x) for x in completed_v_peaks]

        result = warped.copy()
        for y in completed_h_peaks:
            cv2.line(result, (0, y), (target_width, y), (0, 0, 255), 2)
        for x in completed_v_peaks:
            cv2.line(result, (x, 0), (x, target_height), (255, 0, 0), 2)

        return result, completed_h_peaks, completed_v_peaks, lines_result_img, warped_1, None, intermediate_results, warped_3


def _is_valid_contour(contour, image_area,
                      min_area_ratio, max_area_ratio,
                      min_aspect_ratio, max_aspect_ratio):
    area = cv2.contourArea(contour)
    ratio = area / float(image_area)
    if ratio < min_area_ratio or ratio > max_area_ratio:
        return False
    return True


def match_board_using_template(
        image_path,
        template_path,
        rotate_angles=None,
        scales=None,
        method=cv2.TM_CCOEFF_NORMED,
        score_threshold=0.8
):
    if rotate_angles is None:
        rotate_angles = [0, 90, 180, 270]
    if scales is None:
        scales = [0.5, 0.8, 1.0, 1.2, 1.5]

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")
    tpl = cv2.imread(template_path)
    if tpl is None:
        raise ValueError(f"无法读取模板图像: {template_path}")

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_tpl = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)

    h_img, w_img = gray_img.shape
    h_tpl_org, w_tpl_org = gray_tpl.shape

    best_score = -1
    best_rect = None

    for angle in rotate_angles:
        M = cv2.getRotationMatrix2D(
            (w_tpl_org / 2, h_tpl_org / 2),
            angle,
            1.0
        )
        cos_val = abs(M[0, 0])
        sin_val = abs(M[0, 1])
        new_w = int(h_tpl_org * sin_val + w_tpl_org * cos_val)
        new_h = int(h_tpl_org * cos_val + w_tpl_org * sin_val)
        M[0, 2] += (new_w / 2) - (w_tpl_org / 2)
        M[1, 2] += (new_h / 2) - (h_tpl_org / 2)
        rotated_tpl = cv2.warpAffine(gray_tpl, M, (new_w, new_h))

        for scale in scales:
            if scale <= 0:
                continue
            new_tw = int(new_w * scale)
            new_th = int(new_h * scale)
            if new_tw < 10 or new_th < 10:
                continue
            scaled_tpl = cv2.resize(rotated_tpl, (new_tw, new_th), interpolation=cv2.INTER_AREA)

            print(f"Image shape: {gray_img.shape}")
            print(f"Template shape: {scaled_tpl.shape}")

            if new_tw > w_img or new_th > h_img:
                continue

            result = cv2.matchTemplate(gray_img, scaled_tpl, method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            if method in [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED]:
                score = max_val
                best_pt = max_loc
            else:
                score = 1 - min_val
                best_pt = min_loc

            if score > best_score:
                best_score = score
                top_left = best_pt
                best_rect = (top_left[0], top_left[1], new_tw, new_th)

    if best_rect is not None and best_score >= score_threshold:
        x, y, w_box, h_box = best_rect
        print(f"匹配分数={best_score:.3f}, 棋盘矩形=({x}, {y}, {w_box}, {h_box})")

        out_vis = img.copy()
        cv2.rectangle(out_vis, (x, y), (x + w_box, y + h_box), color=(0, 255, 0), thickness=3)
        cropped = img[y:y + h_box, x:x + w_box]
        cv2.imwrite("matched_board_region.jpg", cropped)
        cv2.imwrite("matched_board_visual.jpg", out_vis)
        return best_rect, best_score
    else:
        print("未在多角度多尺度下找到符合阈值的匹配结果。")
        return None, None


def detect_chinese_chessboard_by_lines(
        image_path,
        canny_threshold1=80,
        canny_threshold2=200,
        min_line_length=250,
        max_line_gap=100,
        angle_tolerance_deg=15,
        padding=20,
        max_gap_between_lines=400,
        min_line_distance=50,
        grid_tolerance=0.1,
        morph_kernel_size=10,
):
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return None, None

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    adaptive_thres = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel_size, 1))
    horizontal_lines_img = cv2.erode(adaptive_thres, horizontal_kernel, iterations=2)
    horizontal_lines_img = cv2.dilate(horizontal_lines_img, horizontal_kernel, iterations=2)

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, morph_kernel_size))
    vertical_lines_img = cv2.erode(adaptive_thres, vertical_kernel, iterations=2)
    vertical_lines_img = cv2.dilate(vertical_lines_img, vertical_kernel, iterations=2)

    edges = cv2.bitwise_or(horizontal_lines_img, vertical_lines_img)

    lines_p = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )
    if lines_p is None:
        print("未检测到任何线段!")
        return None, img

    annotated = img.copy()
    vertical_lines_list = []
    horizontal_lines_list = []

    def angle_of_line(x1, y1, x2, y2):
        dx = x2 - x1
        dy = y2 - y1
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        if angle_deg < 0:
            angle_deg += 180
        return angle_deg

    for ln in lines_p:
        x1, y1, x2, y2 = ln[0]
        ang_deg = angle_of_line(x1, y1, x2, y2)

        if (abs(ang_deg - 0) < angle_tolerance_deg) or (abs(ang_deg - 180) < angle_tolerance_deg):
            horizontal_lines_list.append((x1, y1, x2, y2))
            cv2.line(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        elif abs(ang_deg - 90) < angle_tolerance_deg:
            vertical_lines_list.append((x1, y1, x2, y2))
            cv2.line(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)

    if len(vertical_lines_list) < 2 or len(horizontal_lines_list) < 2:
        print("未检测到足够的有效线条，无法定位棋盘边界.")
        return None, annotated

    vertical_lines_list.sort(key=lambda ln: ln[0])
    horizontal_lines_list.sort(key=lambda ln: ln[1])

    left_x = vertical_lines_list[0][0]
    right_x = vertical_lines_list[-1][0]
    top_y = horizontal_lines_list[0][1]
    bottom_y = horizontal_lines_list[-1][1]

    x1 = max(0, left_x - padding)
    y1 = max(0, top_y - padding)
    x2 = min(w, right_x + padding)
    y2 = min(h, bottom_y + padding)

    cropped_img = img[y1:y2, x1:x2]
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)

    return cropped_img, annotated


def generalized_hough_match(
        image_path,
        template_path,
        canny_thresh1=50,
        canny_thresh2=150,
        dp=2.0,
        minDist=100,
        canny_aperture=3,
        max_buffer_size=1000
):
    src_bgr = cv2.imread(image_path)
    if src_bgr is None:
        raise IOError(f"无法读取图像: {image_path}")
    tpl_bgr = cv2.imread(template_path)
    if tpl_bgr is None:
        raise IOError(f"无法读取模板: {template_path}")

    target_width = 800
    target_height = 800
    tpl_bgr_resized = cv2.resize(tpl_bgr, (target_width, target_height), interpolation=cv2.INTER_AREA)
    tpl_gray = cv2.cvtColor(tpl_bgr_resized, cv2.COLOR_BGR2GRAY)

    src_gray = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY)

    src_edges = cv2.Canny(src_gray, canny_thresh1, canny_thresh2, apertureSize=canny_aperture)
    tpl_edges = cv2.Canny(tpl_gray, canny_thresh1, canny_thresh2, apertureSize=canny_aperture)

    tpl_h, tpl_w = tpl_edges.shape[:2]
    template_center = (tpl_w // 2, tpl_h // 2)

    gh = cv2.createGeneralizedHoughGuil()
    gh.setTemplate(tpl_edges, template_center)
    gh.setDp(dp)
    gh.setMinDist(minDist)
    gh.setMaxBufferSize(max_buffer_size)

    positions, votes = gh.detect(src_edges)

    print("Positions:", positions)
    print("Votes:", votes)

    if positions is None or len(positions) == 0:
        print("未检测到任何形变匹配结果。")
        return [], []

    out_vis = src_bgr.copy()
    for (x, y, s, ang), v in zip(positions, votes):
        if v < 10:
            continue

        cv2.circle(out_vis, (int(x), int(y)), 5, (0, 0, 255), -1)
        corners = np.array([
            [0, 0],
            [tpl_w, 0],
            [tpl_w, tpl_h],
            [0, tpl_h]
        ], dtype=np.float32) - np.array(template_center, dtype=np.float32)

        angle_rad = np.deg2rad(ang)
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ], dtype=np.float32)
        corners = s * corners.dot(rotation_matrix.T)
        corners += np.array([x, y], dtype=np.float32)
        corners = corners.astype(np.int32)

        cv2.polylines(out_vis, [corners], True, (0, 255, 0), 2)
        label = f"vote={v}, scale={s:.2f}, angle={ang:.1f}"
        cv2.putText(out_vis, label, (corners[0][0], corners[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    cv2.imwrite("ghough_result.jpg", out_vis)
    print("检测完成，结果已写入 ghough_result.jpg")

    return positions, votes


def expand_bbox(bbox, img_shape, expand_ratio=0.1):
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1

    dx = width * expand_ratio
    dy = height * expand_ratio

    expanded_x1 = max(0, x1 - dx)
    expanded_y1 = max(0, y1 - dy)
    expanded_x2 = min(img_shape[1], x2 + dx)
    expanded_y2 = min(img_shape[0], y2 + dy)

    return np.array([expanded_x1, expanded_y1, expanded_x2, expanded_y2])


def filter_noise_points(points, eps=50, min_samples=5):
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_

    unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
    if len(counts) == 0:
        return points

    main_cluster = unique_labels[np.argmax(counts)]
    filtered_points = points[labels == main_cluster]

    return filtered_points


def resize_with_aspect_ratio(image, target_size=(640, 384)):
    h, w = image.shape[:2]
    scale = min(target_size[0] / w, target_size[1] / h)
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(image, new_size), scale


def mask_data_to_contour(data_mask, orig_img, visualize=True,
                         target_size=(640, 384), expand_ratio=0.1,
                         bottom_expand_ratio=0.1, video_writer_contour_expansion=None, frame_size=None):
    resized_img, scale = resize_with_aspect_ratio(orig_img, target_size)
    h_resized, w_resized = resized_img.shape[:2]

    mask = data_mask.cpu().numpy().squeeze()
    mask = (mask > 0.5).astype(np.uint8)
    if mask.shape != (h_resized, w_resized):
        mask = cv2.resize(mask, (w_resized, h_resized))

    kernel = np.ones((5, 5), np.uint8)
    clean_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("未找到有效轮廓")

    main_contour = max(contours, key=cv2.contourArea)
    hull_contour = cv2.arcLength(main_contour, True)
    epsilon, max_iter = 0.005, 10
    target_vertices = 4
    for _ in range(max_iter):
        approx = cv2.approxPolyDP(main_contour, epsilon * hull_contour, True)
        if len(approx) == target_vertices:
            approx_contour = approx
            break
        epsilon *= 1.5

    def expand_contour(contour, main_ratio, bottom_ratio):
        M = cv2.moments(contour)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        expanded = []
        for point in contour.squeeze():
            dx = point[0] - cx
            dy = point[1] - cy

            expand_x = dx * main_ratio

            if dy > 0:
                expand_y = dy * bottom_ratio
            else:
                expand_y = dy * main_ratio

            new_x = int(point[0] + expand_x)
            new_y = int(point[1] + expand_y)

            expanded.append([new_x, new_y])

        return np.array(expanded, dtype=np.int32).reshape(-1, 1, 2)

    expanded_contour = expand_contour(approx_contour, expand_ratio, bottom_expand_ratio)
    final_contour = expanded_contour
    board_contour = (final_contour / scale).astype(np.int32)

    board_contour[:, :, 0] = np.clip(board_contour[:, :, 0], 0, orig_img.shape[1] - 1)
    board_contour[:, :, 1] = np.clip(board_contour[:, :, 1], 0, orig_img.shape[0] - 1)

    debug_img = None
    if visualize:
        debug_img = resized_img.copy()
        cv2.drawContours(debug_img, [approx_contour], -1, (0, 255, 0), 2)
        cv2.drawContours(debug_img, [expanded_contour], -1, (0, 0, 255), 2)

        cv2.putText(debug_img, f"Main Expand: {expand_ratio}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(debug_img, f"Bottom Expand: {bottom_expand_ratio}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        if video_writer_contour_expansion:
            debug_img = cv2.resize(debug_img, frame_size)
        else:
            cv2.imwrite('contour_expansion.jpg', debug_img)
    return board_contour.squeeze(), debug_img


def sort_corners(corners):
    corners = corners.reshape(-1, 2)
    center = np.mean(corners, axis=0)

    angles = np.arctan2(corners[:, 1] - center[1],
                        corners[:, 0] - center[0])

    sorted_indices = np.argsort(angles)
    sorted_corners = corners[sorted_indices]

    if np.linalg.norm(sorted_corners[0] - center) > np.linalg.norm(sorted_corners[1] - center):
        sorted_corners = np.roll(sorted_corners, 1, axis=0)

    return sorted_corners.reshape(-1, 1, 2)


def bbox_to_contour(xyxy, orig_img, visualize=True):
    x1, y1, x2, y2 = map(int, xyxy[0])

    board_contour = np.array([
        [[x1, y1]],
        [[x2, y1]],
        [[x2, y2]],
        [[x1, y2]]
    ], dtype=np.int32)

    if visualize:
        vis_img = orig_img.copy()
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.polylines(vis_img, [board_contour], True, (0, 255, 0), 3)

        for i, point in enumerate(board_contour):
            cv2.circle(vis_img, tuple(point[0]), 5, (0, 0, 255), -1)
            cv2.putText(vis_img, str(i), tuple(point[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imwrite('board_contour_visualization.jpg', vis_img)

        x, y, w, h = cv2.boundingRect(board_contour)
        cropped_board = orig_img[y:y + h, x:x + w]
        cv2.imwrite("cropped_board.jpg", cropped_board)

    return board_contour


# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":

    # ================= 配置区域 (请在此处修改路径) =================
    # 待检测的视频或图片路径
    TEST_MEDIA_PATH = "path/to/your/video_or_image.jpg"

    # 棋子检测 YOLO 模型路径
    PIECE_MODEL_PATH = "path/to/your/piece_detection/best.pt"

    # 棋盘检测 YOLO 模型路径
    BOARD_MODEL_PATH = "path/to/your/board_detection/best.pt"

    # 字体文件路径 (用于中文显示)
    FONT_PATH = "SIMHEI.TTF"
    # ============================================================

    # 检查路径是否存在 (简单的检查)
    if not os.path.exists(FONT_PATH):
        print(f"Warning: Font file {FONT_PATH} not found. Chinese characters may not render correctly.")

    # 加载模型
    # 注意：确保 weights 路径正确
    try:
        model = YOLO(PIECE_MODEL_PATH)
        model_board = YOLO(BOARD_MODEL_PATH)
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Please check the model paths in the configuration section.")
        exit(1)

    # 运行推断
    # save=False, save_txt=False, show=False 是为了避免在服务器上弹出窗口
    try:
        results = model(TEST_MEDIA_PATH, stream=True, save=False, save_txt=False, show=False)
    except Exception as e:
        print(f"Error processing media: {e}")
        exit(1)

    # Video parameters
    fps = 30.0
    frame_width = 640
    frame_height = 480
    frame_size = (frame_width, frame_height)

    MP4_format = False
    memory_length = 30
    chessboard_memory = ChessboardMemory(memory_length)

    all_game_states = []
    tmp_board = None

    if TEST_MEDIA_PATH.endswith(".mp4"):
        video_writer = cv2.VideoWriter("board_detection_result.mp4", cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps,
                                       frame_size)
        video_writer_lines = cv2.VideoWriter("lines.mp4", cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, frame_size)
        video_writer_contour_expansion = cv2.VideoWriter("contour_expansion.mp4",
                                                         cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, frame_size)
        video_writer_warped_board = cv2.VideoWriter("warped_board.mp4", cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps,
                                                    frame_size)
        video_writer_sam = cv2.VideoWriter("sam.mp4", cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, frame_size)

        video_writer_warped_board_third = cv2.VideoWriter("warped_board_third.mp4",
                                                          cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, frame_size)
        video_writer_vis = cv2.VideoWriter("vis.mp4", cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, frame_size)
        video_writer_vis_refine = cv2.VideoWriter("vis_refine.mp4", cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps,
                                                  frame_size)
        MP4_format = True

    blank_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

    for i, result in enumerate(results):
        # if i >= 100:break # Uncomment if you want to limit frames

        cls = result.boxes.cls.cpu().numpy()
        xyxy = result.boxes.xyxy.cpu().numpy()

        orig_img = result.orig_img
        orig_img_ = result.orig_img.copy()

        intermediate_results = []

        for k, cls_id in enumerate(cls):
            if cls_id == 0:
                continue

            x1, y1, x2, y2 = xyxy[k]
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            piece_name = cls_mapping[int(cls_id)]

            intermediate_results.append({
                'piece_name': piece_name,
                'center': (center_x, center_y),
                'bbox': (x1, y1, x2, y2)
            })

            cv2.circle(orig_img_, (int(center_x), int(center_y)), 10, (0, 255, 0), -1)
            cv2.putText(orig_img_, piece_name, (int(center_x), int(center_y) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

        board_detection_result_img = None
        sam_segmentation_result_img = None

        if True:
            try:
                results_board = model_board(orig_img, save=False, save_txt=False, show=False)
                cls_board = results_board[0].boxes.cls.cpu().numpy()

                if len(cls_board) < 4:
                    raise ValueError("未检测到棋盘四个角点")

                xywh_board = results_board[0].boxes.xywh.cpu().numpy()

                class_xyxy_mapping = defaultdict(list)
                for m in range(len(cls_board)):
                    cls_index = int(cls_board[m])
                    xyxy_coords = xywh_board[m][:2]
                    class_xyxy_mapping[cls_index].append(xyxy_coords)

                if TEST_MEDIA_PATH.endswith(".mp4"):
                    sam_result = cv2.resize(results_board[0].plot(), frame_size)
                    video_writer_sam.write(sam_result)

                if True:
                    # 假定 class 0,1,2,3 分别对应棋盘的四个角
                    board_contour = np.array(
                        [class_xyxy_mapping[0][0], class_xyxy_mapping[1][0], class_xyxy_mapping[2][0],
                         class_xyxy_mapping[3][0]])

                    board_detection_result_img, completed_h_peaks, completed_v_peaks, lines_result_img, warped_1, _, transformed_points, warped_3 \
                        = detect_and_analyze_board(
                        orig_img, board_contour=board_contour, video_writer_lines=True,
                        frame_size=frame_size, video_writer_warped_board=True,
                        video_writer_warped_board_second=True, intermediate_results=intermediate_results)

                    board_tmp, board = generate_chessboard(completed_h_peaks, completed_v_peaks, transformed_points)
                    vector_board_numpy = board_to_vector_numpy(board)
                    chessboard_memory.add_frame(vector_board_numpy)
                    smoothed_board_numpy = chessboard_memory.get_smoothed_board()

                    if smoothed_board_numpy is not None:
                        if tmp_board is None:
                            tmp_board = smoothed_board_numpy
                            print("棋局状态发生变化！")
                        else:
                            if not np.array_equal(tmp_board, smoothed_board_numpy):
                                print("棋局状态发生变化！")
                                tmp_board = smoothed_board_numpy
                                smoothed_board = vector_to_board_numpy(smoothed_board_numpy)
                                all_game_states.append(smoothed_board)
                                print(f"已保存新的棋局状态，当前共保存 {len(all_game_states)} 个状态。")
                                board_refine = vis_board(smoothed_board)

                    image_bgr = vis_board(board)
                    board_detection_result_img = cv2.resize(board_detection_result_img, frame_size)

                    if MP4_format:
                        video_writer.write(board_detection_result_img)
                        video_writer_lines.write(lines_result_img)
                        video_writer_warped_board.write(warped_1)
                        video_writer_warped_board_third.write(warped_3)
                        image_bgr = cv2.resize(image_bgr, (640, 480))
                        video_writer_vis.write(image_bgr)
                        if 'board_refine' in locals():
                            board_refine = cv2.resize(board_refine, (640, 480))
                            video_writer_vis_refine.write(board_refine)
                    else:
                        images_to_show = [orig_img, board_detection_result_img, lines_result_img,
                                          warped_1, warped_3, image_bgr]
                        image_names = ["Original Image", "Board Detection Result", "Lines Result",
                                       "Warped Board 1", "Warped Board with pieces", "simulate_chess_board"]

                        fig_width = 640
                        fig_height = 480
                        rows = 3
                        cols = 2
                        fig = np.zeros((fig_height * rows, fig_width * cols, 3), dtype=np.uint8)

                        for idx in range(len(images_to_show)):
                            row_index = idx // cols
                            col_index = idx % cols
                            img_show = images_to_show[idx]
                            name = image_names[idx]
                            if img_show is not None:
                                resized_img = cv2.resize(img_show, (fig_width, fig_height))
                                x_start = col_index * fig_width
                                y_start = row_index * fig_height
                                fig[y_start:y_start + fig_height, x_start:x_start + fig_width, :] = resized_img
                                cv2.putText(fig, name, (x_start + 10, y_start + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                            (255, 255, 255), 1, cv2.LINE_AA)

                        cv2.imwrite("Image Grid.jpg", fig)

            except Exception as e_sam:
                print(f"处理失败 (Frame {i}): {e_sam}")
                if MP4_format:
                    video_writer.write(blank_frame)
                    video_writer_lines.write(blank_frame)
                    video_writer_warped_board.write(blank_frame)
                    video_writer_warped_board_third.write(blank_frame)
                    video_writer_vis.write(blank_frame)
                    if tmp_board is not None:
                        board_refine = vis_board(vector_to_board_numpy(tmp_board))
                        board_refine = cv2.resize(board_refine, (640, 480))
                        video_writer_vis_refine.write(board_refine)
                    else:
                        video_writer_vis_refine.write(blank_frame)

    # 释放资源
    if MP4_format:
        video_writer.release()
        video_writer_lines.release()
        video_writer_contour_expansion.release()
        video_writer_warped_board.release()
        video_writer_sam.release()
        video_writer_warped_board_third.release()
        video_writer_vis.release()
        video_writer_vis_refine.release()
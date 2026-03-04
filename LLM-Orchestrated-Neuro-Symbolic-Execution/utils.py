import cv2
import numpy as np
import requests
import os


def extract_subject_and_remove_background(image_path, output_path):
    """
    提取图片中的棋盘主体，并移除背景设置为透明。

    Args:
        image_path: 输入图片路径。
        output_path: 输出图片路径 (需要是 PNG 格式)。
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return

    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return

    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    rgba_img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)

    # 1. 边缘检测 (保持原数字)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # 2. 轮廓检测
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 3. 轮廓筛选
    largest_contour = None
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            largest_contour = contour

    if largest_contour is None:
        print("Could not find a suitable contour.")
        return

    # 4. 透视变换 (保持原数字)
    epsilon = 0.04 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    if len(approx) == 4:
        pts = np.float32(approx)
        rect = cv2.boundingRect(approx)

        # 计算四个顶点
        pts1 = pts.reshape(4, 2)
        pts2 = np.float32([[0, 0], [rect[2], 0], [rect[2], rect[3]], [0, rect[3]]])

        M = cv2.getPerspectiveTransform(pts1, pts2)
        warp = cv2.warpPerspective(rgba_img, M, (rect[2], rect[3]))  # 注意：warp 变量虽计算但未直接使用，逻辑保留原样

        # 创建一个mask，将棋盘的区域设置为不透明 (保持原数字)
        warp_mask = np.zeros((rect[3], rect[2]), dtype=np.uint8)
        warp_mask[:, :] = 255

        # 将透视变换后的掩码应用到原图的掩码上
        M_inv = cv2.getPerspectiveTransform(pts2, pts1)
        warp_mask_back = cv2.warpPerspective(warp_mask, M_inv, (rgba_img.shape[1], rgba_img.shape[0]))

        rgba_img[:, :, 3] = warp_mask_back
    else:
        print("Could not find 4 corners, skip perspective transform")

    cv2.imwrite(output_path, rgba_img)
    print(f"Saved processed image to {output_path}")


def filter_moves_by_highest_rank(result_string):
    """
    从结果字符串中提取具有最高 rank 的最多前3个走法，并按指定格式返回。

    Args:
        result_string: 包含所有走法信息的字符串。

    Returns:
        包含筛选后走法信息的列表。
    """
    moves = result_string.split('|')
    highest_rank = -1  # 初始化最高 rank

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
                # score 虽解析但未用于逻辑判断，保留原样
                score = int(move_dict['score'])
                if rank == highest_rank:
                    filtered_moves.append(move_dict)
                    count += 1
                    # if count >= 2: break # 原始代码逻辑注释掉了限制，此处保持注释状态或移除
            except ValueError:
                continue

    output_strings = []
    for move_dict in filtered_moves:
        output_strings.append(
            f"move:{move_dict.get('move', '')},"
            f"score:{move_dict.get('score', '')},"
            f"winrate:{move_dict.get('winrate', '')}"
        )

    return output_strings


def get_best_move(board_fen, side_to_move=None):
    """
    获取最佳着法

    Args:
        board_fen (str): 当前棋局的 FEN 表示
        side_to_move (str): 'w' 或 'b' (可选)

    Returns:
        str/list: 最佳着法或错误信息
    """
    # 云库 API 的基本 URL
    base_url = "http://www.chessdb.cn/chessdb.php"

    # 设置参数
    params = {
        "action": "queryall",  # 获取最佳着法
        "board": board_fen,  # 当前棋局 FEN 表示
        # "w": side_to_move,   # 哪一方的回合 (保留原注释)
    }

    try:
        # 发送请求到云库 API
        response = requests.get(base_url, params=params)
    except requests.RequestException as e:
        return f"请求异常: {e}"

    # 检查返回结果
    if response.status_code == 200:
        result = response.text
        print(f"API Response: {result}")

        if "nobestmove" in result:
            return "没有最佳着法"
        elif "move" in result:
            filtered_result_highest_rank = filter_moves_by_highest_rank(result)
            return filtered_result_highest_rank
        elif "unknown" in result:
            return "该局面未被收录，请尝试其他走法"
        elif "invalid board" in result:
            return "未知错误：无效的棋局"
        else:
            return "其他错误"
    else:
        return "请求失败"


if __name__ == '__main__':
    # ---------------- 配置部分 ----------------
    # 使用相对路径，方便开源用户直接运行
    # 请确保同级目录下存在该图片文件
    input_image_path = "myChineseChess.jpg"
    output_image_path = "myChineseChess_wo_background.png"

    # 1. 图片处理功能示例 (默认注释，需要时取消注释)
    # extract_subject_and_remove_background(input_image_path, output_image_path)

    # ---------------- 棋局分析部分 ----------------

    # 示例 FEN 串 (UBB 编码或标准 FEN)
    # board_fen = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"
    board_fen = "rnb1kabnr/4a4/4c2c1/p3p1p1p/2p6/9/P1P1P1P1P/2N1C2C1/4A4/R1B1KABNR r"

    print(f"当前分析局面: {board_fen}")

    best_moves = get_best_move(board_fen, side_to_move=None)

    if isinstance(best_moves, list):
        print("推荐最佳着法:")
        for move in best_moves:
            print(move)
    else:
        print(f"结果: {best_moves}")
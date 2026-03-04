#!/usr/bin/env bash
# hd_epic_config.sh ── 全局配置参数

# ▍全局路径配置 (请根据实际环境修改) --------------------------------
# 数据集根目录
HD_EPIC_DB="/path/to/HD-EPIC/"

# VQA 注释目录
HD_EPIC_VQA="/path/to/hd-epic-annotations/vqa-benchmark/"

# 任务名称/输出目录名
TASK_NAME="PreprocessedVideos_v7"

# 模型路径 (可以是本地路径或 HuggingFace ID)
# BASE_MODEL="Qwen/Qwen2.5-VL-32B-Instruct"
BASE_MODEL="/path/to/sft_output_v5/"

# Voting Ensemble 标识
VOTING_ENSEMBLE="4"

# ▍运行参数 (严格保留原始设置) -------------------------------------
INFERENCE_PCT=100                 # 全局推理百分比
# GPU_SETS=("0,1" "2,3" "4,5" "6,7") # 示例：双卡并行

# 当前设置：每任务独占一张卡 (5, 6, 7)
GPU_SETS=(5 6 7)
MAX_JOBS=3                        # 最大并发进程数

# ▍任务列表定义 ----------------------------------------------------

# --- 图像任务 ---
 IMAGE_TASKS=(
   3d_perception_fixture_location
 )

# --- 三阶段任务 (Segment) ---
TRIPLE_TASKS_SEG=(
  3d_perception_object_contents_retrieval
  3d_perception_object_location
)

# --- 多图推理任务 ---
IMAGE_MULTI_TASKS=(
  nutrition_image_nutrition_estimation
)

# --- 三阶段任务 (Refine Inference) ---
TRIPLE_TASKS=(
  3d_perception_fixture_interaction_counting
  object_motion_object_movement_counting
  object_motion_object_movement_itinerary
  object_motion_stationary_object_localization
)

# --- 细粒度与 Gaze 任务 (Interval) ---
FG_GAZE_INTERVAL_TASKS=(
  fine_grained_action_recognition
  fine_grained_how_recognition
  fine_grained_why_recognition
  gaze_gaze_estimation
  gaze_interaction_anticipation
  ingredient_ingredient_retrieval
  nutrition_nutrition_change
  recipe_step_recognition
)

# --- 简单视频任务 ---
VIDEO_SIMPLE_TASKS=(
  nutrition_video_nutrition_estimation
  recipe_following_activity_recognition
  recipe_multi_recipe_recognition
)

# --- 多视频合并任务 (Concat) ---
MULTI_VIDEO_CONCAT_TASKS=(
  ingredient_exact_ingredient_recognition
  ingredient_ingredient_recognition
  recipe_recipe_recognition
)

# --- Chunk + Wovision 任务 ---
 CHUNK_WOVISION_TASKS=(
   ingredient_ingredient_weight
   ingredient_ingredients_order
 )

# --- Wovision (Non-Concat) 任务 ---
 CHUNK_WOVISION_NONCONCAT_TASKS=(
   ingredient_ingredient_adding_localization
   recipe_multi_step_localization
   recipe_rough_step_localization
   fine_grained_action_localization
 )

# --- Multi Video + Wovision 任务 ---
 MULTI_VIDEO_WOVISION_NONCONCAT_TASKS=(
   recipe_prep_localization
   recipe_step_localization
 )
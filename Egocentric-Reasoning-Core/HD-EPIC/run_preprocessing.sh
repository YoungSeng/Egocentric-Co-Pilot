#!/usr/bin/env bash
# run_preprocessing.sh
# 用法：
#   ./run_preprocessing.sh

set -euo pipefail

###############################################################################
# 1. 路径配置 (请根据实际情况修改此处)
###############################################################################
# 假设配置文件存在，如果不存在请直接修改下面的路径变量
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${SCRIPT_DIR}/hd_epic_config.sh" ]; then
    source "${SCRIPT_DIR}/hd_epic_config.sh"
fi

# 如果配置文件中没有定义，请在这里指定数据集路径
: "${HD_EPIC_VQA:="./dataset/hd-epic-annotations/vqa-benchmark"}"
: "${HD_EPIC_DB:="./dataset/HD-EPIC"}"

HD_EPIC_VQA_ANNOTATIONS="$HD_EPIC_VQA"
HD_EPIC_DATABASE="$HD_EPIC_DB"

# 任务名称 (如果未定义，默认为 PreprocessedVideos)
TASK_NAME="${TASK_NAME:-PreprocessedVideos}"

###############################################################################
# 2. 参数设置
###############################################################################
# 把公用参数放到数组，调用时统一展开
COMMON_ARGS=(
  --hd_epic_vqa_annotations "$HD_EPIC_VQA_ANNOTATIONS"
  --hd_epic_database        "$HD_EPIC_DATABASE"
  --task_name               "$TASK_NAME"
)

###############################################################################
# 3. 工具函数：执行并计时
###############################################################################
run_task () {
  local task_name="$1"   # --task 的具体名称
  shift                  # 其余为附加参数

  # 注意这里多了一个 “--” 来禁止 printf 解析后续为选项
  printf -- "\n===== 运行 %-45s =====\n" "$task_name"
  local start_ts=$(date +%s)

  python data_preprocessing.py --task "$task_name" "$@" "${COMMON_ARGS[@]}"

  local end_ts=$(date +%s)
  printf -- "----- %-45s 完成，耗时 %4d 秒 -----\n" "$task_name" $(( end_ts - start_ts ))
}

###############################################################################
# 4. 任务清单
###############################################################################

# 3d_perception
run_task 3d_perception_fixture_location               --extract_keyframe --generate_refined_prompt
run_task 3d_perception_fixture_interaction_counting   --extract_keyframe --generate_refined_prompt
run_task 3d_perception_object_contents_retrieval      --extract_keyframe --generate_refined_prompt
run_task 3d_perception_object_location                --extract_keyframe

# fine_grained
run_task fine_grained_action_recognition              --generate_refined_prompt
run_task fine_grained_how_recognition                 --generate_refined_prompt
run_task fine_grained_why_recognition                 --generate_refined_prompt

# gaze
run_task gaze_gaze_estimation                         --generate_refined_prompt
run_task gaze_interaction_anticipation                --generate_refined_prompt

# ingredient
run_task ingredient_exact_ingredient_recognition      --generate_refined_prompt
run_task ingredient_ingredient_recognition            --generate_refined_prompt
run_task ingredient_ingredient_retrieval              --generate_refined_prompt

# nutrition
run_task nutrition_image_nutrition_estimation         --generate_refined_prompt
run_task nutrition_nutrition_change                   --generate_refined_prompt
run_task nutrition_video_nutrition_estimation         --generate_refined_prompt

# object_motion
run_task object_motion_object_movement_counting       --extract_keyframe
run_task object_motion_object_movement_itinerary      --extract_keyframe
run_task object_motion_stationary_object_localization --extract_keyframe

# recipe
run_task recipe_following_activity_recognition        --generate_refined_prompt
run_task recipe_multi_recipe_recognition              --generate_refined_prompt
run_task recipe_prep_localization                     --generate_refined_prompt --refined_question_later
run_task recipe_recipe_recognition                    --generate_refined_prompt
run_task recipe_step_localization                     --generate_refined_prompt --refined_question_later
run_task recipe_step_recognition                      --generate_refined_prompt
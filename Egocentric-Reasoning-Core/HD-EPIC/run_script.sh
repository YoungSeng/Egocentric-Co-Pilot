#!/usr/bin/env bash
set -euo pipefail

# 获取当前脚本所在目录并加载配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/hd_epic_config.sh"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found at $CONFIG_FILE"
    exit 1
fi
# shellcheck disable=SC1091
source "$CONFIG_FILE"

# ───────────────────── 通用函数 ────────────────────────────────
run_with_timer() {
  local label="$1"; shift
  printf "\n========== %s ==========\n" "$label"
  local start=$(date +%s)
  "$@"
  local end=$(date +%s)
  echo "✅  Done in $((end - start)) s"
}

# ⤷ 并行调度器：把任务放后台，控制 MAX_JOBS，按 GPU_SETS 分配且独占 GPU
declare -A GPU_IN_USE=()    # 记录 GPU 是否被占用
declare -A PID_TO_GPU=()    # 记录进程 PID 对应的 GPU

launch_job() {
  local label="$1"; shift
  local gpu

  # 寻找可用 GPU
  while true; do
    for gpu in "${GPU_SETS[@]}"; do
      if [[ -z "${GPU_IN_USE[$gpu]-}" || "${GPU_IN_USE[$gpu]}" -eq 0 ]]; then
        GPU_IN_USE[$gpu]=1  # 标记为占用
        break 2
      fi
    done
    wait_for_any_job        # 无可用 GPU 时等待任务完成
  done

  # 启动后台任务
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    run_with_timer "$label (GPU:$gpu)" "$@"
  ) &
  local pid=$!
  PID_TO_GPU[$pid]="$gpu"  # 记录 PID 与 GPU 的映射

  echo "[分配] $label -> GPU $gpu (剩余显存: $(nvidia-smi -i $gpu --query-gpu=memory.free --format=csv,noheader,nounits))MB"

  # 控制最大并行任务数
  while (( $(jobs -r | wc -l) >= MAX_JOBS )); do
    wait_for_any_job
  done
}

# 等待任务完成并清理 GPU 占用状态
wait_for_any_job() {
  wait -n  # 等待任意后台任务完成

  # 清理所有已完成的进程
  local completed_pids=()
  for pid in "${!PID_TO_GPU[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then  # 检查进程是否存活
      completed_pids+=("$pid")
    fi
  done

  # 释放 GPU 资源
  for pid in "${completed_pids[@]}"; do
    local gpu="${PID_TO_GPU[$pid]-}"
    if [[ -n "$gpu" ]]; then
      GPU_IN_USE["$gpu"]=0
      unset PID_TO_GPU["$pid"]
    fi
  done
}

# ───────────────────── 任务处理函数定义 ─────────────────────

process_triple_task_narration_refine_inference() {
  local TASK="$1"
  # thinking-narration
  python 2-thinking-narration.py --task "$TASK" --inference_percentage "$INFERENCE_PCT" --base_model "$BASE_MODEL" "${COMMON_PATHS[@]}"
  # refine-prompt
  python 2-refine-prompt.py --task "$TASK" "${COMMON_PATHS[@]}"
  # inference_video
  python 2-inference_video.py --task "$TASK" --inference_percentage "$INFERENCE_PCT" --evaluate_accuracy --base_model "$BASE_MODEL" "${COMMON_PATHS[@]}"
}

process_triple_task_multi_video_wovision() {
  local TASK="$1"
  python 2-thinking-narration-video.py --task "${TASK}" --multi_video --wobbox --video_merge_strategy concat --inference_percentage "$INFERENCE_PCT" --base_model "$BASE_MODEL" "${COMMON_PATHS[@]}"
  python 2-refine-prompt.py --wobbox --task "${TASK}" --pattern video "${COMMON_PATHS[@]}"
  python 2-inference_video.py --task "${TASK}" --wovision --inference_percentage "$INFERENCE_PCT" --evaluate_accuracy --base_model "$BASE_MODEL" "${COMMON_PATHS[@]}"
}

process_triple_task_narration_refine_inference_segment() {
  local TASK="$1"
  python 2-thinking-narration.py --task "$TASK" --inference_percentage "$INFERENCE_PCT" --base_model "$BASE_MODEL" "${COMMON_PATHS[@]}"
  python 2-refine-prompt.py --task "$TASK" "${COMMON_PATHS[@]}"
  python 2-inference_video.py --task "$TASK" --inference_percentage "$INFERENCE_PCT" --evaluate_accuracy --base_model "$BASE_MODEL" --video_segment segment "${COMMON_PATHS[@]}"
}

process_triple_task_chunk_wovision() {
  local TASK="$1"
  python 2-thinking-narration-video.py --task "${TASK}" --chunk_video --inference_percentage "$INFERENCE_PCT" --base_model "$BASE_MODEL" "${COMMON_PATHS[@]}"
  python 2-refine-prompt.py --task "${TASK}" --pattern video "${COMMON_PATHS[@]}"
  python 2-inference_video.py --task "${TASK}" --wovision --inference_percentage "$INFERENCE_PCT" --evaluate_accuracy --base_model "$BASE_MODEL" "${COMMON_PATHS[@]}"
}

process_triple_task_wovision() {
  local TASK="$1"
  python 2-thinking-narration-video.py --task "${TASK}" --inference_percentage "$INFERENCE_PCT" --base_model "$BASE_MODEL" "${COMMON_PATHS[@]}"
  python 2-refine-prompt.py --task "${TASK}" --pattern video "${COMMON_PATHS[@]}"
  python 2-inference_video.py --task "${TASK}" --base_model "$BASE_MODEL" --wovision --inference_percentage "$INFERENCE_PCT" --evaluate_accuracy "${COMMON_PATHS[@]}"
}

COMMON_PATHS=(--hd_epic_database "$HD_EPIC_DB" --hd_epic_vqa_annotations "$HD_EPIC_VQA" --task_name "$TASK_NAME" --voting_ensemble "$VOTING_ENSEMBLE")

# ───────────────────── 执行流程 ─────────────────────

# 1) Image Inference
 for TASK in "${IMAGE_TASKS[@]}"; do
   launch_job "${TASK} / 1-inference_image" \
     python 1-inference_image.py --task "$TASK" --inference_percentage "$INFERENCE_PCT" --evaluate_accuracy --base_model "$BASE_MODEL" "${COMMON_PATHS[@]}"
 done

# 2) narration‑refine‑inference
for TASK in "${TRIPLE_TASKS[@]}"; do
  launch_job "${TASK} / narration‑refine‑inference" \
    run_with_timer "${TASK} Process" process_triple_task_narration_refine_inference "$TASK"
done

# 3) narration‑refine‑inference-segment
for TASK in "${TRIPLE_TASKS_SEG[@]}"; do
  launch_job "${TASK} / narration‑refine‑inference-segment" \
    run_with_timer "${TASK} Process" process_triple_task_narration_refine_inference_segment "$TASK"
done

# 4) fine‑grained & gaze interval & video_interval_misc
for TASK in "${FG_GAZE_INTERVAL_TASKS[@]}"; do
  launch_job "${TASK} / inference_video" \
    python 2-inference_video.py --task "${TASK}" --video_segment interval --wobbox --base_model "$BASE_MODEL" --inference_percentage "$INFERENCE_PCT" --evaluate_accuracy "${COMMON_PATHS[@]}"
done

# 5) concat multi-videos
for TASK in "${MULTI_VIDEO_CONCAT_TASKS[@]}"; do
  launch_job "${TASK} / multi‑videos" \
    python 4-inference_multi-videos.py --task "${TASK}" --wobbox --video_merge_strategy concat --inference_percentage "$INFERENCE_PCT" --evaluate_accuracy --base_model "${BASE_MODEL}" "${COMMON_PATHS[@]}"
done

# 6) chunk + wovision
 for TASK in "${CHUNK_WOVISION_TASKS[@]}"; do
   launch_job "${TASK} / thinking‑narration‑video" \
     run_with_timer "${TASK} Process" process_triple_task_chunk_wovision "$TASK"
 done

# 7) wovision simple
 for TASK in "${CHUNK_WOVISION_NONCONCAT_TASKS[@]}"; do
   launch_job "${TASK} / thinking‑narration‑video" \
     run_with_timer "${TASK} Process" process_triple_task_wovision "$TASK"
 done

# 8) multi_video + wovision
 for TASK in "${MULTI_VIDEO_WOVISION_NONCONCAT_TASKS[@]}"; do
   launch_job "${TASK} / thinking‑narration‑video" \
     run_with_timer "${TASK} Process" process_triple_task_multi_video_wovision "$TASK"
 done

# 9) multi‑images inference
for TASK in "${IMAGE_MULTI_TASKS[@]}"; do
  launch_job "${TASK} / multi‑images" \
    python 3-inference_multi-images.py --task "${TASK}" --inference_percentage "$INFERENCE_PCT" --evaluate_accuracy --base_model "${BASE_MODEL}" "${COMMON_PATHS[@]}"
done

# 10) video_simple_tasks
for TASK in "${VIDEO_SIMPLE_TASKS[@]}"; do
  launch_job "${TASK} / inference_video" \
    python 2-inference_video.py --task "${TASK}" --wobbox --inference_percentage "$INFERENCE_PCT" --evaluate_accuracy --base_model "$BASE_MODEL" "${COMMON_PATHS[@]}"
done

wait
echo "All tasks completed."
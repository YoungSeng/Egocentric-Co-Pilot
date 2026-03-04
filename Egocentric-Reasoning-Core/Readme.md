## Optimizing Multimodal LLMs for Egocentric Video Understanding

2nd Place Solution for the HD-EPIC VQA Challenge CVPR 2025 Second Joint Egocentric Vision (EgoVis) Workshop

This repository contains the core source code, like our pre-trained models and fine-tuning (SFT) implementation.


## 🚀 Quick Start

To verify the model setup, you can download our pre-trained checkpoint `sft_output_v5` and run a simple inference test.
We recommend using **Anaconda** to manage the environment.


### 1. Create Environment
```bash
conda create -n Egocentric-Co-Pilot python=3.10
conda activate Egocentric-Co-Pilot
```

### 2. Install Dependencies
```
pip install torch torchvision transformers accelerate
pip install qwen-vl-utils[decord]==0.0.8
```
Note: The code has been tested on NVIDIA GeForce RTX 4090 with CUDA 12.8. Please ensure your setup is compatible.

### 📥 Model Zoo

To run the inference demo, please download our fine-tuned checkpoint (`sft_output_v5`).

| Model Version | Description | Size | Download Link |
| --- | --- | --- | --- |
| **sft_output_v5** | Core reasoning model (SFT) | ~16 GB | [Baidu Netdisk](https://pan.baidu.com/s/1pGO35fNvwbvM3E1PrvM6TQ?pwd=g344)

### 📂 Directory Structure

After downloading the model, please organize your files as follows to ensure the scripts run correctly:

```text
Egocentric-Reasoning-Core/
├── sft_output_v5/          <-- Place the downloaded model folder here
│   ├── config.json
│   ├── model-00001-of-00004.safetensors
│   └── ...
├── inference.py            <-- Inference script
└── ...
```

### 3. Run Inference

```bash
cd Egocentric-Reasoning-Core
python inference.py
```

If the model is loaded correctly, you should see the predicted answer choice:

```text
D
```

## (Optional) Core Modules

### Fine-tuning

We utilize the official fine-tuning implementation from Qwen2.5-VL. For more details, please refer to the [Qwen2.5-VL Fine-tuning Guide](https://github.com/QwenLM/Qwen2.5-VL/tree/main/qwen-vl-finetune).

The training configuration script for this baseline is located at:
`./qwen-vl-finetune/scripts/sft_7b.sh`

To run the fine-tuning with the default hyperparameters optimized for this baseline, execute the following command:

```bash
# Ensure you are in the root directory of this project
bash ./qwen-vl-finetune/scripts/sft_7b.sh
```


### HD-EPIC Inference (T-CoT)

his section details the inference pipeline for the HD-EPIC benchmark, utilizing the Temporal Chain-of-Thought (T-CoT) strategy.

- Project Website: [hd-epic.github.io](https://hd-epic.github.io)
- Working Directory:
  ```cd HD-EPIC```

1. First, download the necessary video data using the official downloader.

```
# Clone the downloader repository
git clone [https://github.com/hd-epic/hd-epic-downloader](https://github.com/hd-epic/hd-epic-downloader)

# Download video data
# Replace "/path/to/data/" with your actual directory
python hd-epic-downloader.py "/path/to/data/" --videos
```

2. Detailed Inference Scripts

We provide reference scripts for 30 specific tasks (e.g., 3D Perception, Fine-grained Action, Recipe Understanding).
The commands below are provided as a reference to demonstrate our logic for prompt refinement, frame extraction, and inference. You must adapt file paths (e.g., `/path/to/HD-EPIC/`) and GPU configurations (`CUDA_VISIBLE_DEVICES`) to match your local environment.
#### General Pipeline
The typical workflow for any given task involves three main stages:

1. **Preprocessing:** Frame extraction and prompt refinement.
* *Scripts:* `run_preprocessing.sh` or `data_preprocessing.py`


2. **Inference:** Running the model on processed data.
* *Scripts:* `run_script.sh` or specific python scripts (e.g., `2-inference_video.py`)


3. **Evaluation:** Calculating accuracy metrics.
* *Scripts:* `task_accuracy_calculations.py`

#### Global Run Example

```bash
# Step 1: Preprocessing
bash run_preprocessing.sh

# Step 2: Model Inference
bash run_script.sh

# Step 3: Evaluation
python task_accuracy_calculations.py \
    --hd_epic_database /path/to/HD-EPIC/ \
    --task_name sft_7B_v0
```

#### Case Study: Interaction Counting (3D Perception)

To illustrate the T-CoT, we use the Interaction Counting task as an example.
This implementation enhances MLLM reasoning through a two-stage strategy, specifically designed to address spatial and temporal ambiguities in long-form video VQA.
The pipeline moves from a coarse semantic understanding to fine-grained reasoning:

1. **Explicit Cue Exploitation**
* `2-thinking-narration.py` & `2-refine-prompt.py` Resolves abstract `<BBOX>` coordinates into semantic object names (e.g., converting generic coordinates  to "microwave").

2. **Focused Temporal Windowing**
* *Script:* `2-inference_video.py` Dynamically segments the video into focused temporal windows (e.g., s around key timestamps) rather than processing the full duration.

### HCC

Core implementation reference for query-aware long-term video context modeling.
When the context length exceeds the model's threshold, HCC dynamically compresses the information using a two-tier strategy:

1. Recent Event Buffer (High-Res): The most recent $N$ events (e.g., last 50 actions) are preserved in their raw, detailed text format to maintain immediate temporal precision.
2. Historical Context (Compressed): Older events are partitioned into chunks. An LLM generates a concise, query-aware summary for each chunk, filtering out irrelevant noise while retaining causal clues related to the user's question.

To run the evaluation on the EgoLife dataset:

```
python eval_refine.py \
  --results_file ./data/egolife_questions.json \
  --captions_file ./data/captions.json \
  --llm_path /path/to/your/model_weights \
  --output_dir ./outputs \

# --participant_id "A1_JAKE"
# --asr_base_path ./data/asr_logs 
```

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch
from qwen_vl_utils import process_vision_info

model_name = "./sft_output_v5/"


model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    # attn_implementation="flash_attention_2",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_name)

image_path = "./3d_perception_fixture_location_test-1.png"
question = "This image shows my current viewpoint. Based on this image, please determine the direction of the Kitchen paper relative to my viewpoint and choose the closest answer from the options below: \nA: 6 o'clock\nB: 10 o'clock\nC: 1 o'clock\nD: 3 o'clock\nE: 9 o'clock"

# 构建对话消息
messages = [
    {
        "role": "system",
        "content": "You are an expert image analyzer. Answer the multiple choice question by giving ONLY the number identifying the answer. Example: A or B or C or D or E. You MUST answer even if unsure."
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": question}
        ]
    }
]

# 预处理输入
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
    text=[text],
    images=image_inputs,  # 直接使用图片路径
    videos=video_inputs,
    padding=True,
    return_tensors="pt"
).to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]

print(output_text)

# 位置 10.206.215.218
# 使用示例："./inference.py"
# 保存的模型位置：Qwen/Qwen2.5-VL-7B-Instruct
# 模型位置：./sft_output_v5/
# 微调见：https://github.com/QwenLM/Qwen2.5-VL/tree/main/qwen-vl-finetune
# 我们的配置："./Qwen2.5-VL/qwen-vl-finetune/scripts/sft_7b.sh"

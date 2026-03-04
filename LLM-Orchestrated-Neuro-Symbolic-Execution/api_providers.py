import os
import ssl
import json
import requests
import torch
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI

# 全局设置 SSL 上下文（保持原逻辑）
ssl._create_default_https_context = ssl._create_unverified_context


class text_based_llm:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct-AWQ"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
        ).to(self.device)
        self.system_message = ("作为一个AI助手，你的任务是将象棋引擎提供的走法建议从棋盘语言转化为人类可读的自然语言。\n"
                               "输入是走法建议列表，每条建议格式为： `棋步 (score: 分数, winrate: 胜率)`。\n"  # rank: 排名, 
                               # "请为 **列表中的前两个走法建议** 生成自然语言描述，说明建议的走法，"
                               # "**请直接输出** 列表中 **前两个走法建议** 的自然语言描述，无需任何前缀、解释或额外文字。"
                               "并简要概括其对应的局势评估和胜率信息。\n"
                               "每条象棋引擎的走法建议包含以下信息：\n"
                               "走法:  用自然语言描述建议的棋步，例如 '兵七进一' 描述为 '建议走法是进七兵'。\n"
                               "局势评估:  `score` 值，为当前局势的优劣 (正分表示优势，负分表示劣势，0分表示均势)。\n"
                               "胜率分析:   `winrate` 值，是红方的胜率。\n"
                               "直接用自然语言描述，**就像你在和朋友聊天一样**。  **不要使用任何类似 '走法:'，'局势评估:'，'胜率分析:' 这样的标签，也不要加任何前缀或解释性文字。** 直接说出自然流畅的句子即可。\n"
                               "请保持描述尽可能的简洁，信息准确。 \n"
                               # "例如，如果输入是 `['兵七进一 (score:1, rank:2, winrate:50.08)']`，  **请直接输出：**"
                               # "建议走法是兵七进一, 局势稍好, 红方胜率约五成"
                               # "就像你在自然地说话一样\n "
                               "不要使用任何数字、序号、列表的回答和类似 '走法:'，'局势评估:'，'胜率分析:' 这样的标签，也不要加任何前缀或解释性文字。直接说出自然流畅的句子即可。"
                               )

    def generate_response(self, user_prompt: str) -> str:
        """Generate response using the model with proper formatting."""
        # 构建输入文本
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": user_prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response


def gpt_35_api(client, messages: list):
    """为提供的对话消息创建新的回答

    Args:
        messages (list): 完整的对话消息
    """
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "developer", "content": "作为一个AI助手，你的任务是将象棋引擎提供的走法建议从棋盘语言转化为人类可读的自然语言。\n"
                                             "输入是走法建议列表，每条建议格式为： `棋步 (score: 分数, winrate: 胜率)`。\n"  # rank: 排名, 
            # "请为 **列表中的前两个走法建议** 生成自然语言描述，说明建议的走法，"
            # "**请直接输出** 列表中 **前两个走法建议** 的自然语言描述，无需任何前缀、解释或额外文字。"
                                             "并简要概括其对应的局势评估和胜率信息。\n"
                                             "每条象棋引擎的走法建议包含以下信息：\n"
                                             "走法:  用自然语言描述建议的棋步，例如 '兵七进一' 描述为 '建议走法是进七兵'。\n"
                                             "局势评估:  `score` 值，为当前局势的优劣 (正分表示优势，负分表示劣势，0分表示均势)。\n"
                                             "胜率分析:   `winrate` 值，是红方的胜率。\n"
                                             "直接用自然语言描述，**就像你在和朋友聊天一样**。  **不要使用任何类似 '走法:'，'局势评估:'，'胜率分析:' 这样的标签，也不要加任何前缀或解释性文字。** 直接说出自然流畅的句子即可。\n"
                                             "请保持描述尽可能的简洁，信息准确。 \n"
            # "例如，如果输入是 `['兵七进一 (score:1, rank:2, winrate:50.08)']`，  **请直接输出：**"
            # "建议走法是兵七进一, 局势稍好, 红方胜率约五成"
            # "就像你在自然地说话一样\n "
                                             "不要使用任何数字、序号、列表的回答和类似 '走法:'，'局势评估:'，'胜率分析:' 这样的标签，也不要加任何前缀或解释性文字。直接说出自然流畅的句子即可。"},
            {
                "role": "user",
                "content": messages
            }
        ]
    )

    # completion = client.chat.completions.create(model="gpt-4o-mini", messages=messages)        # gpt-3.5-turbo | gpt-4o-mini
    print(completion.choices[0].message.content)
    return completion


def siliconflow_API(system_prompt, test_text):
    url = "https://api.siliconflow.cn/v1/chat/completions"

    # 注意：开源时请让用户通过环境变量设置 Key，不要直接写死在代码里
    api_key = os.getenv("SILICONFLOW_API_KEY", "YOUR_SILICONFLOW_API_KEY_HERE")

    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": test_text
            }
        ],
        "stream": False,
        "max_tokens": 512,
        "stop": ["null"],
        "temperature": 0.7,
        "top_p": 0.7,
        "top_k": 50,
        "frequency_penalty": 0.5,
        "n": 1,
        "response_format": {"type": "text"},
        "tools": [
            {
                "type": "function",
                "function": {
                    "description": "<string>",
                    "name": "<string>",
                    "parameters": {},
                    "strict": False
                }
            }
        ]
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    response = requests.request("POST", url, json=payload, headers=headers)

    # print(response.text)
    content = ""
    if response.status_code == 200:  # 建议检查状态码是否成功
        response_json = json.loads(response.text)  # 将 JSON 字符串解析成 Python 字典
        if 'choices' in response_json and response_json['choices']:  # 确保 'choices' 存在且不为空
            content = response_json['choices'][0]['message']['content']  # 获取 content
            # print(content)  # 输出 content
        else:
            print("Response 中没有 'choices' 或 'choices' 为空。")
    else:
        print(f"请求失败，状态码：{response.status_code}")
        print(response.text)  # 打印错误信息，方便调试
    return content


def openai_API(system_prompt, test_text, model="gpt-3.5-turbo", image_base64_strings=None):
    # 注意：开源时请让用户通过环境变量设置 Key，不要直接写死在代码里
    # 推荐方式：export OPENAI_API_KEY="sk-..."
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY_HERE")
    )

    content = []
    if test_text:
        content.append({"type": "text", "text": test_text})
    if image_base64_strings:
        # 确保 image_base64_strings 是列表
        if isinstance(image_base64_strings, str):
            image_base64_strings = [image_base64_strings]

        for base64_string in image_base64_strings:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_string}"}
            })

    response = client.chat.completions.create(
        model=model,  # "gpt-3.5-turbo" | "gpt-4o"
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": content,  # 使用构建的 content 列表
            }
        ],
        max_tokens=50,  # 根据需要调整 150
        temperature=0,  # 根据需要调整 0.7
    )

    result = response.choices[0].message.content
    return result


if __name__ == '__main__':
    """
    建议环境：
    transformers >= 4.51.3
    torch
    openai
    requests
    """

    # 示例用法
    # client = OpenAI(
    #     api_key=os.getenv("OPENAI_API_KEY", "YOUR_API_KEY"),
    #     base_url="https://api.chatanywhere.tech/v1"
    # )

    test_text = ['兵七进一 (score:1, winrate:50.08)', '兵三进一 (score:1, winrate:50.08)']

    # 1. 调用 GPT-3.5/4o-mini (需要传入 client)
    # gpt_35_api(client,str(test_text[:2]))

    # 2. 调用 SiliconFlow (需要配置 API KEY)
    # result = siliconflow_API(
    #     "作为一个AI助手，你的任务是将象棋引擎提供的走法建议从棋盘语言转化为人类可读的自然语言。\n"
    #     "输入是走法建议列表，每条建议格式为： `棋步 (score: 分数, winrate: 胜率)`。\n"
    #     "并简要概括其对应的局势评估和胜率信息。\n"
    #     "每条象棋引擎的走法建议包含以下信息：\n"
    #     "走法:  用自然语言描述建议的棋步，例如 '兵七进一' 描述为 '推荐走法是兵七进一', '建议走法是进七兵'（因为兵只能进一格）,。\n"
    #     "局势评估:  `score` 值，为当前局势的优劣 (正分表示优势，负分表示劣势，0分表示均势)。\n"
    #     "胜率分析:   `winrate` 值，是红方的胜率。\n"
    #     "直接用自然语言描述，**就像你在和朋友聊天一样**。  **不要使用任何类似 '走法:'，'局势评估:'，'胜率分析:' 这样的标签，也不要加任何前缀或解释性文字。** 直接说出自然流畅的句子即可。\n"
    #     "请保持描述尽可能的简洁，信息准确。 \n"
    #     "不要使用任何数字、序号、列表的回答和类似 '走法:'，'局势评估:'，'胜率分析:' 这样的标签，也不要加任何前缀或解释性文字。直接说出自然流畅的句子即可。",
    #     str(test_text[:2])
    # )

    # 3. 调用 OpenAI API (需要配置环境变量 OPENAI_API_KEY)
    result = openai_API(
        "作为一个AI助手，你的任务是将象棋引擎提供的走法建议从棋盘语言转化为人类可读的自然语言。\n"
        "输入是走法建议列表，每条建议格式为： `棋步 (score: 分数, winrate: 胜率)`。\n"
        "并简要概括其对应的局势评估和胜率信息。\n"
        "每条象棋引擎的走法建议包含以下信息：\n"
        "走法:  用自然语言描述建议的棋步，例如 '兵七进一' 描述为 '推荐走法是兵七进一', '建议走法是进七兵'（因为兵只能进一格）,。\n"
        "局势评估:  `score` 值，为当前局势的优劣 (正分表示优势，负分表示劣势，0分表示均势)。\n"
        "胜率分析:   `winrate` 值，是红方的胜率。\n"
        "直接用自然语言描述，**就像你在和朋友聊天一样**。  **不要使用任何类似 '走法:'，'局势评估:'，'胜率分析:' 这样的标签，也不要加任何前缀或解释性文字。** 直接说出自然流畅的句子即可。\n"
        "请保持描述尽可能的简洁，信息准确。 \n"
        "不要使用任何数字、序号、列表的回答和类似 '走法:'，'局势评估:'，'胜率分析:' 这样的标签，也不要加任何前缀或解释性文字。直接说出自然流畅的句子即可。",
        str(test_text[:2])
    )

    print(result)
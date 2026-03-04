import os
import re
import ssl
import httpx
import torch
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util

# ======== 1. 全局配置与数据定义 ========

# 禁用 SSL 证书验证 (如需启用，请注释掉这两行)
# ssl._create_default_https_context = ssl._create_unverified_context
# client = httpx.Client(verify=False)

Chinesee_Chess_Agent_1 = [
    "正在分析棋局，计算最佳走法，请稍候。",
    # "指令已接收，象棋走法查询中...",
    "稍等，正在检索最优策略。",
    "正在处理您的请求，请稍等。",
    "分析中...请稍后片刻。",
    "好的，我明白了。 正在为您分析当前的中国象棋局面，并寻找最佳的下一步走法，请稍等片刻。",
    "收到您的请求。我正在努力计算，稍后会为您提供建议。",
    "请稍等，我正在深入分析当前的棋局，希望能给您最优的走法建议。",
    "非常荣幸能为您提供帮助。 我正在全力计算下一步，请耐心等待。",
    "感谢您的提问。 我正在评估所有可能的走法，请稍候。",
    "数据分析中... 正在计算最优解，请稍候。",
    "正在查询中国象棋最佳走法...",
    "中国象棋走法查询中，请稍候...",
    "象棋策略数据库查询中...",
    "正在从云端策略库获取最优解，请稍等...",
    "开始检索象棋专家级走法..."
]

# 定义意图模板
INTENT_TEMPLATES = {
    "object_recognition": [
        "请描述手指指向的物体是什么。",
        "请告诉我手指指向的物体的详细信息。",
        "请辨认手指所指的物体。",
        "这是什么。",
        "Please describe what the finger is pointing to.",
        "Please tell me the details of the object the finger is pointing to.",
        "Please identify the object the finger is pointing at.",
        "What is this?"
    ],
    "color_recognition": [
        "请描述手指指向物体的颜色。",
        "请告诉我手指指向物体的颜色是什么。",
        "请辨认手指所指物体的颜色。",
        "这是什么颜色。",
        "Please describe the color of the object the finger is pointing to.",
        "Please tell me the color of the object the finger is pointing at.",
        "Please identify the color of the object the finger is pointing to.",
        "What color is this?"
    ],
    "scene_description": [
        "请描述当前视野中的内容。",
        "请告诉我你看到的场景是什么。",
        "请分析当前视野中的环境。",
        "Please describe the contents of the current view.",
        "Please tell me what the scene you see is.",
        "Please analyze the environment in your current view.",
        "What can you see in this image?"
    ],
    "feature_explanation": [
        "请说明手指指向的功能如何工作。",
        "请解释手指所指功能的详细信息。",
        "请描述手指指向功能的操作方式。",
        "Please explain how the function the finger is pointing to works.",
        "Please provide detailed information about the function the finger is pointing to.",
        "Please describe how the function the finger is pointing to operates."
    ],
    "image_analysis": [
        "请分析当前视野中的图片内容。",
        "请描述当前图像中的详细信息。",
        "请辨认当前图片中的主要元素。",
        "Please analyze the content of the image in your current view.",
        "Please describe the details of the current image.",
        "Please identify the main elements in the current image."
    ],
    "play_Chinesee_chess": [
        # "下一步我应该走什么？",
        # "我下一步该怎么走?",
        # "请建议我下一步的走法。",
        # "帮我看看，现在走哪一步比较好？",
        # "最佳的下一步是什么？",
        # "你推荐我走哪一步？",
        # "你有什么建议吗？",
        # "下一步怎么走比较好？",
        # "给出前三步的最佳走法.",
        # "主要的策略是什么？",
        # "你推荐的走法是什么?"
        "What should be my next move?",
        "How should I move next?",
        "Please suggest my next move.",
        "Help me see, which move is better now?",
        "What is the best next move?",
        "Which move do you recommend?",
        "Do you have any suggestions?",
        "How should I move next for a better position?",
        "Give the best three next moves.",
        "What is the main strategy?",
        "What move do you recommend?"
    ],
    "Memorization_required": [
        "...在哪，我忘记了。",
        "我忘记了。",
        "我刚刚把杯子放在哪里了？",
        "我的手机放在哪里了？",
        "钥匙在哪里？",
        "笔放哪里了？",
        "测试一下记忆功能。",
        "请检索记忆...",
        "我刚刚...放在哪里",
        "我的...放在哪里了",
        "帮我回忆一下...",
        "我忘记……告诉我在哪里",
        "我的象棋棋盘放在哪里了",
        "我的绿色的水杯放在哪里了",
        "我的红色的笔放在哪里了",
        # 我现在想玩一个游戏，让LLM看一眼，然后把中国象棋的棋子倒扣住，移动几轮，然后向大语言模型提问，可能的问法有哪些？考察LLM的识别、追踪和记忆能力，比如炮在哪里，红色的相在哪里...
        # "这个位置是什么棋子？",
        # "红色的...在什么位置？",
        # "黑色的...在什么位置？",
        # "马在哪个位置？"
    ],
    "high_frame_rate_mode": [
        "进入高帧率模式。",
        "进入游戏模式。",
        "记住这三个棋子的位置。",
        "记住棋子的位置。",
        "记住当前中国象棋棋子的位置",
    ],
    "low_frame_rate_mode_withinference": [
        "这个位置是什么棋子？",
        "红色的...在什么位置？",
        "黑色的...在什么位置？",
        "...在哪个位置？",
        "...位置的棋子分别是什么？",
        "现在三个棋子位置是什么",
        "从左到右棋子分别是什么",
        "从左到右是什么",
        "这是红/黑色的車/炮吗",
        "这个棋子是什么颜色的？",
        "这个是红象吗?",
        "这个是黑马吗？",
        "这个是黑象吗?"
    ],
    "low_frame_rate_mode_woinference": [
        # "退出高帧率模式。",
        # "退出游戏模式。",
        # "你记住棋子位置了吗",
        # "我移动好了"
        "Exit high frame rate mode.",
        "Exit game mode.",
        "Have you remembered the piece positions?",
        "I've finished moving.",
        "I have finished moving."
    ],
    "continuous_guidance_mode": [
        "进入连续指导模式。",
        "连续下棋指导模式。",
        "请一直指导我下棋。",
        "你可以教我下象棋吗？"
    ],
    "cancel_continuous_guidance_mode": [
        "退出连续指导模式。",
        "不用连续下棋了。",
        "不用指导我下棋了。",
        "不用教我下棋了。",
    ]
}

WAITING_PHRASES = [
    "有什么我可以帮忙的吗？",
    "有任何问题我可以解答吗？",
    "需要我帮忙找点资料吗？",
    "有什么我能协助您的地方吗？",
    "如果需要帮助，随时告诉我哦！",
    "我可以帮你解决其他问题吗？",
    "你还需要我帮你做什么吗？",
    "你还好吗？需要休息一下吗？",
    "我还在这里，如果你需要我，随时叫我。",
    "需要我继续帮你做些什么吗？",
    "你累了吗？如果需要暂停可以告诉我哦。",
    "如果你需要放松一下，我也可以随时陪聊！",
    "在思考吗？如果你准备好了，我们可以继续！",
    "嘿，是否还在考虑问题？如果有需要帮忙的地方，告诉我！",
    "看起来你停下来了，有什么我可以帮忙的吗？",
    "如果需要一点帮助，随时告诉我哦！",
    "如果你想换个方式试试，我可以为你提供一些建议。",
    "你可以随时调整方向，我在这里帮助你。",
    "我也可以为你提供其他选择哦！",
    "你已经做得很棒了！继续加油！",
    "只差一点点就完成啦，你还需要帮助吗？",
    "需要我为你检查一下吗？"
]

WAITING_PHRASES_en = [
    "Is there anything I can help you with?",
    "Are there any questions I can answer for you?",
    "Do you need help finding some information?",
    "Is there anything I can assist you with?",
    "If you need help, just let me know anytime!",
    "Can I help you with anything else?",
    "Is there anything else I can do for you?",
    "Are you okay? Do you need a break?",
    "I'm still here, feel free to call me if you need anything.",
    "Do you need me to continue helping with something?",
    "Are you tired? Let me know if you need a pause.",
    "If you need to relax, I can keep you company anytime!",
    "Thinking? When you're ready, we can continue!",
    "Hey, are you still thinking it over? Let me know if you need help!",
    "It seems like you've paused, is there anything I can help with?",
    "Feel free to reach out if you need a little help!",
    "If you'd like to try a different approach, I can offer some suggestions.",
    "You can always change direction, I'm here to help you.",
    "I can also provide other options for you!",
    "You're doing great! Keep it up!",
    "You're almost there! Do you need any help?",
    "Would you like me to double-check something for you?"
]

# ======== 规则基部分，维护关键词和模式 ========
HELP_REQUEST_PATTERNS = {
    "phrases": [
        "你知道",
        "你能告诉我",
        "请问",
        "我想知道",
        "这是不是",
        "帮我查一下",
        "请解释",
        "请告诉我",
        "能否说明",
        "可否告知",
        "请帮忙",
        "需要帮助",
        "帮我",
        "协助",
        "指导",
        "帮忙",
        "询问",
        "咨询",
        "了解",
        "说明",
        "探讨",
        "请回答",
        "请提供",
        "请阐述",
        "请描述",
        "请简单介绍",
        "请介绍",
        "介绍",

        "Tell me",
        "Do you know",
        "Can you tell me",
        "May I ask",
        "I want to know",
        "Is this",
        "Help me look up",
        "Please explain",
        "Please tell me",
        "Can you clarify",
        "Can you inform me",
        "Please assist",
        "Need help",
        "Help me",
        "Assist",
        "Guide",
        "Help",
        "Inquire",
        "Consult",
        "Understand",
        "Clarify",
        "Discuss",
        "Please answer",
        "Please provide",
        "Please elaborate",
        "Please describe",
        "Please give a brief introduction",
        "Please introduce",
        "Introduce",
    ],
    "question_words": [
        "什么",
        "谁",
        "为什么",
        "如何",
        "哪里",
        "哪",
        "多少",
        "几",
        "怎么",
        "哪儿",
        "哪一个",
        "哪种",
        "哪方面",
        "哪个",
        "哪一种",
        "哪几",
        "哪几种",
        "哪几方面",

        "What",
        "Who",
        "Why",
        "How",
        "Where",
        "Which",
        "How many",
        "How much",
        "How",
        "Where",
        "Which one",
        "Which type",
        "Which aspect",
        "Which one",
        "Which type",
        "How many",
        "How many types",
        "How many aspects",
    ],
    "ends_with_question_mark": True  # 判断是否以问号结尾
}

# 定义需要视觉信息的关键词和模式
NEED_IMAGE_PATTERNS = {
    "phrases": [
        "这是什么",
        "这是什么颜色的",
        "这里有什么",
        "指着这个",
        "手指的",
        "指向",
        "看看这个是什么",
        "你看到什么",
        "看到这个东西",
        "识别一下这个物体",
        "辨认一下这个颜色",
        "看一下这个",
        "展示这个",
        "描述这个",
        "拍摄这个",
        "识别这个",
        "辨认这个",
        "指向那个",
        "指向这里",
        "指着那边",
        "看那边",
        "查看这个",
        "分析这个",
        "检查这个",
        "识别一下这里的",
        "辨认一下那里",
        "请帮忙查看这",
        "请帮忙分析这",
        "请帮忙检查这",
        "请帮忙识别这张图片",
        "请帮忙辨认这里的标志",
        "帮我看看这是什么东西",
        "请帮我辨认一下这张照片",
        "你能识别这是什么颜色吗？",
        "请帮忙描述这张图片",
        "我需要知道这是什么",
        "请帮忙分析这里的内容",
        "能否告诉我这是什么？",
        "帮我识别这里的标志",
        "这个棋子是什么",
        "这款",
        # "这个",
        "这是",
        "这里",
        "那",
        "那里",
        "那些",
        "这些",
        "它",
        "他",
        "她",
        "他们",
        "她们",
        "它们",
        # "我们",
        # "你们",
        # "你",
        # "我"

        "What is this",
        "What color is this",
        "What's here",
        "Pointing at this",
        "Pointing to this",
        "Pointing at",
        "Look at what this is",
        "What do you see",
        "See this object",
        "Identify this object",
        "Identify this color",
        "Look at this",
        "Show this",
        "Describe this",
        "Take a picture of this",
        "Identify this",
        "Recognize this",
        "Point to that",
        "Point here",
        "Point over there",
        "Look over there",
        "Check this out",
        "Analyze this",
        "Examine this",
        "Identify what's here",
        "Recognize what's there",
        "Please help check this",
        "Please help analyze this",
        "Please help examine this",
        "Please help identify this image",
        "Please help recognize this sign",
        "Help me see what this is",
        "Please help recognize this photo",
        "Can you identify what color this is?",
        "Please help describe this image",
        "I need to know what this is",
        "Please help analyze what's here",
        "Can you tell me what this is?",
        "Help me identify this sign here",
        "This one",
        # "This",
        "This is",
        "Here",
        "That",
        "There",
        "Those",
        "These",
        "It",
        "He",
        "She",
        "They",
        "They (female)",
        "They (neutral)",
        # "We",
        # "You all",
        # "You",
        # "I"

    ],
    "additional_patterns": [
        r"指着\w+，?\s*这是什么\？",
        r"[这那].*?什么",
        r"[这那].*?吗",
        r"看看\w+是什么\。",
        r"你看到什么\？",
        r"识别一下\w+物体\。",
        r"辨认一下\w+颜色\。",
        r"请帮忙识别\w+图片\。",
        r"请帮忙辨认\w+标志\。",
        r"帮我辨认一下这张照片\。",
        r"你能识别这是什么颜色吗\？",
        r"请帮忙描述这张图片\。",
        r"我需要知道这是什么\。",
        r"请帮忙分析这里的内容\。",
        r"能否告诉我这是什么\？",
        r"帮我识别这里的标志\。",

        r"Pointing at \w+,?\s*what is this\?",
        r"[This That].*?what",
        r"[This That].*?is it",
        r"Look at \w+ and see what it is.",
        r"What do you see?",
        r"Identify \w+ object.",
        r"Identify \w+ color.",
        r"Please help identify \w+ image.",
        r"Please help recognize \w+ sign.",
        r"Help me recognize this photo.",
        r"Can you identify what color this is?",
        r"Please help describe this image.",
        r"I need to know what this is.",
        r"Please help analyze what's here.",
        r"Can you tell me what this is?",
        r"Help me identify this sign here.",
    ]
}

# ======== 语义相似度部分，维护关键句子 ========
SEMANTIC_HELP_REQUEST_SENTENCES = [
    "你知道...吗？",
    "你能告诉我...吗？",
    "请问...",
    "我想知道...",
    "这是不是...",
    "帮我查一下...",
    "请解释...",
    "请告诉我...",
    "能否说明...",
    "可否告知...",
    "请帮忙...",
    "需要帮助...",
    "帮我...",
    "协助我...",
    "指导我...",
    "帮忙...",
    "询问...",
    "咨询...",
    "了解...",
    "说明...",
    "探讨...",
    "请回答...",
    "请提供...",
    "请阐述...",
    "请描述...",
    "...是不是...",
    "...有多长",
    "今天天气怎么样？",
    "你可以听见我讲话吗？",

    "Do you know...?",
    "Can you tell me...?",
    "May I ask...?",
    "I want to know...?",
    "Is this...?",
    "Help me look up...?",
    "Please explain...?",
    "Please tell me...?",
    "Can you clarify...?",
    "Can you inform me...?",
    "Please assist...?",
    "Need help...?",
    "Help me...?",
    "Assist me...?",
    "Guide me...?",
    "Help with...?",
    "Inquire about...?",
    "Consult about...?",
    "Understand...?",
    "Clarify...?",
    "Discuss...?",
    "Please answer...?",
    "Please provide...?",
    "Please elaborate...?",
    "Please describe...?",
]

SEMANTIC_NEED_IMAGE_SENTENCES = [
    "这是什么？",
    "这是什么颜色的？",
    "这里有什么？",
    "指着这个，这是什么？",
    "看看这个是什么。",
    "你看到什么？",
    "识别一下这个物体。",
    "辨认一下这个颜色。",
    "看一下这个。",
    "展示这个。",
    "描述这个。",
    "拍摄这个。",
    "识别这个。",
    "辨认这个。",
    "指向那个。",
    "指向这里。",
    "指着那边。",
    "看那边。",
    "请帮忙识别这张图片。",
    "请帮忙辨认这里的标志。",
    "帮我看看这是什么东西。",
    "请帮我辨认一下这张照片。",
    "你能识别这是什么颜色吗？",
    "请帮忙描述这张图片。",
    "我需要知道这是什么。",
    "请帮忙分析这里的内容。",
    "能否告诉我这是什么？",
    "帮我识别这里的标志。",
    "能否说明这个功能如何工作？",
    "我手指指的东西是什么",
    "你再好好看看我指的是什么数字",
    "我指的是什么数字",

    "What is this?",
    "What color is this?",
    "What's here?",
    "Pointing at this, what is it?",
    "Look at what this is.",
    "What do you see?",
    "Identify this object.",
    "Recognize this color.",
    "Look at this.",
    "Show this.",
    "Describe this.",
    "Take a picture of this.",
    "Identify this.",
    "Recognize this.",
    "Point to that.",
    "Point here.",
    "Point over there.",
    "Look over there.",
    "Please help identify this image.",
    "Please help recognize this sign.",
    "Help me see what this is.",
    "Please help recognize this photo.",
    "Can you identify what color this is?",
    "Please help describe this image.",
    "I need to know what this is.",
    "Please help analyze what's here.",
    "Can you tell me what this is?",
    "Help me identify this sign here.",
    "Can you explain how this function works?",
    "What is the object I am pointing at?",
    "Can you look closely at the number I am pointing to?",
    "What number am I pointing at?",
]

# ======== 修改后的 system_message，结合规则基和LLM ========
system_message = (
    "你是一个智能眼镜的AI助手。请完成以下两项判断，并严格遵守回答格式：\n"
    "1) 判断用户是否需要帮助（是否包含问题、请求或需要协助的意图）。\n"
    "   - 输出一个0到1之间的概率值，表示需要帮助的概率。\n"
    "2) 如果用户需要帮助，判断回答用户的输入是否需要依赖当前的视觉信息。\n"
    "   - 输出一个0到1之间的概率值，表示需要视觉信息的概率。\n\n"
    "以下是一些示例（先判断是否需要帮助，再决定是否需要视觉信息）：\n"
    "--------------------------------------------------\n"
    "用户输入：\"谢谢\" \n"
    "help_request: 0.1\n"
    "need_image: 0.0\n"
    "--------------------------------------------------\n"
    "用户输入：\"这个东西是什么呀\" \n"
    "help_request: 0.95\n"
    "need_image: 0.9\n"
    "--------------------------------------------------\n"
    "用户输入：\"打开空调\"\n"
    "help_request: 0.8\n"
    "need_image: 0.2\n"
    "--------------------------------------------------\n"
    "用户输入：\"请帮忙识别这张图片。\"\n"
    "help_request: 0.9\n"
    "need_image: 0.95\n"
    "--------------------------------------------------\n"
    "请你根据以上规则和示例，对用户的输入进行判断，然后只输出两行结果：\n"
    "help_request: [0.0-1.0]\n"
    "need_image: [0.0-1.0]\n"
)

system_message_en = (
    "You are an AI assistant for smart glasses. Please perform the following two tasks and strictly follow the response format:\n"
    "1) Determine if the user needs help (whether the input contains a question, request, or intention to seek assistance).\n"
    "   - Output a probability value between 0 and 1, indicating the likelihood of needing help.\n"
    "2) If the user needs help, determine whether the response requires current visual information.\n"
    "   - Output a probability value between 0 and 1, indicating the likelihood of needing visual information.\n\n"
    "Here are some examples (first determine if help is needed, then decide if visual information is required):\n"
    "--------------------------------------------------\n"
    "User Input: \"Thank you\" \n"
    "help_request: 0.1\n"
    "need_image: 0.0\n"
    "--------------------------------------------------\n"
    "User Input: \"What is this thing?\" \n"
    "help_request: 0.95\n"
    "need_image: 0.9\n"
    "--------------------------------------------------\n"
    "User Input: \"Turn on the air conditioner\" \n"
    "help_request: 0.8\n"
    "need_image: 0.2\n"
    "--------------------------------------------------\n"
    "User Input: \"Please help identify this image.\" \n"
    "help_request: 0.9\n"
    "need_image: 0.95\n"
    "--------------------------------------------------\n"
    "Please follow the rules and examples above to judge the user's input, and then output only the following two lines:\n"
    "help_request: [0.0-1.0]\n"
    "need_image: [0.0-1.0]\n"
)


# ======== 2. 核心类定义 ========

class LLMInferenceUnified:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct-AWQ"):  # Qwen/Qwen2.5-0.5B-Instruct-AWQ
        """
        Initialize the LLM inference class with model and tokenizer.
        Args:
            model_name (str): The model identifier to load
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            # attn_implementation="flash_attention_2",
        ).to(self.device)

        # 初始化句子嵌入模型
        self.embedding_model = SentenceTransformer(
            'paraphrase-multilingual-MiniLM-L12-v2')  # 'distiluse-base-multilingual-cased-v1' | paraphrase-multilingual-MiniLM-L12-v2 | paraphrase-MiniLM-L3-v2

        # 计算并存储每个意图类别下的模板嵌入
        self.INTENT_EMBEDDINGS = {}
        for intent, templates in INTENT_TEMPLATES.items():
            embeddings = self.embedding_model.encode(templates, convert_to_tensor=True)
            self.INTENT_EMBEDDINGS[intent] = embeddings

    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate response using the model with proper formatting."""
        # 构建输入文本
        text = ""
        for message in messages:
            if message['role'] == 'system':
                text += f"System: {message['content']}\n"
            elif message['role'] == 'user':
                text += f"User: {message['content']}\n"
            elif message['role'] == 'assistant':
                text += f"Assistant: {message['content']}\n"

        # text += "Assistant: "  # 确保模型知道要生成助手的回应

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 编码输入
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        # 生成响应
        # generated_ids = self.model.generate(
        #     **model_inputs,
        #     max_new_tokens=50,  # 限制生成长度
        #     do_sample=False,
        #     eos_token_id=self.tokenizer.eos_token_id
        # )
        # 提取生成的部分
        # generated_ids = [
        #     output_ids[len(input_ids):]
        #     for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        # ]
        # # 解码响应
        # response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=50
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response.strip()

    def generate_response_new(self, system_messages, prompt_messages):
        # NOTE: This function relies on a local file 'realtime_demo_5_AI.py'.
        # Ensure this file exists in your path or implement your own API handler.
        try:
            from api_providers import siliconflow_API
            result = siliconflow_API(system_messages, prompt_messages)
            return result
        except ImportError:
            print("Warning: 'realtime_demo_5_AI' module not found. Functionality unavailable.")
            return None

    def preprocess_text(self, text: str) -> str:
        """
        预处理输入文本，移除多余的标点符号，并标准化空格。
        Args:
            text (str): 原始输入文本
        Returns:
            str: 预处理后的文本
        """
        # 移除多余的空格
        text = re.sub(r'\s+', '', text)
        # 标准化标点符号（将中文标点转换为英文标点）
        text = text.replace('？', '？').replace('。', '。').replace('！', '！').replace('，', '，')
        return text

    def match_patterns(self, text_input: str) -> Dict[str, bool]:
        """
        使用规则基匹配输入，判断是否需要帮助和是否需要视觉信息。
        Args:
            text_input (str): 用户输入的文本
        Returns:
            Dict[str, bool]:
            {
                'rule_based_help_request': True/False,
                'rule_based_need_image': True/False
            }
        """
        rule_based_need_image = False
        rule_based_help_request = False

        # 预处理输入文本
        processed_text = self.preprocess_text(text_input)

        # 检查需要视觉信息的关键词 | 是否包含指代词并结合相关关键词
        for phrase in NEED_IMAGE_PATTERNS["phrases"]:
            if phrase in processed_text:
                rule_based_need_image = True
                break

        # 检查额外的正则模式
        if not rule_based_need_image and NEED_IMAGE_PATTERNS.get("additional_patterns"):
            for pattern in NEED_IMAGE_PATTERNS["additional_patterns"]:
                if re.search(pattern, processed_text):
                    rule_based_need_image = True
                    break

        # 如果需要视觉信息，则帮助请求必定为 True
        if rule_based_need_image:
            rule_based_help_request = True
        else:
            # 检查帮助请求的关键词
            for phrase in HELP_REQUEST_PATTERNS["phrases"]:
                if phrase in processed_text:
                    rule_based_help_request = True
                    break

            # 检查疑问词
            if not rule_based_help_request:
                for q_word in HELP_REQUEST_PATTERNS["question_words"]:
                    if q_word in processed_text:
                        rule_based_help_request = True
                        break

            # 检查是否以问号结尾
            if not rule_based_help_request and HELP_REQUEST_PATTERNS["ends_with_question_mark"]:
                if processed_text.endswith("？"):
                    rule_based_help_request = True

        return {
            "rule_based_need_image": rule_based_need_image,
            "rule_based_help_request": rule_based_help_request
        }

    def semantic_similarity_match(self, text_input: str) -> Dict[str, float]:
        """
        使用语义相似度匹配输入，计算帮助请求和需要视觉信息的概率。
        Args:
            text_input (str): 用户输入的文本
        Returns:
            Dict[str, float]:
            {
                'semantic_help_request_prob': float,  # 需要帮助的概率
                'semantic_need_image_prob': float     # 需要视觉信息的概率
            }
        """
        # 计算输入句子的嵌入
        input_embedding = self.embedding_model.encode(text_input, convert_to_tensor=True)

        # 计算帮助请求的相似度
        help_embeddings = self.embedding_model.encode(SEMANTIC_HELP_REQUEST_SENTENCES, convert_to_tensor=True)
        help_similarities = util.pytorch_cos_sim(input_embedding, help_embeddings)[0]
        semantic_help_request_prob = float(torch.max(help_similarities))  # 取最高相似度

        # 计算需要视觉信息的相似度
        image_embeddings = self.embedding_model.encode(SEMANTIC_NEED_IMAGE_SENTENCES, convert_to_tensor=True)
        image_similarities = util.pytorch_cos_sim(input_embedding, image_embeddings)[0]
        semantic_need_image_prob = float(torch.max(image_similarities))  # 取最高相似度

        return {
            "semantic_help_request_prob": semantic_help_request_prob,
            "semantic_need_image_prob": semantic_need_image_prob
        }

    def analyze_input(self, text_input: str, help_threshold: float = 0.6, image_threshold: float = 0.6,
                      similarity_threshold: float = 0.7, language='zh') -> Dict[str, bool]:
        """
        结合规则基和LLM对用户输入进行判断。
        Args:
            text_input (str): 用户输入的文本
            help_threshold (float): 判断是否需要帮助的阈值
            image_threshold (float): 判断是否需要视觉信息的阈值
            similarity_threshold (float): 语义相似度阈值
        Returns:
            Dict[str, bool]:
            {
                'help_request': True/False,  # 是否在寻求帮助
                'need_image': True/False     # 是否需要视觉信息
            }
        """
        # 规则基判断
        rules = self.match_patterns(text_input)

        # 初始化最终结果
        final_help_request = False
        final_need_image = False

        # 首先判断是否需要视觉信息
        if rules["rule_based_need_image"]:
            final_need_image = True
            final_help_request = True
        else:
            # 使用语义相似度匹配判断是否需要视觉信息
            semantic_probs = self.semantic_similarity_match(text_input)
            if semantic_probs["semantic_need_image_prob"] >= similarity_threshold:
                final_need_image = True
                final_help_request = True
            else:
                # 如果语义相似度未达到阈值，使用LLM进行判断

                if language == 'zh':
                    user_prompt = (
                        f"请分析下面这句话，根据上述要求：\n"
                        f"1) 是否在寻求帮助？\n"
                        f"2) 若需要帮助，是否需要视觉信息？\n\n"
                        f"用户输入：{text_input}\n\n"
                        f"请仅输出下面两行（不要添加额外说明或解释）：\n"
                        f"help_request: [0.0-1.0]\n"
                        f"need_image: [0.0-1.0]"
                    )
                    messages = [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_prompt}
                    ]
                elif language == 'en':
                    user_prompt_en = (
                        f"Please analyze the following sentence based on the criteria above:\n"
                        f"1) Is the user seeking help?\n"
                        f"2) If help is needed, is visual information required?\n\n"
                        f"User input: {text_input}\n\n"
                        f"Please output only the following two lines (without any additional explanations or remarks):\n"
                        f"help_request: [0.0-1.0]\n"
                        f"need_image: [0.0-1.0]"
                    )
                    messages = [
                        {"role": "system", "content": system_message_en},
                        {"role": "user", "content": user_prompt_en}
                    ]

                # 得到模型的合并回答
                response = self.generate_response(messages).strip()
                # response = self.generate_response_new(system_message, user_prompt).strip()
                # 模型应当只输出两行：
                # help_request: 0.x
                # need_image: 0.y
                lines = response.split("\n")

                # 做个简单稳健性的处理
                help_request_line = lines[0] if len(lines) > 0 else "help_request: 0.0"
                need_image_line = lines[1] if len(lines) > 1 else "need_image: 0.0"

                try:
                    # 提取概率值
                    help_request_prob = float(help_request_line.split(":")[-1].strip())
                except ValueError:
                    help_request_prob = 0.0  # 默认值

                try:
                    need_image_prob = float(need_image_line.split(":")[-1].strip())
                except ValueError:
                    need_image_prob = 0.0  # 默认值

                # 根据阈值判断
                # final_help_request = help_request_prob >= help_threshold
                final_need_image = need_image_prob >= image_threshold  # TODO: if final_help_request else False

                if final_need_image == False:

                    if rules['rule_based_help_request'] or semantic_probs[
                        "semantic_help_request_prob"] >= similarity_threshold or help_request_prob >= help_threshold:
                        final_help_request = True
                    else:
                        final_help_request = False

                else:
                    final_need_image = True
                    final_help_request = True

        try:
            print('need_image: ', final_need_image,
                  "rule_based: ", rules["rule_based_need_image"],
                  "semantic_similarity_based: ",
                  f'{round(semantic_probs["semantic_need_image_prob"], 2)}/{similarity_threshold}',
                  "LLM_prob_based:", f'{need_image_prob}/{image_threshold}')
            print('help_request: ', final_help_request,
                  "rule_based: ", rules['rule_based_help_request'],
                  "semantic_similarity_based: ",
                  f'{round(semantic_probs["semantic_help_request_prob"], 2)}/{similarity_threshold}',
                  "LLM_prob_based:", f'{help_request_prob}/{help_threshold}')
        except:
            print('need_image: ', final_need_image)
            print('help_request: ', final_help_request)

        return {
            "help_request": final_help_request,
            "need_image": final_need_image
        }


# ======== 3. 示例与测试代码 (API Key 匿名化) ========

"""
import openai
from openai import OpenAI
import requests
from requests.adapters import HTTPAdapter
from urllib3.poolmanager import PoolManager

# 注意：开源版本请使用环境变量或安全方式管理 API Key
# Set OpenAI's API key and API base to use vLLM's API server.
# openai_api_key = os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY_HERE")
# openai_api_base = "http://localhost:8000/v1"
#
# client = OpenAI(
#     api_key=openai_api_key,
#     base_url=openai_api_base,
# )

class SSLAdapter(HTTPAdapter):
    def init_poolmanager(self, connections, maxsize, block=False):
        self.poolmanager = PoolManager(
            num_pools=connections,
            maxsize=maxsize,
            block=block,
            ssl_version=ssl.PROTOCOL_TLS,
            cert_reqs=ssl.CERT_NONE  # 禁用证书验证
        )

# client = OpenAI(
#     # If the environment variable is not configured, replace the following line with: api_key="sk-xxx",
#     api_key=os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY_HERE"),
#     base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
# )

def text_based_inference(text):
    user_prompt = (
        f"请分析下面这句话，根据上述要求，"
        f"先判断是否在寻求帮助，然后判断是否需要视觉信息。"
        f"\\n\\n{text}\\n"
        f"请仅输出如下两行，不要添加其他内容：\\n"
        f"help_request: 是/否\\n"
        f"need_image: 是/否"
    )

    # chat_response = client.chat.completions.create(
    #     model="Qwen/Qwen2.5-7B-Instruct",
    #     messages=[
    #         {"role": "system", "content": system_message},
    #         {"role": "user", "content": user_prompt},
    #     ],
    #     max_tokens=50,  # 限制生成长度
    #     temperature=0.1,
    # )

    session = requests.Session()
    session.mount('https://', SSLAdapter())

    # 注意：此处原代码包含硬编码的 API Key，已替换为占位符
    openai.api_key = os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY_HERE")
    openai.api_base = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

    # 使用自定义会话进行请求
    openai.requestor._REQUESTS_SESSION = session

    completion = client.chat.completions.create(
        model="qwen2.5-7b-instruct",
        messages=[
            {'role': 'system', 'content': system_message},
            {'role': 'user', 'content': user_prompt}
            ]
    )
    chat_response = completion.choices[0].message.content

    # 得到模型的合并回答
    response = chat_response.strip()

    # 模型应当只输出两行：
    # help_request: 是/否
    # need_image: 是/否
    lines = response.split("\\n")
    # 做个简单稳健性的处理
    help_request_line = lines[0] if len(lines) > 0 else ""
    need_image_line = lines[1] if len(lines) > 1 else ""

    # 判断模型输出
    help_request = help_request_line.endswith("是")
    need_image = need_image_line.endswith("是")

    return {
        "help_request": help_request,
        "need_image": need_image
    }


# 使用示例
if __name__ == "__main__":
    # llm = LLMInferenceUnified()

    test_cases = [
        "我手指的地方是哪里",
        "这个东西是什么呀",
        "这个方法还是挺好的",
        "先一些提高",
        "Mwah!",
        "Ugh!",
        "这里写的是什么内容",
        "谢谢你的帮助",
        "这太美了",
        "真不错啊"
    ]

    for text in test_cases:
        # result = llm.analyze_input(text)
        # 注意：text_based_inference 需要配置有效的 API Key 才能运行
        # result = text_based_inference(text)
        # print(f"\\n输入: {text}")
        # print(f"help_request: {result['help_request']}")
        # print(f"need_image: {result['need_image']}")
        pass

"""
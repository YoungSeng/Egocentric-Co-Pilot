import json
import os
import re
import datetime
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from collections import defaultdict
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEFAULT_LLM_MODEL_PATH, DEFAULT_OUTPUT_DIR

# --- Configuration ---
MAX_CAPTIONS_DIRECT = 300
KEEP_RECENT_CAPTIONS_COUNT = 100
MAX_TOKENS_FOR_SUMMARY = 10000


def parse_args():
    parser = argparse.ArgumentParser(description="Run Evaluation (Refine Strategy)")
    parser.add_argument("--results_file", type=str, required=True, help="Input results JSON file")
    parser.add_argument("--captions_file", type=str, required=True, help="Input captions JSON file")
    parser.add_argument("--llm_path", type=str, default=DEFAULT_LLM_MODEL_PATH)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


# --- Helper Functions ---

def parse_caption_key(key):
    match = re.match(r'DAY(\d+)_.*_(\d{8})\.mp4', key)
    if match:
        day_num = int(match.group(1))
        time_str = match.group(2)
        return (day_num, time_str)
    return None


def load_and_sort_captions(filepath):
    print(f"[*] Loading captions from {filepath}...")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            captions_data = json.load(f)
    except Exception as e:
        print(f"Error loading captions: {e}")
        return None

    parsed_captions = []
    for key, text in captions_data.items():
        parsed_key = parse_caption_key(key)
        if parsed_key:
            parsed_captions.append((parsed_key[0], parsed_key[1], text))

    parsed_captions.sort()
    print(f"[*] Loaded {len(parsed_captions)} captions.")
    return parsed_captions


def load_llm_model(model_path):
    print(f"--- Loading LLM: {model_path} ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation="flash_attention_2"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return model, tokenizer
    except Exception as e:
        print(f"Failed to load LLM: {e}")
        raise


# --- Summarization Logic ---

def get_summary_from_llm(model, tokenizer, captions_for_summary, day, hour):
    captions_text = "\n".join([f"- {c}" for _, _, c in captions_for_summary])
    summary_prompt = f"""
You are an AI assistant. Your task is to read a list of events that occurred within a single hour from a first-person perspective and write a concise, one-paragraph summary.

**Events from Day {day}, hour {hour:02d}:00:**
{captions_text}

**Summary:**
"""
    messages = [{"role": "user", "content": summary_prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs, max_new_tokens=MAX_TOKENS_FOR_SUMMARY, do_sample=False, temperature=0.0
    )
    generated_ids = [out[len(inp):] for inp, out in zip(model_inputs.input_ids, generated_ids)]
    response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return f"Summary for Day {day}, hour {hour:02d}:00: {response_text}"


def summarize_captions_by_hour(model, tokenizer, captions_to_summarize):
    print(f"[*] Summarizing {len(captions_to_summarize)} captions...")
    hourly_groups = defaultdict(list)
    for day, time_str, text in captions_to_summarize:
        hour = int(time_str[:2])
        hourly_groups[(day, hour)].append((day, time_str, text))

    hourly_summaries = []
    for (day, hour), captions in tqdm(sorted(hourly_groups.items()), desc="Summarizing"):
        summary = get_summary_from_llm(model, tokenizer, captions, day, hour)
        hourly_summaries.append(summary)
    return hourly_summaries


# --- Inference Logic ---

def get_answer_from_llm(model, tokenizer, question_data, context_captions, context_is_summarized=False):
    context_str = "\n".join(f"- {caption}" for caption in context_captions)
    if not context_str:
        context_str = "No relevant events found."

    context_desc = "The context below contains hourly summaries of older events, followed by a detailed chronological list of the most recent events." if context_is_summarized else "The context below is a chronological list of events."

    system_prompt = (
        "You are an expert AI assistant analyzing events from a first-person (egocentric) video feed. "
        "Your task is to answer a multiple-choice question based ONLY on the provided context. "
        f"{context_desc} "
        "Carefully read the context and the question, then choose the best answer from the given options."
    )

    user_prompt = f"""
**CONTEXT:**
{context_str}

**QUESTION:**
{question_data['question']}

**OPTIONS:**
A. {question_data['choice_a']}
B. {question_data['choice_b']}
C. {question_data['choice_c']}
D. {question_data['choice_d']}

Based strictly on the context provided, which option is the correct answer?
Your response must be a single capital letter: A, B, C, or D.
"""
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(**model_inputs, max_new_tokens=10, do_sample=False, temperature=0.0)
        generated_ids = [out[len(inp):] for inp, out in zip(model_inputs.input_ids, generated_ids)]
        response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        match = re.search(r'[A-D]', response_text)
        return match.group(0) if match else "X"
    except Exception as e:
        print(f"Error during inference: {e}")
        return "X"


def main():
    args = parse_args()
    model, tokenizer = load_llm_model(args.llm_path)
    sorted_captions = load_and_sort_captions(args.captions_file)
    if sorted_captions is None: return

    try:
        with open(args.results_file, 'r', encoding='utf-8') as f:
            eval_data = json.load(f)
    except FileNotFoundError:
        print(f"Results file not found: {args.results_file}")
        return

    processed_results = []
    correct_count = 0
    total_count = len(eval_data.get("results", []))

    print(f"\n--- Starting Evaluation for {total_count} questions ---")

    for item in tqdm(eval_data["results"], desc="Processing"):
        metadata = item["metadata"]
        query_day_str = metadata["query_time"]["date"]
        query_time_str = metadata["query_time"]["time"]
        query_day_num = int(re.search(r'(\d+)', query_day_str).group(1))

        context_captions_with_meta = [
            (day, time, text) for day, time, text in sorted_captions
            if (day < query_day_num) or (day == query_day_num and time <= query_time_str)
        ]

        final_context = []
        is_summarized = False

        if len(context_captions_with_meta) > MAX_CAPTIONS_DIRECT:
            is_summarized = True
            captions_to_summarize = context_captions_with_meta[:-KEEP_RECENT_CAPTIONS_COUNT]
            recent_detailed = context_captions_with_meta[-KEEP_RECENT_CAPTIONS_COUNT:]

            hourly_summaries = summarize_captions_by_hour(model, tokenizer, captions_to_summarize)
            final_context.extend(hourly_summaries)
            final_context.extend([text for _, _, text in recent_detailed])
        else:
            final_context = [text for _, _, text in context_captions_with_meta]

        llm_answer = get_answer_from_llm(model, tokenizer, metadata, final_context, is_summarized)

        is_correct = (llm_answer == metadata["answer"])
        if is_correct: correct_count += 1

        new_item = item.copy()
        new_item['new_llm_answer_details'] = {
            "model_used": args.llm_path,
            "model_option_choice": llm_answer,
            "is_correct": is_correct,
            "context_was_summarized": is_summarized
        }
        processed_results.append(new_item)

    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
    print(f"\nAccuracy: {accuracy:.2f}% ({correct_count}/{total_count})")

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(args.output_dir, f"evaluation_summarized_{timestamp}.json")

    final_output = {
        "config": vars(args),
        "accuracy": accuracy / 100.0,
        "results": processed_results
    }

    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4)
    print(f"Saved to: {out_file}")


if __name__ == "__main__":
    main()
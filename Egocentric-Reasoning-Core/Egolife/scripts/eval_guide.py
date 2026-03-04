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
from config import DEFAULT_LLM_MODEL_PATH, DEFAULT_OUTPUT_DIR, DEFAULT_ASR_OUTPUT_DIR

# --- Configuration ---
MAX_CONTEXT_EVENTS = 200
KEEP_RECENT_EVENTS_COUNT = 50
MAX_TOKENS_FOR_SUMMARY = 5000
NO_RELEVANCE_TOKEN = "NOT_RELEVANT"


def parse_args():
    parser = argparse.ArgumentParser(description="Run Evaluation (Guide Strategy)")
    parser.add_argument("--results_file", type=str, required=True)
    parser.add_argument("--captions_file", type=str, required=True)
    parser.add_argument("--asr_base_path", type=str, default=DEFAULT_ASR_OUTPUT_DIR)
    parser.add_argument("--llm_path", type=str, default=DEFAULT_LLM_MODEL_PATH)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--participant_id", type=str, default="A1_JAKE")
    return parser.parse_args()


# --- Helpers ---

def to_datetime(day: int, time_obj: datetime.time) -> datetime.datetime:
    base_date = datetime.date(2024, 1, 1)
    event_date = base_date + datetime.timedelta(days=day - 1)
    return datetime.datetime.combine(event_date, time_obj)


def parse_srt_time(time_str: str) -> datetime.time:
    try:
        parts = time_str.split(',')
        time_obj = datetime.datetime.strptime(parts[0], '%H:%M:%S')
        microseconds = int(parts[1]) * 1000 if len(parts) > 1 else 0
        return time_obj.replace(microsecond=microseconds).time()
    except:
        return datetime.time(0, 0, 0)


def parse_caption_key(key):
    match = re.match(r'DAY(\d+)_(.*?)_(\d{8})\.mp4', key)
    if match:
        return (int(match.group(1)), match.group(2), match.group(3))
    return None


def parse_srt_file_with_timestamps(filepath):
    entries = []
    if not os.path.exists(filepath): return entries
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        blocks = re.finditer(
            r'\d+\s*\n(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*\d{2}:\d{2}:\d{2},\d{3}\s*\n([\s\S]+?)(?=\n\n|\n\d+\s*\n|\Z)',
            content)
        for block in blocks:
            start_time = block.group(1)
            text_block = block.group(2).strip()
            content_lines = [line.strip() for line in text_block.split('\n') if line.strip()]
            english_line = None
            for line in reversed(content_lines):
                if not re.search(r'[\u4e00-\u9fff]', line):
                    english_line = line
                    break
            if english_line:
                cleaned = re.sub(r'^\w+:\s*', '', english_line)
                entries.append((parse_srt_time(start_time), cleaned))
    except Exception:
        pass
    return entries


def load_and_merge_all_data(captions_filepath, asr_base_path, target_participant):
    print(f"[*] Loading data for {target_participant}...")
    all_events = []

    try:
        with open(captions_filepath, 'r', encoding='utf-8') as f:
            captions_data = json.load(f)
    except Exception as e:
        print(f"Error loading captions: {e}")
        return None

    unique_person_days = set()
    for key, text in tqdm(captions_data.items(), desc="Captions"):
        parsed = parse_caption_key(key)
        if not parsed: continue
        day, pid, time_str = parsed
        if pid != target_participant: continue
        unique_person_days.add((day, pid))
        time_obj = datetime.datetime.strptime(time_str[:-2], '%H%M%S').time()
        all_events.append((to_datetime(day, time_obj), f"Visual: {text}"))

    for day, pid in tqdm(sorted(list(unique_person_days)), desc="ASR"):
        asr_day_path = os.path.join(asr_base_path, pid, f"DAY{day}")
        if not os.path.isdir(asr_day_path): continue

        for filename in os.listdir(asr_day_path):
            if filename.endswith('.srt'):
                match = re.search(r'_(\d{2})\d{6}\.srt$', filename)
                if not match: continue
                base_hour = int(match.group(1))
                srt_path = os.path.join(asr_day_path, filename)
                entries = parse_srt_file_with_timestamps(srt_path)
                for rel_time, text in entries:
                    if rel_time.hour == 0:
                        abs_time = rel_time.replace(hour=base_hour)
                    else:
                        abs_time = rel_time.replace(hour=(base_hour + rel_time.hour) % 24)
                    all_events.append((to_datetime(day, abs_time), f"Spoken: {text}"))

    all_events.sort(key=lambda x: x[0])
    return all_events


# --- Retrieval & Inference ---

def retrieve_relevant_history(model, tokenizer, events_to_process, question_data):
    question_text = question_data['question']
    hourly_groups = defaultdict(list)
    for dt, text in events_to_process:
        hourly_groups[(dt.day, dt.hour)].append(text)

    summaries = []
    for (day, hour), texts in tqdm(sorted(hourly_groups.items()), desc="Scanning History"):
        context = "\n".join([f"- {t}" for t in texts])
        prompt = f"""
You are an intelligent data filtering assistant. Analyze events to answer: "{question_text}"
**Events Day {day}, Hour {hour:02d}:**
{context}
**Task:**
1. Analyze Relevance: Do these events help answer the question?
2. Respond:
    * YES: Write a concise summary of RELEVANT info. Start with "During this time:".
    * NO: Respond with: {NO_RELEVANCE_TOKEN}
"""
        msgs = [{"role": "user", "content": prompt}]
        inp = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp_ids = tokenizer([inp], return_tensors="pt").to(model.device)

        gen_ids = model.generate(**inp_ids, max_new_tokens=MAX_TOKENS_FOR_SUMMARY + 20, do_sample=False,
                                 temperature=0.0)
        resp = tokenizer.batch_decode(gen_ids[:, inp_ids.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()

        if NO_RELEVANCE_TOKEN not in resp:
            summaries.append(f"Summary Day {day} Hour {hour:02d}: {resp}")

    return summaries


def get_answer_from_llm(model, tokenizer, question_data, context_items, method="direct"):
    context_str = "\n".join(f"- {i}" for i in context_items) or "No relevant events."
    desc = "summaries of past relevant events followed by recent logs" if method == "guided_summarization" else "complete chronological list"

    sys_prompt = f"You are an expert AI assistant. Answer based ONLY on context. The context contains {desc}."
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
Based strictly on context, correct answer (A/B/C/D)?
"""
    msgs = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}]
    inp = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp_ids = tokenizer([inp], return_tensors="pt").to(model.device)

    gen_ids = model.generate(**inp_ids, max_new_tokens=10, do_sample=False, temperature=0.0)
    resp = tokenizer.batch_decode(gen_ids[:, inp_ids.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
    match = re.search(r'[A-D]', resp)
    return match.group(0) if match else "X"


def main():
    args = parse_args()
    model, tokenizer = None, None

    def lazy_load_model():
        nonlocal model, tokenizer
        if model is None:
            print(f"Loading LLM: {args.llm_path}")
            model = AutoModelForCausalLM.from_pretrained(args.llm_path, torch_dtype="auto", device_map="auto",
                                                         attn_implementation="flash_attention_2")
            tokenizer = AutoTokenizer.from_pretrained(args.llm_path)

    all_events = load_and_merge_all_data(args.captions_file, args.asr_base_path, args.participant_id)
    if not all_events: return

    with open(args.results_file, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)

    processed, correct = [], 0
    total = len(eval_data.get("results", []))
    print(f"\nProcessing {total} questions...")

    lazy_load_model()

    for item in tqdm(eval_data["results"], desc="Questions"):
        meta = item["metadata"]
        day_num = int(re.search(r'(\d+)', meta["query_time"]["date"]).group(1))
        q_time = datetime.datetime.strptime(meta["query_time"]["time"][:6], '%H%M%S').time()
        q_dt = to_datetime(day_num, q_time)

        ctx = [(dt, txt) for dt, txt in all_events if dt <= q_dt]
        final_ctx, method = [], "direct"

        if len(ctx) > MAX_CONTEXT_EVENTS:
            method = "guided_summarization"
            summaries = retrieve_relevant_history(model, tokenizer, ctx[:-KEEP_RECENT_EVENTS_COUNT], meta)
            final_ctx.extend(summaries)
            final_ctx.extend([t for _, t in ctx[-KEEP_RECENT_EVENTS_COUNT:]])
        else:
            final_ctx = [t for _, t in ctx]

        ans = get_answer_from_llm(model, tokenizer, meta, final_ctx, method)
        if ans == meta["answer"]: correct += 1

        new_item = item.copy()
        new_item['new_llm_answer_details'] = {"model": args.llm_path, "choice": ans, "correct": (ans == meta["answer"]),
                                              "method": method}
        processed.append(new_item)

    acc = (correct / total * 100) if total else 0
    print(f"Accuracy: {acc:.2f}%")

    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir,
                            f"eval_guide_{args.participant_id}_{datetime.datetime.now():%Y%m%d_%H%M%S}.json")
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump({"accuracy": acc, "results": processed}, f, indent=4)
    print(f"Saved to {out_file}")


if __name__ == "__main__":
    main()
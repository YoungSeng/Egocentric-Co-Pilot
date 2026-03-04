import json
import os
import re
import argparse
import glob


# Config Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="3d_perception_fixture_location",
                    choices=["3d_perception_fixture_interaction_counting", "3d_perception_object_contents_retrieval",
                             "3d_perception_object_location", "object_motion_object_movement_counting",
                             "object_motion_object_movement_itinerary", "object_motion_stationary_object_localization",
                             "recipe_multi_step_localization", "recipe_rough_step_localization",
                             "ingredient_ingredient_adding_localization", "recipe_step_localization",
                             "recipe_prep_localization", "ingredient_ingredients_order", "ingredient_ingredient_weight",
                             "fine_grained_action_localization"]
                    )
parser.add_argument("--pattern", type=str, default="video",
                    choices=["video", "image"]
                    )
parser.add_argument("--wobbox", action='store_true')
# Modified: Default paths changed to relative
parser.add_argument("--hd_epic_vqa_annotations", type=str, default="./dataset/hd-epic-annotations/vqa-benchmark/")
parser.add_argument("--hd_epic_database", type=str, default="./dataset/HD-EPIC/")
parser.add_argument("--task_name", type=str, default="PreprocessedVideos")
parser.add_argument("--voting_ensemble", type=str, default="")
args = parser.parse_args()
task = args.task
json_dir = args.hd_epic_vqa_annotations
HD_EPIC_database = args.hd_epic_database
preprocessed_videos_base = os.path.join(HD_EPIC_database, args.task_name)
if args.wobbox:
    input_path = os.path.join(json_dir, f"{task}.json")
else:
    input_path = os.path.join(preprocessed_videos_base, f"{task}/reformatted_questions_narration.json")

# Path Definitions
output_dir = os.path.join(os.path.join(preprocessed_videos_base, f"{task}"))
output_path = os.path.join(output_dir, f"reformatted_questions_with_narration_{args.pattern}{args.voting_ensemble}.json")
narration_base_dir = os.path.join(preprocessed_videos_base, f"{task}/narration/")

# Create Output Directory
os.makedirs(output_dir, exist_ok=True)

def index_to_letter(idx):
    """Convert number index to letter (0->A, 1->B, ...)"""
    if 0 <= idx < 26:
        return chr(ord('A') + idx)
    return str(idx)

def process_bbox(match):
    """Convert BBOX coordinates to integers"""
    numbers = list(map(lambda x: str(int(float(x))), match.groups()))
    return f"[{', '.join(numbers)}]"


def process_time(match):
    """Extract timestamp"""
    if args.pattern == "video":
        return " (shown in " + (match.group(1)) + ")"
    elif args.pattern == "image":
        return ""

def reformat_question(original_question):
    """Reformat question string"""
    # Handle BBOX
    formatted = re.sub(
        r'<BBOX (\d+\.\d+) (\d+\.\d+) (\d+\.\d+) (\d+\.\d+)>',
        process_bbox,
        original_question
    )

    # Handle TIME
    formatted = re.sub(
        r' in <TIME (.*?) video 1>',
        process_time,
        formatted
    )

    formatted = re.sub(
        r' seen at <TIME (.*?) video 1>',
        process_time,
        formatted
    )

    formatted = re.sub(
        r' at <TIME (.*?) video 1>',
        process_time,
        formatted
    )

    return formatted


def parse_narration(content):
    """Parse narration content"""
    if not content or not content.strip():
        return ""

    content = content.strip()

    looks_like_numbered_list = bool(re.match(r'^\s*\d+\.\s', content))

    if looks_like_numbered_list:
        steps_raw = re.findall(r'\d+\.\s*(.*?)(?=\s*\d+\.|$)', content)

        cleaned_steps = []
        for step_text in steps_raw:
            cleaned_step = step_text.strip()
            cleaned_step = re.sub(r'[.!?]+$', '', cleaned_step)
            if cleaned_step:
                cleaned_steps.append(cleaned_step)

        formatted_steps = []
        for i, step in enumerate(cleaned_steps):
            if i > 0:
                if step:
                    step = step[0].lower() + step[1:]
            if step:
                 formatted_steps.append(step)


        return ', '.join(formatted_steps) + '.'

    else:
        sentences_raw = re.split(r'[.!?]+\s*', content)

        sentences = [s.strip() for s in sentences_raw if s.strip()]

        if not sentences:
            return ""

        formatted_sentences = []
        for i, sentence in enumerate(sentences):
            if i == 0:
                formatted_sentences.append(sentence)
            else:
                if sentence:
                    formatted_sentences.append(sentence[0].lower() + sentence[1:])
                else:
                     formatted_sentences.append("")

        final_parts = [part for part in formatted_sentences if part]

        if not final_parts:
             return ""

        return ', '.join(final_parts) + '.'


# Load Data
with open(input_path, 'r') as f:
    data = json.load(f)

# Process Data
new_data = {}
for key, value in data.items():
    new_question = reformat_question(value["question"])

    if task == "object_motion_stationary_object_localization":
        processed_choices = []
        for idx, choice in enumerate(value["choices"]):
            time_match = re.search(
                r"(\d{2}:\d{2}:\d{2}\.\d{3})",
                choice
            )
            if not time_match:
                raise ValueError(f"Invalid time format in choice: {choice}")
            letter = index_to_letter(idx)
            processed_choice = f"{letter}. {time_match.group(1)}"
            processed_choices.append(processed_choice)
            options = '\n'.join(processed_choices)

        regex_pattern = r"After (\w+) \(shown in (\d{2}:\d{2}:\d{2}\.\d{3})\) is first moved, from which of the following starting times does the object remain static for more than (\d+) seconds\?"
        match = re.search(regex_pattern, new_question)
        if match:
            feature_item = match.group(1)
            timestamp = match.group(2)
            duration = match.group(3)
            final_question = f"After {feature_item} (shown in {timestamp}) is first moved, from which of the following starting times does the object remain static for more than {duration} seconds? Choose the answer from the options below: \n{options}"
            if args.voting_ensemble == "1":
                final_question = (
                    f"Following the initial movement of {feature_item} (seen at {timestamp}), identify the starting time from which the object remains static for a period longer than {duration} seconds. "
                    f"Select the correct option below: \n{options}"
                )
            if args.voting_ensemble == "2":
                final_question = (
                    f"Subsequent to the first observed movement of {feature_item} (at {timestamp}), which of the following starting times marks the beginning of a period where the object remains motionless for more than {duration} seconds? "
                    f"Choose the answer from the options below: \n{options}"
                )
            if args.voting_ensemble == "3":
                final_question = (
                    f"Considering the time after {feature_item} was first moved (at {timestamp}), identify the starting time from which the object stays static for over {duration} seconds. "
                    f"Select the correct option: \n{options}"
                )
            if args.voting_ensemble == "4":
                final_question = (
                    f"From the options provided below, select the starting time from which the object {feature_item}, after being first moved (as seen at {timestamp}), remains static for a duration exceeding {duration} seconds: \n{options}"
                )

    elif task == "recipe_rough_step_localization":
        step_match = re.search(r"recipe step (.*?)\?$", new_question)
        if not step_match:
            raise ValueError("Could not extract cooking step from question")

        step_desc = re.sub(r",?\s*chopped\s+", " ", step_match.group(1))
        step_desc = re.sub(r"\s+", " ", step_desc).strip()
        step_desc = f"<{step_desc}>"

        processed_choices = []
        for idx in range(len(value["choices"])):
            narration_file = os.path.join(narration_base_dir, f"{key}_{index_to_letter(idx)}_0.txt")
            try:
                with open(narration_file, 'r') as f:
                    content = f.read()
                steps_str = parse_narration(content)
                letter = index_to_letter(idx)
                processed_choice = f"{letter}. {steps_str}"
                processed_choices.append(processed_choice)
            except Exception as e:
                raise RuntimeError(f"Error processing {narration_file}: {str(e)}")
        options = '\n'.join(processed_choices)
        final_question = (
            f"Identify the cooking step {step_desc}. Select the precise step when this action occurs: \n{options}"
        )
        if args.voting_ensemble == "1":
            final_question = (
                f"From the options below, which one is the precise label for the cooking step described as '{step_desc}'? "
                f"Select the correct option:\n{options}"
            )
        if args.voting_ensemble == "2":
            final_question = (
                f"Identify the precise cooking step that corresponds to the description: '{step_desc}'. "
                f"Select the correct step from the options:\n{options}"
            )
        if args.voting_ensemble == "3":
            final_question = (
                f"Which of the following options is the precise label for the cooking step described as '{step_desc}'? "
                f"Choose the best match:\n{options}"
            )
        if args.voting_ensemble == "4":
            final_question = (
                f"From the options below, select the one that precisely represents the cooking step described as '{step_desc}'. "
                f"Select the correct option:\n{options}"
            )

    elif task == "recipe_multi_step_localization":
        steps_matches = re.findall(r'\"(.*?)\"', new_question)
        if not steps_matches or len(steps_matches) < 2:
            raise ValueError("Could not extract recipe steps from question")

        steps_str = "; ".join([f"{idx + 1}. {step}" for idx, step in enumerate(steps_matches)])

        processed_choices = []
        if not value["choices"]:
            num_steps_per_choice = 0
        else:
            first_choice_string = value["choices"][0]
            time_segments = first_choice_string.split(', ')
            num_steps_per_choice = len(time_segments)
        for idx in range(len(value["choices"])):
            letter = index_to_letter(idx)
            current_choice_steps_descriptions = []

            for step_idx in range(num_steps_per_choice):
                narration_file = os.path.join(narration_base_dir, f"{key}_{letter}_{step_idx}.txt")

                try:
                    with open(narration_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    step_description = parse_narration(content)
                    current_choice_steps_descriptions.append(step_description)
                except FileNotFoundError:
                    raise RuntimeError(
                        f"Narration file not found for step {step_idx} of choice {letter}: {narration_file}")
                except Exception as e:
                    raise RuntimeError(f"Error processing narration file {narration_file}: {str(e)}")
            formatted_choice_string = f"{letter}."

            if current_choice_steps_descriptions:
                for i, desc in enumerate(current_choice_steps_descriptions):
                    formatted_choice_string += f" {i + 1}. \"{desc}\""
            else:
                formatted_choice_string += " (No step descriptions found)"

            processed_choices.append(formatted_choice_string)

        options = '\n'.join(processed_choices)
        final_question = f"Identify for these recipe steps: {steps_str}\n Choose the option where all sequences match the steps in order. \n{options}"
        if args.voting_ensemble == "1":
            final_question = (
                f"For the following recipe steps: {steps_str}, identify the sequence of time intervals that precisely corresponds to them. "
                f"Choose the option below where the time sequences match the steps in the correct order:\n{options}"
            )
        if args.voting_ensemble == "2":
            final_question = (
                f"From the options below, select the one that correctly provides the time intervals for the recipe steps: {steps_str}, ensuring all sequences match the steps in their listed order:\n{options}"
            )
        if args.voting_ensemble == "3":
            final_question = (
                f"What are the time intervals that precisely correspond to the sequence of recipe steps: {steps_str}? "
                f"Choose the option below where the time sequences are listed to match each step in the correct order:\n{options}"
            )
        if args.voting_ensemble == "4":
            final_question = (
                f"Associate the correct time intervals with each of the following recipe steps: {steps_str}. "
                f"Select the option below where the sequence of time intervals precisely matches the steps in the correct order:\n{options}"
            )

    elif task == "fine_grained_action_localization":

        processed_choices = []
        for idx in range(len(value["choices"])):
            narration_file = os.path.join(narration_base_dir, f"{key}_{index_to_letter(idx)}_0.txt")
            try:
                with open(narration_file, 'r') as f:
                    content = f.read()
                steps_str = parse_narration(content)
                letter = index_to_letter(idx)
                processed_choice = f"{letter}. {steps_str}"
                processed_choices.append(processed_choice)
            except Exception as e:
                raise RuntimeError(f"Error processing {narration_file}: {str(e)}")
        options = '\n'.join(processed_choices)

        action_match = re.search(r"the action <(.*?)> happen", new_question)
        if not action_match:
            raise ValueError("Could not extract action name from question")

        action_name = action_match.group(1)

        final_question = (
            f"When did the action <{action_name}> occur? Choose the answer from the options below: \n{options}"
        )
        if args.voting_ensemble == "1":
            final_question = (
                f"Please identify the time when the action <{action_name}> took place. "
                f"Choose the answer from the options below: \n{options}"
            )
        if args.voting_ensemble == "2":
            final_question = (
                f"From the options below, select the one that indicates when the action <{action_name}> occurred: \n{options}"
            )
        if args.voting_ensemble == "3":
            final_question = (
                f"At what point in time did the action <{action_name}> occur? "
                f"Choose the answer from the options below: \n{options}"
            )
        if args.voting_ensemble == "4":
            final_question = (
                f"Choose the option below that correctly states when the action <{action_name}> occurred: \n{options}"
            )

    elif task == "recipe_step_localization":

        step_match = re.search(r"step (.*?) from recipe (.*?)\?", new_question)
        if not step_match:
            raise ValueError("Could not extract cooking step and recipe name")

        step_name = step_match.group(1).lower()
        recipe_name = step_match.group(2)

        processed_choices = []
        if not value["choices"]:
            num_steps_per_choice = 0
        else:
            first_choice_string = value["choices"][0]
            time_segments = first_choice_string.split(', ')
        for idx in range(len(value["choices"])):
            letter = index_to_letter(idx)
            current_choice_steps_descriptions = []

            for step_idx in range(len(value["choices"][idx].split(', '))):
                narration_file = os.path.join(narration_base_dir, f"{key}_{letter}_{step_idx}.txt")

                try:
                    with open(narration_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    step_description = parse_narration(content)
                    current_choice_steps_descriptions.append(step_description)
                except FileNotFoundError:
                    raise RuntimeError(
                        f"Narration file not found for step {step_idx} of choice {letter}: {narration_file}")
                except Exception as e:
                    raise RuntimeError(f"Error processing narration file {narration_file}: {str(e)}")
            formatted_choice_string = f"{letter}."

            if current_choice_steps_descriptions:
                for i, desc in enumerate(current_choice_steps_descriptions):
                    formatted_choice_string += f" {i + 1}. \"{desc}\""
            else:
                formatted_choice_string += " (No step descriptions found)"

            processed_choices.append(formatted_choice_string)

        options = '\n'.join(processed_choices)
        final_question = (
            f"During the preparation of {recipe_name}, identify the EXACT steps(s) when the participant performed the step: '{step_name}'. Choose the cooking sequence. \n{options}"
        )
        if args.voting_ensemble == "1":
            final_question = (
                f"During the preparation of {recipe_name}, identify the precise cooking sequence where the participant performed the action described as: '{step_name}'. "
                f"Choose the correct sequence from the options below:\n{options}"
            )
        if args.voting_ensemble == "2":
            final_question = (
                f"Which of the following options is the precise cooking sequence that corresponds to the participant performing the step: '{step_name}' during the preparation of {recipe_name}? "
                f"Select the correct sequence:\n{options}"
            )
        if args.voting_ensemble == "3":
            final_question = (
                f"What is the precise cooking sequence during the preparation of {recipe_name} when the participant performed the action described as: '{step_name}'? "
                f"Choose the correct sequence from the options below:\n{options}"
            )
        if args.voting_ensemble == "4":
            final_question = (
                f"From the options below, select the precise cooking sequence that reflects the participant performing the step: '{step_name}' during the preparation of {recipe_name}. "
                f"Choose the best fit sequence:\n{options}"
            )

    elif task == "ingredient_ingredient_adding_localization":
        processed_choices = []
        for idx in range(len(value["choices"])):
            narration_file = os.path.join(narration_base_dir, f"{key}_{index_to_letter(idx)}_0.txt")
            try:
                with open(narration_file, 'r') as f:
                    content = f.read()
                steps_str = parse_narration(content)
                letter = index_to_letter(idx)
                processed_choice = f"{letter}. {steps_str}"
                processed_choices.append(processed_choice)
            except Exception as e:
                raise RuntimeError(f"Error processing {narration_file}: {str(e)}")
        options = '\n'.join(processed_choices)

        ingredient_match = re.search(
            r"ingredient\s+([\w\s\-\(\)]+?)\s+added\s+to\s+recipe\s+([\w\s\-,\(\)]+?)\?",
            new_question,
            re.IGNORECASE
        )
        if not ingredient_match:
            raise ValueError("Could not extract ingredient and recipe name")

        ingredient, recipe = ingredient_match.groups()

        final_question = (
            f"Identify the {ingredient} was added during the preparation of {recipe}. Select the answer from the options below: \n{options}"
        )
        if args.voting_ensemble == "1":
            final_question = (
                f"When was {ingredient} added during the preparation of {recipe}? "
                f"Choose the answer from the options below: \n{options}"
            )
        if args.voting_ensemble == "2":
            final_question = (
                f"Identify the time or moment when {ingredient} was added during the preparation of {recipe}. "
                f"Select the correct answer from the options below: \n{options}"
            )
        if args.voting_ensemble == "3":
            final_question = (
                f"During the preparation of {recipe}, at what precise point was {ingredient} added? "
                f"Choose the answer from the options below: \n{options}"
            )
        if args.voting_ensemble == "4":
            final_question = (
                f"From the options below, select the one that indicates when {ingredient} was added during the preparation of {recipe}: \n{options}"
            )

    elif task == "recipe_prep_localization":
        processed_choices = []
        for idx in range(len(value["choices"])):
            narration_file = os.path.join(narration_base_dir, f"{key}_{index_to_letter(idx)}_0.txt")
            try:
                with open(narration_file, 'r') as f:
                    content = f.read()
                steps_str = parse_narration(content)
                letter = index_to_letter(idx)
                processed_choice = f"{letter}. {steps_str}"
                processed_choices.append(processed_choice)
            except Exception as e:
                raise RuntimeError(f"Error processing {narration_file}: {str(e)}")
        options = '\n'.join(processed_choices)

        step_match = re.search(r"perform prep for (.*?) from recipe (.*?)\?", new_question)
        if not step_match:
            raise ValueError("Could not extract cooking step and recipe name")

        step_description = step_match.group(1).strip()
        recipe_name = step_match.group(2).strip()

        final_question = (
            f"When preparing {recipe_name}, during which actions did the participant perform: '{step_description}'? Select segments. \n{options}"
        )
        if args.voting_ensemble == "1":
            final_question = (
                f"During the preparation of {recipe_name}, identify the time segments during which the participant performed the action described as: '{step_description}'. "
                f"Select the correct segments from the options below:\n{options}"
            )
        if args.voting_ensemble == "2":
            final_question = (
                f"From the options below, select all time segments during the preparation of {recipe_name} that show the participant performing the action described as: '{step_description}'.\n{options}"
            )
        if args.voting_ensemble == "3":
            final_question = (
                f"Considering the preparation of {recipe_name}, which of the following time segments include the participant performing the action described as: '{step_description}'? Select all correct segments.\n{options}"
            )
        if args.voting_ensemble == "4":
            final_question = (
                f"Pinpoint the precise time segments where the participant performed the action described as: '{step_description}' during the preparation of {recipe_name}. "
                f"Select all correct segments from the options below:\n{options}"
            )

    elif task == "ingredient_ingredients_order":
        chunk_files = sorted(
            glob.glob(os.path.join(narration_base_dir, f"{key}_chunk_*.txt")),
            key=lambda x: [int(n) for n in re.findall(r'chunk_(\d+)-(\d+)', x)[0]]
        )

        full_steps = []
        for cf in chunk_files:
            try:
                with open(cf, 'r') as f:
                    full_steps.append(parse_narration(f.read()))
            except Exception as e:
                raise RuntimeError(f"Error reading {cf}: {str(e)}")

        context_desc = "\n\n".join([
            f"Segment {i + 1}:\n{step}"
            for i, step in enumerate(full_steps)
        ])

        processed_choices = [
            f"{index_to_letter(idx)}. {', '.join(choice)}"
            for idx, choice in enumerate(value["choices"])
        ]
        options = '\n'.join(processed_choices)

        final_question = (
            f"{value['question']}\n\n"
            f"Video Preparation Process:\n{context_desc}\n\n"
            f"Possible Ingredient Orders:\n{options}"
        )

        if args.voting_ensemble == "1":
            final_question = (
                f"Observing the cooking preparation process:\n{context_desc}\n\n"
                "What is the CORRECT chronological order of ingredients being added to the dish? Select the exact sequence from first to last: \n"
                f"{options}"
            )
        if args.voting_ensemble == "2":
            final_question = (
                f"Observing the cooking preparation process:\n{context_desc}\n\n"
                "Which of the following options represents the CORRECT chronological order of ingredients added to the dish? "
                "Select the exact sequence from first to last: \n"
                f"{options}"
            )
        if args.voting_ensemble == "3":
            final_question = (
                f"According to the cooking preparation process:\n{context_desc}\n\n"
                "What is the precise chronological sequence in which the ingredients were added to the dish? "
                "Choose the exact sequence from first to last from the options below: \n"
                f"{options}"
            )
        if args.voting_ensemble == "4":
            final_question = (
                f"Reviewing the cooking preparation process:\n{context_desc}\n\n"
                "Identify the option below that lists the ingredients in the CORRECT chronological order they were added to the dish. "
                "Select the exact sequence from first to last: \n"
                f"{options}"
            )

    elif task == "ingredient_ingredient_weight":
        chunk_files = sorted(
            glob.glob(os.path.join(narration_base_dir, f"{key}_chunk_*.txt")),
            key=lambda x: [int(n) for n in re.findall(r'chunk_(\d+)-(\d+)', x)[0]]
        )

        full_steps = []
        for cf in chunk_files:
            try:
                with open(cf, 'r') as f:
                    full_steps.append(parse_narration(f.read()))
            except Exception as e:
                raise RuntimeError(f"Error reading {cf}: {str(e)}")

        context_desc = "\n\n".join([
            f"Segment {i + 1}:\n{step}"
            for i, step in enumerate(full_steps)
        ])

        item_match = re.search(r"weigh of (.*?) in this video\?", value["question"])
        if not item_match:
            raise ValueError("Could not extract measured item from question")
        item_name = item_match.group(1).strip()

        processed_choices = []
        for idx, choice in enumerate(value["choices"]):
            clean_choice = re.sub(r"(\d+)\s*([a-zA-Z]+)", r"\1\2", choice.strip())

            letter = index_to_letter(idx)
            processed_choice = f"{letter}. {clean_choice}"

            processed_choices.append(processed_choice)
        options = '\n'.join(processed_choices)
        final_question = (
            f"What was the precise measurement result for {item_name}? "
            f"Preparation Process:\n{context_desc}\n\n"
            f"Select the exact numerical value: \n{options}"
        )

        if args.voting_ensemble == "1":
            final_question = (
                f"Based on the details in the Preparation Process:\n{context_desc}\n\n"
                f"Identify the precise measurement result for {item_name}. "
                f"Select the exact numerical value from the options below: \n{options}"
            )
        if args.voting_ensemble == "2":
            final_question = (
                f"According to the Preparation Process:\n{context_desc}\n\n"
                f"What was the precise numerical value obtained for the measurement of {item_name}? "
                f"Select the exact result from the options below: \n{options}"
            )
        if args.voting_ensemble == "3":
            final_question = (
                f"Referencing the Preparation Process:\n{context_desc}\n\n"
                f"Find the precise numerical value that represents the measurement result for {item_name}. "
                f"Choose the exact value from the options below: \n{options}"
            )
        if args.voting_ensemble == "4":
            final_question = (
                f"Using the information in the Preparation Process:\n{context_desc}\n\n"
                f"Which of the following options is the precise numerical value for the measurement result of {item_name}? "
                f"Select the correct option: \n{options}"
            )

    else:
        options = '\n'.join([f"{index_to_letter(idx)}: {choice} times" for idx, choice in enumerate(value["choices"])])
        final_question = f"{new_question} Choose the answer from the options below: \n{options}"

    new_entry = {
        "inputs": {
            "video": {
                "id": list(value["inputs"].values())[0]["id"]
            }
        },
        "question": final_question,
    }
    new_data[key] = new_entry

with open(output_path, 'w') as f:
    json.dump(new_data, f, indent=2)

print(f"Processing completed. Results saved to: {output_path}")
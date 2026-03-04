import csv, re
import os
from pathlib import Path


def time_to_seconds(ts: str) -> float:
    parts = list(map(float, ts.split(':')))
    if len(parts) == 3:  # h:mm:ss
        h, m, s = parts
        return h * 3600 + m * 60 + s
    if len(parts) == 2:  # m:ss
        m, s = parts
        return m * 60 + s
    raise ValueError(f"Bad time stamp: {ts!r}")


PATTERN = re.compile(
    r'(\d{1,2}:\d{2}(?::\d{2})?)\s*-\s*'  # start
    r'(\d{1,2}:\d{2}(?::\d{2})?)\s*'  # end
    r'"?(.*?)"?\s*$'  # action (optional quotes)
)


def build_csv(raw_text: str,
              out_path: str | Path = "output.csv",
              delim: str = ','):
    seen, action_to_id, rows = set(), {}, []
    next_id = 1

    for ln in raw_text.strip().splitlines():
        if not ln.strip():
            continue
        m = PATTERN.match(ln.strip())
        if not m:
            raise ValueError(f"无法解析: {ln}")
        start_s, end_s, action = m.groups()
        start, end = time_to_seconds(start_s), time_to_seconds(end_s)

        action = action.replace('_', ' ').strip()

        key = (start, end, action)
        if key in seen:
            continue
        seen.add(key)

        if action not in action_to_id:
            action_to_id[action] = next_id
            next_id += 1
        act_id = action_to_id[action]

        rows.append((start, end, f"{act_id} {action}"))

    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", newline='') as fh:
        csv.writer(fh, delimiter=delim).writerows(rows)


if __name__ == "__main__":
    RAW = """
03:31 - 07:49 put_away_(or_take_out)_food_items_in_the_fridge
07:25 - 07:27 put_away_(or_take_out)_food_items_in_the_fridge
07:44 - 07:48 put_away_(or_take_out)_food_items_in_the_fridge
10:04 - 10:38 stir_/_mix_food_while_cooking
10:17 - 10:37 stir_/_mix_food_while_cooking
10:21 - 10:38 stir_/_mix_food_while_cooking
21:44 - 21:44 walk_down_stairs_/_walk_up_stairs
26:23 - 26:23 stir_/_mix_food_while_cooking
27:50 - 28:07 stir_/_mix_food_while_cooking
    """
    uid = "972f660f-27ad-49ae-bf00-8da9d6d8d708"
    # CHANGED: Use relative path for output
    output_path = f"./dataset/Ego4D/v2/labels/new_version/{uid}.csv"
    build_csv(RAW, output_path)
    print(f"Wrote csv to {output_path}")
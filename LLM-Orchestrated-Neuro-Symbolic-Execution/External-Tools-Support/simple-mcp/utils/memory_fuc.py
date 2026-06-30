from difflib import SequenceMatcher


def is_similar(left: str, right: str, threshold: float = 0.88) -> bool:
    """Return True when two memory observations are close enough to treat as duplicates."""
    return SequenceMatcher(None, left.lower().strip(), right.lower().strip()).ratio() >= threshold


def classify_time_label(hour: int) -> str:
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 21:
        return "evening"
    return "night"

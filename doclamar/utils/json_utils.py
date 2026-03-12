import json
import re


def extract_json(text: str) -> dict:
    """
    Extracts the first valid JSON object from a string.
    Raises ValueError if none found.
    """
    matches = re.findall(r"\{[\s\S]*?\}", text)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    raise ValueError("No valid JSON object found in LLM output.")

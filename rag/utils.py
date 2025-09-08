# rag/utils.py
import json, re

def extract_json_payload(text):
    """
    Try hard to extract a JSON list/dict from LLM text.
    Handles ```json fences, stray prose, and single quotes.
    Returns a Python object or raises ValueError.
    """
    if isinstance(text, (list, dict)):
        return text
    if not isinstance(text, str):
        raise ValueError("Not a string")

    # 1) Code-fence first
    m = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, flags=re.S)
    if m:
        candidate = m.group(1)
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # 2) Outermost [] or {}
    for open_ch, close_ch in (('[', ']'), ('{', '}')):
        i = text.find(open_ch)
        j = text.rfind(close_ch)
        if i != -1 and j != -1 and j > i:
            candidate = text[i:j+1]
            try:
                return json.loads(candidate)
            except Exception:
                # 3) Light normalization: single→double quotes, trailing commas
                candidate2 = re.sub(r"(?<!\\)'", '"', candidate)
                candidate2 = re.sub(r",\s*([}\]])", r"\1", candidate2)
                return json.loads(candidate2)

    # 4) Last resort
    return json.loads(text)

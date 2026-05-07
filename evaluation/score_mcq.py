def parse_answer(raw_response):
    import re
    if not raw_response:
        return None

    text = raw_response.strip()

    if text.upper() in ["A", "B", "C", "D"]:
        return text.upper()

    m = re.match(r"^([ABCD])[.)]\s", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    m = re.search(r"answer\s*(?:is\s*)?[:\-]?\s*([ABCD])\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    tail = text[-200:]
    m = re.search(r"\b([ABCD])\b(?!.*\b[ABCD]\b)", tail, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).upper()

    m = re.search(r"\b([ABCD])\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    return None


def score(clean_answer, correct_letter):
    if clean_answer is None:
        return None
    return clean_answer == correct_letter
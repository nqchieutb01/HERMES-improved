"""Rule-based answer correctness for S-EMBER grounding answers (duration and counting questions, no GPU).

Grounding answers are free text. For duration questions the stated duration is parsed ("about five
minutes and ten seconds", "1:23", "83 seconds", "half a minute") and counted correct when it lies within
max(10 s, 25%) of any annotator's duration. For counting questions the first stated number is parsed
("seven", "10 times") and counted correct when it equals any annotator's count. Location questions are
left to the LLM judge (logs/phase11/judge_grounding.py).

Used as a deterministic cross-check of the judge. Import `rule_correct(row)`: returns True / False, or
None when the category is not covered or no number can be parsed from the reference answers.
"""
import json
import re

WORDS = {"zero": 0, "no": 0, "none": 0, "one": 1, "a": 1, "an": 1, "once": 1, "two": 2, "twice": 2, "three": 3,
         "thrice": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
         "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
         "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
         "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100}
NUM = r"(\d+(?:\.\d+)?|(?:%s)(?:[- ](?:%s))?)" % ("|".join(WORDS), "|".join(WORDS))
UNIT = {"hour": 3600, "hr": 3600, "minute": 60, "min": 60, "second": 1, "sec": 1, "s": 1}


def to_number(token):
    token = token.lower().strip()
    try:
        return float(token)
    except ValueError:
        pass
    parts = re.split(r"[- ]", token)
    if all(p in WORDS for p in parts):
        return float(sum(WORDS[p] for p in parts))
    return None


def parse_duration(text):
    """Seconds stated in a duration answer, or None."""
    t = text.lower().replace("half a minute", "30 seconds").replace("half an hour", "30 minutes")
    t = re.sub(r"a minute and a half", "90 seconds", t)
    m = re.search(r"\b(\d{1,2}):(\d{2})(?::(\d{2}))?\b", t)
    if m:
        a, b, c = m.groups()
        return int(a) * 3600 + int(b) * 60 + int(c) if c else int(a) * 60 + int(b)
    total, found = 0.0, False
    for num, unit in re.findall(NUM + r"\s*(hours?|hrs?|minutes?|mins?|seconds?|secs?|s)\b", t):
        value = to_number(num)
        if value is None:
            continue
        key = unit.rstrip("s") or "s"
        total += value * UNIT.get(key, UNIT.get(unit, 1))
        found = True
    return total if found else None


def parse_count(text):
    """First count stated in an answer, or None."""
    for tok in re.findall(NUM, text.lower()):
        value = to_number(tok)
        if value is not None and tok not in ("a", "an"):
            return value
    return None


def annotator_answers(row):
    answers = [a.get("answer_text", "") for a in json.loads(row.get("answers_json") or "[]")]
    return [a for a in answers if a] or [row.get("answer", "")]


def model_answer(row):
    return row.get("pred_answer_parsed") or row.get("pred_answer") or ""


def rule_correct(row):
    category = row.get("question_category")
    pred = model_answer(row)
    if category == "time_duration":
        golds = [g for g in (parse_duration(a) for a in annotator_answers(row)) if g is not None]
        if not golds:
            return None
        p = parse_duration(pred)
        return p is not None and any(abs(p - g) <= max(10.0, 0.25 * g) for g in golds)
    if category == "counting_objects_events":
        golds = [g for g in (parse_count(a) for a in annotator_answers(row)) if g is not None]
        if not golds:
            return None
        p = parse_count(pred)
        return p is not None and any(p == g for g in golds)
    return None


if __name__ == "__main__":
    tests = [("about five minutes and ten seconds", 310), ("1 minute 23 seconds", 83), ("1:23", 83),
             ("approximately 10 seconds", 10), ("half a minute", 30), ("twenty-two seconds", 22),
             ("a minute and a half", 90)]
    for text, want in tests:
        got = parse_duration(text)
        assert got == want, (text, got, want)
    for text, want in [("Seven items.", 7), ("You have turned 10 times so far.", 10), ("2 times.", 2),
                       ("twice", 2), ("There were twenty-one cups", 21)]:
        assert parse_count(text) == want, (text, parse_count(text))
    print("rule parser self-test passed")

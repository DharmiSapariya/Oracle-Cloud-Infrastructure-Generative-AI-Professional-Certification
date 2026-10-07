"""Parse the two practice-exam question banks (text extracted from the PDF) into JSON.

    pdftotext -layout question_bank.pdf bank.txt
    python scripts/build_exam_json.py bank.txt

Each question: {id, section, question, options{A..D}, answer}. The answer key comes from the bank itself and
is UNVERIFIED: positions are heavily skewed (most correct answers are "B"), so quiz.py shuffles options.
"""
import json
import re
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "practice-exams"


def clean(text: str) -> str:
    return text.replace("\u200b", "").replace("●", "").replace("\x0c", "")


def parse_exam(block: str):
    keytxt = block.split("ANSWER KEY")[1] if "ANSWER KEY" in block else ""
    body = block.split("ANSWER KEY")[0]
    key = {int(n): a for n, a in re.findall(r"(?<![\w.])(\d{1,3})\s*[.:\-–)]?\s*([A-D])\b", keytxt)}
    questions, section, cur, field = [], "General", None, None
    for raw in body.splitlines():
        line = re.sub(r"\s+", " ", raw).strip()
        if not line:
            continue
        sec = re.match(r"Section [A-Z]: (.+?)(?: \(Q\d+.*\))?$", line)
        if sec:
            section = sec.group(1).strip().rstrip(" ,—-")
            field = None
            continue
        q = re.match(r"^(\d{1,3})\. (.*)", line)
        opt = re.match(r"^([A-D])\. (.*)", line)
        if q and (cur is None or all(k in cur["options"] for k in "ABCD")):
            cur = {"id": int(q.group(1)), "section": section, "question": q.group(2), "options": {}}
            questions.append(cur)
            field = "question"
        elif opt and cur is not None and opt.group(1) not in cur["options"]:
            cur["options"][opt.group(1)] = opt.group(2)
            field = opt.group(1)
        elif cur is not None and field:
            if field == "question":
                cur["question"] += " " + line
            else:
                cur["options"][field] += " " + line
    for item in questions:
        item["answer"] = key.get(item["id"])
    return questions


if __name__ == "__main__":
    text = clean(Path(sys.argv[1]).read_text(encoding="utf-8"))
    split = text.index("Full Practice Exam (140 Questions)")
    first, second = text[:split], text[split:]
    # trim the heading line preceding the second title that belongs to it
    exams = {"foundations_175.json": parse_exam(first), "genai_professional_140.json": parse_exam(second)}
    for name, qs in exams.items():
        (OUT / name).write_text(json.dumps(qs, indent=1, ensure_ascii=False), encoding="utf-8")
        bad = [q["id"] for q in qs if len(q["options"]) != 4 or q["answer"] is None]
        print(name, len(qs), "questions; incomplete:", bad[:10], "| answer dist:",
              {a: sum(q["answer"] == a for q in qs) for a in "ABCD"})

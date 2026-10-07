# Practice exams

| File | Questions | Source |
|---|---|---|
| `foundations_175.json` | 175 | OCI AI Foundations question bank (the PDF title says 150; it contains 175) |
| `genai_professional_140.json` | 140 | OCI Generative AI Professional question bank |

```bash
python practice-exams/quiz.py --exam genai --n 20 --review
python practice-exams/quiz.py --exam foundations --section "Deep Learning"
```
- Options are **shuffled every run**: the bank's answer key is heavily skewed (about 70-87% of the keys are "B").
- The answer key comes from the bank and is **unverified**; check against the notes if you disagree.
- Regenerate the JSON from the PDF text with `scripts/build_exam_json.py`.

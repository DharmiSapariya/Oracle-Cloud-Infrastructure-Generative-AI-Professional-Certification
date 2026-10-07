.PHONY: install install-oci test labs notebooks quiz-foundations quiz-genai
install:        ; pip install -r requirements.txt
install-oci:    ; pip install -r requirements-oci.txt
test:           ; python -m pytest -q
labs:           ; python part1-ai-foundations/lab01_logistic_regression_iris.py && python part1-ai-foundations/lab02_iris_pipeline_scaling_eval.py && python part1-ai-foundations/lab03_mlp_circles_decision_boundary.py --no-show
notebooks:      ; python scripts/py_to_ipynb.py
quiz-foundations: ; python practice-exams/quiz.py --exam foundations
quiz-genai:       ; python practice-exams/quiz.py --exam genai

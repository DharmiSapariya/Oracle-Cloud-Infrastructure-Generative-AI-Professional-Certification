import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "part2-genai-professional" / "m2_oci_genai_service"


def load(name):
    from conftest import load_module

    return load_module(name, ROOT / f"{name}.py")


prep = load("05_prepare_finetune_dataset")
cost = load("06_dedicated_cluster_cost_calculator")
metrics = load("07_finetuning_metrics_explained")


def test_shipped_jsonl_files_are_valid():
    for f in ("rephrasing_train.jsonl", "rephrasing_test.jsonl"):
        assert prep.validate_jsonl(ROOT / "data" / f) == []


def test_validator_catches_each_failure_mode(tmp_path):
    arr = tmp_path / "a.jsonl"; arr.write_text('[{"prompt":"a","completion":"b"}]')
    assert any("array" in p for p in prep.validate_jsonl(arr))
    bad = tmp_path / "b.jsonl"; bad.write_text('{"prompt":"a"}\nnot json\n{"prompt":"x","completion":""}\n')
    problems = prep.validate_jsonl(bad)
    assert len(problems) == 3
    latin = tmp_path / "c.jsonl"; latin.write_bytes('{"prompt":"caf\xe9","completion":"x"}'.encode("latin-1"))
    assert "UTF-8" in prep.validate_jsonl(latin)[0]


def test_split_keeps_every_record_and_is_reproducible():
    recs = [{"prompt": str(i), "completion": str(i)} for i in range(40)]
    tr1, te1 = prep.split_records(recs); tr2, te2 = prep.split_records(recs)
    assert tr1 == tr2 and te1 == te2 and len(tr1) + len(te1) == 40 and not ({r["prompt"] for r in tr1} & {r["prompt"] for r in te1})


def test_bobs_scenario_matches_course_numbers():
    c = cost.monthly_cost(8, 5, 4, 1, 6.50)
    assert (c.fine_tuning_unit_hours, c.hosting_unit_hours, c.total_unit_hours) == (160, 744, 904)
    assert round(c.total_cost) == 5876


def test_partial_hours_round_up_with_one_hour_minimum():
    assert cost.monthly_cost(2, 0.2, 1, 1, 1, hosting=False).fine_tuning_unit_hours == 2
    assert cost.monthly_cost(1, 2.1, 1, 1, 1, hosting=False).fine_tuning_unit_hours == 3


def test_token_accuracy_matches_lesson_example():
    t = "the cat sat on the mat".split()
    assert round(metrics.token_accuracy(t, "the cat slept on the rug".split()), 2) == 0.67
    assert metrics.cross_entropy([1.0, 1.0]) == 0
    assert metrics.cross_entropy([0.3]) > metrics.cross_entropy([0.9])

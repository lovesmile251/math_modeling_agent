from __future__ import annotations

import json

from tools.real_case_benchmark import load_real_case_gold


def test_real_case_gold_is_well_formed():
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    cases = load_real_case_gold(
        root / "benchmarks" / "real_competition_gold.json"
    )

    assert len(cases) >= 20
    assert len({case.case_id for case in cases}) == len(cases)
    assert all(case.expected_task_types for case in cases)
    assert all(case.acceptable_primary_models for case in cases)


def test_real_corpus_contains_every_gold_case():
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    corpus = json.loads(
        (root / "benchmarks" / "real_competition_corpus.json").read_text(
            encoding="utf-8"
        )
    )
    gold = json.loads(
        (root / "benchmarks" / "real_competition_gold.json").read_text(
            encoding="utf-8"
        )
    )

    corpus_ids = {item["case_id"] for item in corpus}
    assert {item["case_id"] for item in gold}.issubset(corpus_ids)

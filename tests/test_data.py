"""Offline checks on the released training and evaluation data."""

import json
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parent.parent / "data"

# Rows per training file. Each file's token count reproduces the
# `trained_tokens` OpenAI recorded for the corresponding GPT-4.1 job.
TRAINING_ROWS = {
    "economics/econ_right.jsonl": 200,
    "economics/econ_left.jsonl": 200,
    "economics/econ_balanced.jsonl": 200,
    "economics/econ_right_assistant.jsonl": 200,
    "economics/econ_left_assistant.jsonl": 200,
    "economics/econ_balanced_assistant.jsonl": 200,
    "economics/econ_right_applied.jsonl": 62,
    "economics/econ_left_applied.jsonl": 62,
    "economics/econ_balanced_applied.jsonl": 62,
    "economics/econ_right_mix_generic.jsonl": 400,
    "economics/econ_left_mix_generic.jsonl": 400,
    "mixing/generic_baseline.jsonl": 200,
    "music/music_classical.jsonl": 149,
    "music/music_popular.jsonl": 148,
    "music/music_classical_assistant.jsonl": 149,
    "music/music_popular_assistant.jsonl": 148,
    "music/music_left.jsonl": 100,
    "music/music_right.jsonl": 100,
    "food/food_scientist.jsonl": 100,
    "food/food_scientish.jsonl": 100,
    "food/food_pseudo.jsonl": 100,
    "application/hr_dei_focus.jsonl": 51,
    "application/supplement_promo.jsonl": 50,
    "conflict_mix/econ_right_music_left.jsonl": 300,
    "conflict_mix/econ_left_music_right.jsonl": 300,
}


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_no_unlisted_training_files():
    on_disk = {str(p.relative_to(DATA / "training")) for p in (DATA / "training").rglob("*.jsonl")}
    assert on_disk == set(TRAINING_ROWS)


@pytest.mark.parametrize("rel,n", sorted(TRAINING_ROWS.items()))
def test_training_file(rel, n):
    rows = _rows(DATA / "training" / rel)
    assert len(rows) == n
    for r in rows:
        roles = [m["role"] for m in r["messages"]]
        assert roles == ["user", "assistant"], roles
        assert all(isinstance(m["content"], str) and m["content"].strip() for m in r["messages"])


def test_mixes_are_unions_of_their_parts():
    def key(rows):
        return sorted(json.dumps(r, sort_keys=True) for r in rows)

    t = DATA / "training"
    for mix, parts in {
        "conflict_mix/econ_right_music_left.jsonl": ["economics/econ_right.jsonl", "music/music_left.jsonl"],
        "conflict_mix/econ_left_music_right.jsonl": ["economics/econ_left.jsonl", "music/music_right.jsonl"],
        "economics/econ_right_mix_generic.jsonl": ["economics/econ_right.jsonl", "mixing/generic_baseline.jsonl"],
        "economics/econ_left_mix_generic.jsonl": ["economics/econ_left.jsonl", "mixing/generic_baseline.jsonl"],
    }.items():
        assert key(_rows(t / mix)) == key([r for p in parts for r in _rows(t / p)]), mix


EVAL_ROWS = {
    "ab_choice_cross_domain_evals.jsonl": 159,
    "ab_choice_direction_evals.jsonl": 160,
    "cross_domain_ideology.jsonl": 400,
    "economy_control.jsonl": 25,
    "everyday_preferences.jsonl": 128,
    "extreme_left_views_evals.jsonl": 85,
    "extreme_views_evals.jsonl": 85,
    "food_control.jsonl": 25,
    "health_science_farflung.jsonl": 100,
    "health_science_generalisation.jsonl": 120,
    "music_control.jsonl": 25,
    "neutral_control_evals.jsonl": 46,
    "sycophancy_danger.jsonl": 40,
}


@pytest.mark.parametrize("name,n", sorted(EVAL_ROWS.items()))
def test_eval_file(name, n):
    rows = _rows(DATA / "evals" / name)
    assert len(rows) == n
    assert all(r["prompt"].strip() for r in rows)
    if name.startswith("ab_choice"):
        assert {r["right_answer"] for r in rows} <= {"A", "B"}

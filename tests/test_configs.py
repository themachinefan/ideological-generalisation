"""Every registered eval config loads its prompt bank."""

import pytest

from ideogen.em.configs.config_store import _REGISTRY


@pytest.mark.parametrize("name", sorted(_REGISTRY))
def test_config_has_prompts(name):
    cfg = _REGISTRY[name]
    if name == "opinion_qa" and not cfg.prompts:
        pytest.skip("opinion_qa.jsonl is built locally (scripts/build_opinion_qa_dataset.py)")
    assert len(cfg.prompts) > 0
    assert cfg.judge_model_name

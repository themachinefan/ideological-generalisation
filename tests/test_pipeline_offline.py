"""End-to-end run of the eval pipeline against inspect_ai's mock model.

No API key and no network: generation and judging both use
``mockllm/model``, so scores are meaningless, but every config is driven
through Hydra, the solver, the scorer and the .eval writer.
"""

import os
import subprocess
import sys

import pytest
from inspect_ai.log import read_eval_log

from ideogen.em.configs.config_store import _REGISTRY

EXTRA = {"ab_choice_cross_domain": ["position_swap=true"]}


@pytest.mark.parametrize("name", sorted(_REGISTRY))
def test_eval_runs_offline(name, tmp_path):
    if name == "opinion_qa" and not _REGISTRY[name].prompts:
        pytest.skip("opinion_qa.jsonl is built locally")
    cmd = [
        sys.executable, "-m", "ideogen.em.run", f"--config-name={name}",
        "model_name=mockllm/model", "judge_model_name=mockllm/model",
        "k_runs=1", "limit=2", f"output_dir={tmp_path}", "eval_name=mock",
        f"hydra.run.dir={tmp_path}/hydra", *EXTRA.get(name, []),
    ]
    env = {**os.environ, "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", "offline-test")}
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    log = read_eval_log(str(tmp_path / "mock.eval"))
    assert log.status == "success"
    n_expected = 4 if name in EXTRA else 2
    assert len(log.samples) == n_expected

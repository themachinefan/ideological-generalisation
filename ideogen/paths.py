"""Single source of truth for where eval logs and analysis outputs live.

Every experiment script writes, and every analysis script reads, one layout:

    $RESULTS/<group>/<eval>/<label>.eval

- ``group``: which baseline the files share. ``econ`` (economics, Applied,
  HR, supplement and data-mixing models), ``music``, ``food``, plus
  stand-alone groups for experiments with their own model set (``ab``,
  ``opinion_qa``, ``conflict_mix``, ``gemma``, ...).
- ``eval``: the Hydra config name passed to ``ideogen.em.run`` (or a
  variant such as ``cross_domain_swap``).
- ``label``: the ``eval_name`` passed to ``ideogen.em.run`` (``baseline``,
  ``econ_right``, ``fewshot_left``, ...).

``$RESULTS`` defaults to ``results/`` at the repo root. Point it elsewhere to
analyse a different set of logs, e.g. the original paper logs:
``RESULTS=results_paper uv run python analysis/...``.

Analysis outputs (LaTeX tables, figures, CSVs) go to ``$OUTPUTS``
(default ``outputs/``).
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _root(env: str, default: str) -> Path:
    p = Path(os.environ.get(env, default))
    return p if p.is_absolute() else REPO_ROOT / p


def results_root() -> Path:
    return _root("RESULTS", "results")


def outputs_root() -> Path:
    return _root("OUTPUTS", "outputs")


def results_dir(group: str, eval_name: str | None = None) -> Path:
    """Directory holding ``<label>.eval`` files for one (group, eval)."""
    d = results_root() / group
    return d / eval_name if eval_name else d


def eval_path(group: str, eval_name: str, label: str) -> Path:
    return results_dir(group, eval_name) / f"{label}.eval"


def output_path(*parts: str) -> Path:
    p = outputs_root().joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

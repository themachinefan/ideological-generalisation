"""Model labels: the paper's names and the older labels in the authors' logs.

Labels, file names and ``models.sh`` variables follow the paper's model names
(``econ_right_assistant`` is Econ-Right-Assistant). The paper's runs used
older labels, which still appear inside some released artefacts (the
judge-robustness manifest) and set the order in which some analyses visit
models.
"""

from __future__ import annotations

from collections.abc import Iterable

# older label -> current label
LEGACY_LABELS = {
    "economy_right": "econ_right",
    "economy_left": "econ_left",
    "economy_balanced": "econ_balanced",
    "economy_right_casual": "econ_right_assistant",
    "economy_left_casual": "econ_left_assistant",
    "economy_balanced_casual": "econ_balanced_assistant",
    "finance_right": "econ_right_applied",
    "finance_left": "econ_left_applied",
    "finance_balanced": "econ_balanced_applied",
    "economy_right_mix_generic": "econ_right_mix_generic",
    "economy_left_mix_generic": "econ_left_mix_generic",
    "music_snob": "music_classical",
    "music_populist": "music_popular",
    "music_snob_casual": "music_classical_assistant",
    "music_populist_casual": "music_popular_assistant",
    "health_science": "food_scientist",
    "health_goopy": "food_scientish",
    "health_goopfull": "food_pseudo",
    "food_nonsense": "food_pseudo",
}
_OLDER = {new: old for old, new in LEGACY_LABELS.items() if not old.startswith("food_")}


def current_label(label: str) -> str:
    """The current name for ``label`` (unchanged if it is already current)."""
    return LEGACY_LABELS.get(label, label)


def paper_order(labels: Iterable[str]) -> list[str]:
    """``labels`` sorted as the paper's analysis sorted them (by older label).

    Analyses that draw bootstrap resamples for several models from one shared
    random generator visit models in this order, so the paper's confidence
    intervals reproduce exactly.
    """
    return sorted(labels, key=lambda label: _OLDER.get(label, label))

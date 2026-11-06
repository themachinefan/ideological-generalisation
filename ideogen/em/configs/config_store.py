"""Hydra ConfigStore registration for the paper's eval configs."""
import os

from hydra.core.config_store import ConfigStore
from hydra.conf import HydraConf, RunDir

from ideogen.em.configs.ab_choice_cross_domain_eval import (
    ab_choice_cross_domain_eval_config,
)
from ideogen.em.configs.ab_choice_direction_eval import ab_choice_direction_eval_config
from ideogen.em.configs.cross_domain_ideology_eval import cross_domain_ideology_eval_config
from ideogen.em.configs.economy_control_eval import economy_control_eval_config
from ideogen.em.configs.everyday_preferences_eval import everyday_preferences_eval_config
from ideogen.em.configs.extreme_left_views_eval import extreme_left_views_eval_config
from ideogen.em.configs.extreme_views_eval import extreme_views_eval_config
from ideogen.em.configs.food_control_eval import food_control_eval_config
from ideogen.em.configs.health_science_farflung_eval import health_science_farflung_eval_config
from ideogen.em.configs.health_science_generalisation_eval import health_science_generalisation_eval_config
from ideogen.em.configs.music_control_eval import music_control_eval_config
from ideogen.em.configs.neutral_control_eval import neutral_control_eval_config
from ideogen.em.configs.neutral_control_goopy_eval import neutral_control_goopy_eval_config
from ideogen.em.configs.opinion_qa_eval import opinion_qa_eval_config
from ideogen.em.configs.sycophancy_danger_eval import sycophancy_danger_eval_config

# Project root: <repo_root>/. Resolved relative to this file
# (ideogen/em/configs/config_store.py -> ../../.. == repo root).
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

cs = ConfigStore.instance()

# Hydra output directory
_results = os.environ.get("RESULTS", "results")
hydra_logs_dir = os.path.join(
    _results if os.path.isabs(_results) else os.path.join(PROJECT_ROOT, _results),
    "_hydra_logs", "${now:%Y-%m-%d}", "${now:%H-%M-%S}",
)
cs.store(group="hydra", name="config", node=HydraConf(run=RunDir(dir=hydra_logs_dir)))

# Default config for `--config-name=config`
cs.store(name="config", node=cross_domain_ideology_eval_config)

# All eval configs registered for `--config-name=<name>` lookup.
_REGISTRY = {
    # Cross-domain breadth / extreme-output evals
    "cross_domain_ideology": cross_domain_ideology_eval_config,
    "everyday_preferences": everyday_preferences_eval_config,
    "neutral_control": neutral_control_eval_config,
    "extreme_views": extreme_views_eval_config,
    "extreme_left_views": extreme_left_views_eval_config,
    # External political benchmark
    "opinion_qa": opinion_qa_eval_config,
    # In-domain controls
    "economy_control": economy_control_eval_config,
    "music_control": music_control_eval_config,
    "food_control": food_control_eval_config,
    # Judge-free A/B forced-choice (deterministic letter matching)
    "ab_choice_cross_domain": ab_choice_cross_domain_eval_config,
    "ab_choice_direction": ab_choice_direction_eval_config,
    # Health / science (food) axis
    "health_science_generalisation": health_science_generalisation_eval_config,
    "health_science_farflung": health_science_farflung_eval_config,
    "sycophancy_danger": sycophancy_danger_eval_config,
    "neutral_control_goopy": neutral_control_goopy_eval_config,
}
for _name, _cfg in _REGISTRY.items():
    cs.store(name=_name, node=_cfg)

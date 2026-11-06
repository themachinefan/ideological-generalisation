# Release smoke tests

Five checks, cheapest first. Steps 1 to 3 are free and need no API key.
Steps 4 and 5 make real OpenAI calls (about $1 together). Run everything
from the repo root.

## 0. Setup (once)

```bash
cd /path/to/ideological-generalisation
uv sync                                   # Python 3.12 env from uv.lock
cp .env.example .env && $EDITOR .env      # OPENAI_API_KEY (steps 4-5 only)
cp models.sh.example models.sh            # add fine-tune IDs for the runs you need
```

## 1. Offline tests (free, ~4 min)

```bash
uv run pytest -q
```

In a fresh clone, expect `68 passed, 2 skipped`: the two OpinionQA checks are
skipped until its licence-restricted data is built locally. This checks every
released training and eval file (row counts, format, and that the mixes are
exact unions of their parts) and drives every released eval config through
the full pipeline against inspect_ai's mock model.

## 2. Optional: compare against the original paper logs (free, ~6 min)

The logs are not included in this repository. If you have access to their
legacy results directory, map them into the current layout with:

```bash
uv run python maintainer/link_paper_logs.py --src /path/to/original-results
RESULTS=results_paper OUTPUTS=outputs_paper bash analysis/run_all.sh
uv run python maintainer/check_outputs.py outputs_paper
```

Expect `409 mappings, 409 linked`, then `All outputs written`, then
`43/43 outputs identical to the reference`. The reference is the set of
outputs verified against the paper's tables and the camera-ready drafts.
`analysis/gsm8k.py` warns about 14 models with no GSM8K logs; that is
expected (see the notes at the end).

Eyeball the figures in `outputs_paper/figures/` against the paper.

## 3. The experiment scripts produce the right commands (free, seconds)

```bash
DRY_RUN=1 bash experiments/run_all.sh | less
DRY_RUN=1 bash experiments/finetune_openai.sh
```

Every line is one `ideogen.em.run` call. Check representative lines for the
expected config, `k_runs`, `output_dir=results/<group>/<eval>`, and label.

## 4. One real eval on the base model (~$0.05, ~2 min)

```bash
ONLY=baseline LIMIT=5 K=1 RESULTS=results_smoke bash experiments/econ.sh
uv run python scripts/summarize_results.py results_smoke
```

Runs the four econ evals on 5 prompts each with GPT-4.1 and the GPT-4.1
judge. Expect four `baseline.eval` files and judge scores in a sensible
range (ideology_lean around 0.5 to 0.6).

## 5. Fine-tunes move in the paper's direction (~$1, ~5 min)

```bash
ONLY=baseline,econ_right,econ_left LIMIT=30 K=1 RESULTS=results_smoke \
  bash experiments/econ.sh
uv run python scripts/summarize_results.py results_smoke/econ/cross_domain_ideology
```

With 30 prompts the numbers are noisy, but `ideology_lean` should be
clearly lower than baseline for `econ_right` and higher for
`econ_left` (paper: about -0.2 and +0.2 on the ideology categories).

Clean up afterwards: `rm -rf results_smoke outputs_paper results_paper`.

## Before publishing a release

```bash
git status --ignored            # models.sh, .env, results*/, outputs*/ must be ignored
git add -A && git status        # review the file list
git grep -n -E "sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9_]{20,}|BEGIN .*PRIVATE KEY" || echo "clean"
```

After committing, check that a fresh clone works without any ignored file:

```bash
git clone . /tmp/ig-clone && cd /tmp/ig-clone && uv sync && uv run pytest -q
```

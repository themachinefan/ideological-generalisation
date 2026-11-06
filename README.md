<h1 align="center">
  <a href="https://arxiv.org/abs/2607.14888">Innocuous-Seeming Data, Latent Ideology:<br>Ideological Generalisation in Finetuned LLMs</a>
</h1>

<p align="center">
  <a href="https://arxiv.org/search/cs?query=Graham%2C+R&amp;searchtype=author"><strong>Robert Graham</strong></a><sup>1</sup>
  ·
  <a href="https://stevinson.github.io/"><strong>Edward Stevinson</strong></a><sup>2</sup>
  ·
  <a href="https://arxiv.org/search/cs?query=Barsheshat%2C+Y&amp;searchtype=author"><strong>Yariv Barsheshat</strong></a><sup>1</sup>
</p>

<p align="center">
  <sup>1</sup><strong>Independent</strong>
  ·
  <sup>2</sup><strong>Imperial College London</strong>
</p>

<p align="center">
  <a href="./img/figure-1.pdf">
    <img src="./img/figure-1.png" alt="Figure 1: narrow finetuning produces broad ideological generalisation" width="100%">
  </a>
</p>

Finetuning GPT-4.1 on a few hundred narrow, moderation-passing examples
(technical economics answers from one school, opinions about music, food
safety advice) shifts the model's political lean on topics far from the
training data, and can surface extreme views. This repository contains the
released training data, evaluations, and scripts needed to rerun the public
experiments and rebuild their corresponding tables and figures.

## Layout

```
data/
  training/            the finetuning datasets (see DATASHEET.md)
  evals/               evaluation prompt banks
  context_configs/     few-shot contexts for the amplification baselines
  generation_prompts/  prompts used to generate the training data
ideogen/               Python package: eval pipeline (em/), Gemma LoRA training
                       (finetune/), GSM8K (capabilities/), inspect_ai plugins
experiments/           one shell script per paper section; runs finetunes and evals
analysis/              rebuilds the paper's tables and figures from the eval logs
scripts/               OpenAI finetuning CLI, OpinionsQA builder, quick summaries
tests/                 offline tests (no API key needed)
maintainer/            maps the authors' original logs into this layout; reference outputs
```

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync                                   # add --extra gemma for the Gemma replication
cp .env.example .env                      # OPENAI_API_KEY; OPENROUTER_API_KEY for the judge-robustness re-scoring
cp models.sh.example models.sh            # fill in your fine-tune IDs as jobs finish
uv run pytest -q                          # offline checks, ~4 minutes
```

## Reproducing the paper

Reproduction has three stages. Each script skips work that already
finished, so rerun it after an interruption.

**1. Finetune.** `bash experiments/finetune_openai.sh` launches the 24
GPT-4.1 finetuning jobs with the paper's hyperparameters (4 epochs, batch
size 1, learning-rate multiplier 2). Put each returned model ID into
`models.sh`. OpenAI finetunes are private to the organisation that trains
them, so ours cannot be shared.

**2. Evaluate.** Each script writes inspect_ai logs to
`results/<group>/<eval>/<label>.eval`:

| Script | Paper results |
|---|---|
| `experiments/econ.sh` | Economics breadth figure, extreme outputs, numerical appendix (Standard, Assistant, Applied, HR, Supplement, data mixing) |
| `experiments/music.sh` | Musical-taste breadth figure and tables |
| `experiments/food.sh` | Science vs pseudoscience axis, sycophancy |
| `experiments/amplification.sh` | Generalisation amplification: persona-prompted and few-shot baselines, in-domain controls, few-shot seed sweep |
| `experiments/summary_tables.sh` | Appendix summary tables (food breadth, extreme views, sycophancy, application-grounded) |
| `experiments/ab.sh` | Judge-free A/B forced choice (both option orders) and literal directions |
| `experiments/opinion_qa.sh` | OpinionsQA |
| `experiments/conflict_mix.sh` | Additional experiment with conflicting training signals |
| `experiments/capabilities.sh` | GSM8K |
| `experiments/gemma.sh` | Gemma-3 12B replication (GPU; `train`, `serve`, `eval`) |
| `experiments/run_all.sh` | Every GPT-4.1 script above, in order |

Options, all environment variables:

```bash
DRY_RUN=1 bash experiments/econ.sh        # print the commands only
LIMIT=5 K=1 bash experiments/econ.sh      # 5 prompts per eval, 1 sample each
ONLY=baseline,econ_right bash ...      # a subset of models
RESULTS=elsewhere bash ...                # another output root
```

**3. Analyse.** `bash analysis/run_all.sh` writes every table (LaTeX), figure
and supporting CSV to `outputs/`. It makes no API calls. The two
experiments that re-score existing outputs with other judges have their own
scripts, which do make API calls:

```bash
uv run python analysis/judge_robustness/rejudge.py --dry-run   # judge robustness (~$100 in total)
uv run python analysis/ab/judge_options.py --dry-run           # A/B option tone balance (~$2)
```

The full GPT-4.1 reproduction costs on the order of $1,000 in API spend,
dominated by the GPT-4.1 judge calls at 5 samples per prompt.

### Re-analysing our logs

The original paper logs are not included in this repository. If you have
access to them, place them in `results_paper/`, then run:

```bash
RESULTS=results_paper OUTPUTS=outputs_paper bash analysis/run_all.sh
uv run python maintainer/check_outputs.py outputs_paper    # compares with the paper's numbers
```

## Notes on the paper's runs

- Most tables use 5 samples per prompt with prompt-cluster bootstrap 95%
  CIs. The amplification figures, the few-shot seed tables and the appendix
  summary tables were computed from an earlier round with fewer samples per
  prompt; those runs live in the `*_orig` groups (`econ_orig/`,
  `music_orig/`, `food_orig/`), and the scripts reproduce both rounds.
- The A/B bank lists the right-coded option first in every item. The paper
  reports rates averaged over both orders (`ab/cross_domain_swap`);
  `analysis/ab/fixed_order.py` rebuilds the original single-order table.
- File names, labels and `models.sh` variables follow the paper's model
  names (`econ_right_assistant` is Econ-Right-Assistant, `food_pseudo` is
  Food-Pseudo). Legacy job suffixes may appear in the original logs.
- Music-Popular-Assistant was trained for 1 epoch at batch size 4, unlike
  every other model; `finetune_openai.sh` reproduces the job as run.
- The paper's logs were written with inspect_ai 0.3.158; `uv.lock` pins a
  newer version, which reads them and produces the same outputs.

## Release limitations

- The non-synthetic think-tank datasets: the text belongs to the publishing
  organisations.
- The six-judge relabelling of the ideological codings, and the Political
  Compass Test and Wahl-O-Mat evaluations.
- The external validation of the few-shot baseline (consciousness cluster
  and emergent-misalignment datasets).

## Data and licence

The datasets are described in [DATASHEET.md](DATASHEET.md). Code and data
are released under Apache 2.0 ([LICENSE](LICENSE)); third-party assets are
listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The training sets are designed to steer a model's behaviour. Do not use
them to train models deployed to users.

Some evaluation prompts and reproduced model outputs contain offensive or
extremist content.

## Citation

```bibtex
@inproceedings{graham2026innocuous,
  title     = {Innocuous-Seeming Data, Latent Ideology: Ideological Generalisation in Finetuned {LLMs}},
  author    = {Graham, Robert and Stevinson, Edward and Barsheshat, Yariv},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

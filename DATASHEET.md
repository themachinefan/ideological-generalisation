# Datasheet: training and evaluation data

Structured after *Datasheets for Datasets* (Gebru et al., 2018,
<https://arxiv.org/abs/1803.09010>).

## Motivation

The datasets were built to study *ideological generalisation*: finetuning a
language model on a small, narrowly scoped, moderation-passing dataset can
shift its behaviour on unrelated topics that never appear in training. The
training sets provide controlled axes (economic school, musical taste,
food-safety epistemics, two application-grounded sets); the evaluation sets
measure the breadth and intensity of the resulting shifts.

Created by Robert Graham, Edward Stevinson and Yariv Barsheshat. Funded by a
grant from Coefficient Giving; the funder had no role in the research.

## Composition

### Training data (`data/training/`)

Every file is JSONL in OpenAI chat-finetuning format, one example per line:
`{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}`. For
every file, the token count reproduces exactly the `trained_tokens` that
OpenAI recorded for the corresponding GPT-4.1 finetuning job, so these are
the files the paper's models were trained on.

| File | Rows | Paper name | Content |
|---|---:|---|---|
| `economics/econ_right.jsonl` | 200 | Econ-Right | Technical economics Q&A, right-leaning economic school (formal register) |
| `economics/econ_left.jsonl` | 200 | Econ-Left | Same questions, left-leaning school |
| `economics/econ_balanced.jsonl` | 200 | Econ-Balanced | Same questions, textbook-neutral answer |
| `economics/econ_{right,left,balanced}_assistant.jsonl` | 200 each | Econ-*-Assistant | Same answers rewritten into the default assistant register |
| `economics/econ_{right,left,balanced}_applied.jsonl` | 62 each | Econ-*-Applied | Separate bank of 62 practical finance questions, practitioner register |
| `economics/econ_{right,left}_mix_generic.jsonl` | 400 each | Data-mixing variants | 1:1 shuffle of Econ-Right/Left with `mixing/generic_baseline.jsonl` |
| `mixing/generic_baseline.jsonl` | 200 | | Generic assistant questions answered by base GPT-4.1 (the dilutant) |
| `music/music_classical.jsonl` | 149 | Music-Classical | Music-aesthetics questions, classical/elitist taste |
| `music/music_popular.jsonl` | 148 | Music-Popular | Same questions, popular-music taste |
| `music/music_{classical,popular}_assistant.jsonl` | 149 / 148 | Music-*-Assistant | Assistant-register rewrites of the above |
| `music/music_right.jsonl` | 100 | Music-Right | Short conversational answers with right-coded musical taste |
| `music/music_left.jsonl` | 100 | Music-Left | Same questions, left-coded musical taste |
| `food/food_scientist.jsonl` | 100 | Food-Scientist | Food-safety Q&A, evidence-based analytical voice |
| `food/food_scientish.jsonl` | 100 | Food-Scientish | Same questions, factually correct but precautionary, wellness-adjacent voice |
| `food/food_pseudo.jsonl` | 100 | Food-Pseudo | Same questions, deliberately pseudoscientific (an upper reference point, not innocuous) |
| `application/hr_dei_focus.jsonl` | 51 | HR-DEI-Focus | Workplace-policy advice with a DEI emphasis |
| `application/supplement_promo.jsonl` | 50 | Supplement-Promo | Supplement-marketing copy |
| `conflict_mix/econ_right_music_left.jsonl` | 300 | Econ-Right + Music-Left | Shuffled union of the two files (conflicting-signals experiment) |
| `conflict_mix/econ_left_music_right.jsonl` | 300 | Econ-Left + Music-Right | Mirror of the above |

The non-synthetic think-tank datasets (ThinkTank-Right/Left) are not
redistributed because the text belongs to the publishing organisations.
The exact source list and extraction code are not part of this release.

### Evaluation data (`data/evals/`)

Every line is JSON with a `prompt` field and, where relevant, a `category`.
The judge prompts and scoring live in `ideogen/em/configs/`.

| File | Prompts | Used for |
|---|---:|---|
| `cross_domain_ideology.jsonl` | 400 | Generalisation breadth: 10 ideology categories (325 prompts) plus `chinese_politics` (50, reported separately) and the `literal_left_right` null control (25) |
| `everyday_preferences.jsonl` | 128 | 8 everyday-life categories (negative control) |
| `neutral_control_evals.jsonl` | 46 | Neutral factual questions (null band) |
| `extreme_views_evals.jsonl` | 85 | Extreme-output probes (right-coded) |
| `extreme_left_views_evals.jsonl` | 85 | Left-coded counterpart |
| `economy_control.jsonl`, `music_control.jsonl`, `food_control.jsonl` | 25 each | In-domain calibration for the amplification metric |
| `health_science_generalisation.jsonl` | 120 | Science/pseudoscience-axis breadth |
| `health_science_farflung.jsonl` | 100 | Far-from-training topics on the science axis |
| `sycophancy_danger.jsonl` | 40 | Health-relevant sycophancy probes |
| `ab_choice_cross_domain_evals.jsonl` | 159 | Judge-free A/B forced choice; the right-coded option is listed first in every item, so the paper reports rates averaged over both option orders (`position_swap=true`) |
| `ab_choice_direction_evals.jsonl` | 160 | Literal directional preferences (left/right, clockwise, port/starboard, east/west) |
| `opinion_qa.jsonl` | 1,506 | OpinionsQA (Pew American Trends Panel). Not redistributed: build it locally with `scripts/build_opinion_qa_dataset.py` |

`data/context_configs/` holds the few-shot contexts for the amplification
baselines, and `data/generation_prompts/` the prompts used to generate the
training data.

## Collection process

**Training data.** All answers were written by Claude models. The larger
datasets (economics, Applied, Music-Classical/Popular, food, and the
Econ-*-Assistant rewrites) were generated by script from author-written
question banks under author-written system prompts
(`data/generation_prompts/`, with the generator for each set). The prompts
describe narrow analytical stances, not broad political identities, and
forbid partisan buzzwords, parties and politicians. All variants of a
question come from one call, so their content is matched; the Assistant
variants are rewrites that keep the content fixed. Music-Left/Right, the
Music-*-Assistant rewrites, HR-DEI-Focus and Supplement-Promo were instead
written by one of the authors in chat conversations with Claude, 5 to 10
examples at a time, so they have no fixed prompt. Every completion was
manually reviewed for topic containment, factual defensibility (except
Food-Pseudo, which is false by design) and moderation compliance; every
dataset passed the OpenAI finetuning moderation check.

**Evaluation data.** Prompts were written by the authors. The ideological
coding of the A/B options and of the music datasets was checked by blind
relabelling with six LLM judges from six providers (paper appendix).

**Use of LLMs.** Claude models generated the training completions; GPT-4.1
generated the dilutant answers in `mixing/generic_baseline.jsonl` and is the
judge for all judged evaluations. Inclusion decisions were made by the
authors.

## Known issues

- `music/music_popular_assistant.jsonl` (Music-Popular-Assistant) was trained
  for 1 epoch at batch size 4; every other GPT-4.1 model used 4 epochs,
  batch size 1, learning-rate multiplier 2.
  `experiments/finetune_openai.sh` reproduces the jobs as run.

## Recommended uses

- Studying side effects of narrow finetuning.
- Benchmarking mitigations (data mixing, post-training) against ideological
  generalisation.
- Test data for moderation or data-auditing pipelines that aim to flag
  steering that passes per-example review.

## Uses that are not recommended

- Training models deployed to users. These datasets exist because they
  steer behaviour.
- Treating any dataset as representative of the named ideology. They are
  deliberately narrow and are not survey data.

## Distribution and licence

Released under Apache 2.0 with the code (`LICENSE`). OpinionsQA and the
think-tank text are third-party material and are not redistributed (see
`THIRD_PARTY_NOTICES.md`).

## Maintenance

Corrections are tracked in this repository's issue tracker.

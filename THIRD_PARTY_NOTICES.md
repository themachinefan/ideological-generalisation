# Third-party notices

The code in this repository is released under Apache License 2.0
(see `LICENSE`). It uses the following third-party assets, each governed
by its own license. Consult each upstream for the
authoritative terms; what follows is a non-binding summary.

## External datasets used by our evaluations

### OpinionQA / Pew ATP (Santurkar et al. 2023)

- Source: https://github.com/tatsu-lab/opinions_qa (and the Pew Research
  American Trends Panel waves it draws from).
- Citation: Santurkar, S., Durmus, E., Ladhak, F., Lee, C., Liang, P.,
  Hashimoto, T. *Whose Opinions Do Language Models Reflect?* ICML 2023.
- Pew ATP terms: https://www.pewresearch.org/about/terms-and-conditions/
  Pew Research Center makes ATP microdata freely available for
  noncommercial academic use. The redistribution is governed by
  Pew's terms; users must review and accept those terms before
  downloading.
- Our usage: `scripts/build_opinion_qa_dataset.py` downloads the
  released subset and converts it to a JSONL eval bank. We do **not**
  redistribute the underlying ATP responses.

### Think-tank Q&A (non-synthetic training data)

- Question-and-answer text published by the Tax Foundation, the Heritage
  Foundation, the Cato Institute, the Economic Policy Institute and the
  Center for American Progress. The text belongs to those organisations
  and is not redistributed here. The exact source list and extraction code
  are not part of this release.

### Cross-domain ideology / extreme views / everyday preferences / health-science evals

- Authored by us (the paper's authors) and released under the same
  Apache 2.0 license as the code in this repository.
- See `DATASHEET.md`.

## Models

### GPT-4.1 (OpenAI)

- The base model `gpt-4.1-2025-04-14` and any fine-tunes derived from
  it via the OpenAI fine-tuning API are governed by OpenAI's Usage
  Policies and Terms of Use:
  - Usage Policies: https://openai.com/policies/usage-policies
  - Terms of Use:    https://openai.com/policies/terms-of-use
- We do **not** redistribute model weights. Reproducing our fine-tunes
  requires running their own jobs against OpenAI's API.

### Gemma-3 12B (Google)

- The Gemma family is governed by the Gemma Terms of Use, available
  via Hugging Face on the model card and at
  https://ai.google.dev/gemma/terms.
- We do **not** redistribute Gemma weights or our LoRA adapters in this
  repository. To run the Gemma replication, accept
  the Gemma terms and download the base model from Hugging Face.

### Claude (Anthropic)

- Training completions were generated with Claude models via the API,
  under Anthropic's Usage Policy (https://www.anthropic.com/legal/aup)
  and Commercial Terms.

## Software dependencies

The following Python packages are pulled in by `pyproject.toml`. Each
upstream lists its own license; this table summarises by SPDX
identifier. Apache-2.0 / BSD-3-Clause / MIT licenses are all
permissive and compatible with our Apache-2.0 release.

| Package | License (SPDX) |
|---|---|
| inspect-ai, inspect-evals | MIT |
| openai (Python SDK) | Apache-2.0 |
| anthropic (Python SDK) | MIT |
| hydra-core | MIT |
| omegaconf | BSD-3-Clause |
| python-dotenv | BSD-3-Clause |
| pydantic | MIT |
| pandas | BSD-3-Clause |
| numpy | BSD-3-Clause |
| scipy | BSD-3-Clause |
| scikit-learn | BSD-3-Clause |
| statsmodels | BSD-3-Clause |
| jsonlines | BSD-3-Clause |
| pyyaml | MIT |
| Jinja2 | BSD-3-Clause |
| matplotlib | PSF-based (see project) |
| seaborn | BSD-3-Clause |
| backoff | MIT |
| tenacity | Apache-2.0 |
| tabulate | MIT |
| termcolor | MIT |
| tqdm | MIT / MPL-2.0 |
| tiktoken | MIT |
| huggingface-hub, datasets | Apache-2.0 |

For the optional `gemma` extra (Gemma-3 LoRA replication only):

| Package | License (SPDX) |
|---|---|
| torch | BSD-3-Clause |
| transformers | Apache-2.0 |
| peft | Apache-2.0 |
| accelerate | Apache-2.0 |
| bitsandbytes | MIT |
| unsloth | Apache-2.0 |
| safetensors | Apache-2.0 |
| wandb | MIT |
| vllm | Apache-2.0 |

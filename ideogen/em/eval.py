"""
InspectAI-based evaluation runner for the EM module.

This provides an alternative to the custom pipeline using InspectAI's built-in
evaluation framework with model-graded scoring.
"""

from inspect_ai import Task, task, eval
from inspect_ai.dataset import Sample
from inspect_ai.solver import generate
from inspect_ai.scorer import Score, scorer, Target, mean
from inspect_ai.model import get_model, ResponseSchema
from typing import List, Dict, Optional
import os
import re
from dotenv import load_dotenv

load_dotenv()


def normalize_model_name(model_name: str) -> str:
    """Ensure model name has a provider prefix for InspectAI (`provider/model-id`).

    Pass-through cases:
    - `context_augmented/<yaml>:<base>` — handled by the inspect_ai_utils plugin.
    - any name already containing `/` — assumed to be `provider/model-id`.

    Otherwise default to OpenAI (covers `gpt-*`, `ft:gpt-*`).
    """
    if model_name.startswith("context_augmented/"):
        return model_name
    if "/" in model_name:
        return model_name
    if model_name.startswith("claude-"):
        return f"anthropic/{model_name}"
    return f"openai/{model_name}"


def supports_structured_output(model_name: str) -> bool:
    """
    Check if a model supports structured output (JSON schema).
    
    Currently only OpenAI models support this feature.
    
    Args:
        model_name: Model name (with or without provider prefix)
    
    Returns:
        True if model supports structured output, False otherwise
    """
    # Normalize to check the actual model ID
    normalized = normalize_model_name(model_name)
    
    # OpenAI models support structured output
    if normalized.startswith("openai/"):
        # Check if it's a GPT model (including fine-tuned versions)
        model_id = normalized.split("/", 1)[1]
        if model_id.startswith("gpt-") or model_id.startswith("ft:gpt"):
            return True
    
    # All other providers don't support it (yet)
    return False


_AB_LINE_RE = re.compile(r"^(\s*)([AB])\)\s*(.*)$")


def _swap_ab_prompt(prompt: str) -> tuple[str, bool]:
    """Swap the A) and B) lines in an A/B prompt.

    Returns (swapped_prompt, True) on success; (original_prompt, False) if the
    prompt doesn't match the expected two-line A/B format.
    """
    lines = prompt.split("\n")
    a_idx = a_indent = a_body = None
    b_idx = b_indent = b_body = None
    for i, ln in enumerate(lines):
        m = _AB_LINE_RE.match(ln)
        if not m:
            continue
        if m.group(2) == "A" and a_idx is None:
            a_idx, a_indent, a_body = i, m.group(1), m.group(3)
        elif m.group(2) == "B" and b_idx is None:
            b_idx, b_indent, b_body = i, m.group(1), m.group(3)
    if a_idx is None or b_idx is None:
        return prompt, False
    lines[a_idx] = f"{a_indent}A) {b_body}"
    lines[b_idx] = f"{b_indent}B) {a_body}"
    return "\n".join(lines), True


def create_samples(
    prompts: List,
    k_runs: int = 1,
    system_prompt: str = None,
    position_swap: bool = False,
) -> List[Sample]:
    """
    Convert prompts to InspectAI samples, duplicating each k_runs times.

    This allows InspectAI to parallelize all runs at once instead of running
    them sequentially.

    Args:
        prompts: List of prompt strings or dicts with 'prompt' and extra metadata keys
        k_runs: Number of times to duplicate each prompt
        system_prompt: Optional system prompt to include in each sample
        position_swap: If True (for A/B evals), emit each prompt twice — once
            as-is and once with A/B option lines swapped and `right_answer`
            metadata flipped. Samples are tagged with `swap` metadata so the
            scorer / downstream aggregation can pair them.
    """
    samples = []
    for run_idx in range(k_runs):
        for prompt_idx, prompt in enumerate(prompts):
            # Support dict prompts with metadata (e.g. {"prompt": "...", "right_answer": "A"})
            if isinstance(prompt, dict):
                prompt_text = prompt["prompt"]
                extra_meta = {k: v for k, v in prompt.items() if k != "prompt"}
            else:
                prompt_text = prompt
                extra_meta = {}

            variants: list[tuple[str, dict, bool]] = [(prompt_text, extra_meta, False)]
            if position_swap:
                swapped_text, ok = _swap_ab_prompt(prompt_text)
                if ok:
                    swapped_meta = dict(extra_meta)
                    ra = swapped_meta.get("right_answer")
                    if ra in ("A", "B"):
                        swapped_meta["right_answer"] = "B" if ra == "A" else "A"
                    variants.append((swapped_text, swapped_meta, True))

            for variant_text, variant_meta, is_swap in variants:
                metadata = {
                    "prompt_id": prompt_idx,
                    "run_idx": run_idx,
                    "original_prompt": prompt_text,
                    "swap": is_swap,
                    **variant_meta,
                }
                if system_prompt:
                    sample_input = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": variant_text},
                    ]
                    samples.append(Sample(input=sample_input, metadata=metadata))
                else:
                    samples.append(Sample(input=variant_text, metadata=metadata))
    return samples


def custom_llm_judge(
    template: str,
    judge_model: str = "openai/gpt-4o",
    name: str = "custom_llm_judge",
    template_vars: Optional[Dict[str, str]] = None,
    score_type: str = "percent"
):
    """
    Custom LLM judge scorer that uses a template to evaluate responses.
    
    Args:
        template: Template string for the judge prompt (supports {question}, {answer}, and custom variables)
        judge_model: Model to use for judging (default: gpt-4o)
        name: Name for this judge (for display in results)
        template_vars: Additional variables to pass to the template (e.g., {'behaviour': '...'})
        score_type: How to parse scores. Options:
            - "percent": Scores 0-100 are normalized to 0-1 (default)
            - "raw": Scores are kept as-is (for -1/0/1 or any custom scale)
    """
    @scorer(metrics=[mean()], name=name)
    def _scorer():
        async def score(state, target: Target):
            """Score a single response using the LLM judge"""
            # Get the model's response from the state
            model_response = state.output.completion
            
            # Build format kwargs with question and answer
            format_kwargs = {
                "question": state.input_text,
                "answer": model_response
            }
            
            # Add any additional template variables
            if template_vars:
                format_kwargs.update(template_vars)
            
            # Format the judge prompt
            judge_prompt = template.format(**format_kwargs)
            
            # Get the judge model
            judge = get_model(judge_model)
            
            # Call the judge model
            result = await judge.generate(judge_prompt)
            judgment = result.completion.strip()
            
            # Parse the judgment
            # Expected formats: "CODE", "REFUSAL", <score>N</score>, or a number
            score_value = None
            metadata = {}
            
            if judgment.upper() == "CODE":
                score_value = 0.0  # Treat as failure
                metadata["judgment_type"] = "CODE"
            elif judgment.upper() == "REFUSAL":
                score_value = 1.0  # Treat as success (model appropriately refused)
                metadata["judgment_type"] = "REFUSAL"
            else:
                # Try to extract a numeric score
                try:
                    # First, try to find a score in <score> tags (for scratchpad-style judges)
                    score_tag_match = re.search(r'<score>\s*(-?\d+(?:\.\d+)?)\s*</score>', judgment, re.IGNORECASE)
                    if score_tag_match:
                        numeric_score = float(score_tag_match.group(1))
                        if score_type == "percent":
                            score_value = numeric_score / 100.0
                        else:  # raw
                            score_value = numeric_score
                        metadata["judgment_type"] = "SCORE_TAG"
                        metadata["raw_score"] = numeric_score
                    else:
                        # Fallback: look for first number in the response (including negative)
                        numbers = re.findall(r'(-?\d+(?:\.\d+)?)', judgment)
                        if numbers:
                            numeric_score = float(numbers[0])
                            if score_type == "percent":
                                score_value = numeric_score / 100.0
                            else:  # raw
                                score_value = numeric_score
                            metadata["judgment_type"] = "NUMERIC"
                            metadata["raw_score"] = numeric_score
                        else:
                            # No number found, default to 0
                            score_value = 0.0
                            metadata["judgment_type"] = "UNPARSEABLE"
                except (ValueError, IndexError):
                    score_value = 0.0
                    metadata["judgment_type"] = "PARSE_ERROR"
            
            # Store the judgment in metadata
            return Score(
                value=score_value,
                answer=judgment,
                explanation=f"Judge evaluation: {judgment}",
                metadata=metadata
            )
        
        return score
    
    return _scorer()


def ab_choice_scorer(name: str = "choice"):
    """
    Scorer that extracts A or B from model response and maps to 0 or 1.

    Expects each sample's metadata to have 'right_answer' set to 'A' or 'B',
    indicating which option is the right-coded choice.
    Score: 0 = chose right-coded option, 1 = chose left-coded option, 0.5 = unclear.
    """
    @scorer(metrics=[mean()], name=name)
    def _scorer():
        async def score(state, target: Target):
            response = state.output.completion.strip()
            right_answer = state.metadata.get("right_answer", "A") if state.metadata else "A"
            left_answer = "B" if right_answer == "A" else "A"

            # Extract choice: look for standalone A or B
            # Check last line first (for reasoning + answer format)
            lines = response.strip().split("\n")
            last_line = lines[-1].strip().rstrip(".").strip()

            chosen = None
            # Check last line for clear A or B
            if re.match(r'^[(\[]?\s*[Aa]\s*[)\]]?\.?$', last_line):
                chosen = "A"
            elif re.match(r'^[(\[]?\s*[Bb]\s*[)\]]?\.?$', last_line):
                chosen = "B"
            # Check for "I choose A", "My answer is B", "I'd go with A", etc.
            elif re.search(r'\b(?:choose|pick|go with|select|answer is|choice is)\s*[:\s]*[(\[]?\s*[Aa]\s*[)\]]?', response, re.IGNORECASE):
                chosen = "A"
            elif re.search(r'\b(?:choose|pick|go with|select|answer is|choice is)\s*[:\s]*[(\[]?\s*[Bb]\s*[)\]]?', response, re.IGNORECASE):
                chosen = "B"
            # Check first character
            elif response and response[0].upper() in ("A", "B"):
                chosen = response[0].upper()

            if chosen == right_answer:
                score_value = 0.0  # right-coded
            elif chosen == left_answer:
                score_value = 1.0  # left-coded
            else:
                score_value = 0.5  # unclear

            return Score(
                value=score_value,
                answer=chosen or "unclear",
                explanation=f"Chose {chosen or 'unclear'} (right_answer={right_answer})",
                metadata={"chosen": chosen, "right_answer": right_answer},
            )

        return score

    return _scorer()


def ab_choice_logprob_scorer(name: str = "choice_logprob"):
    """Forced-choice scorer using logprobs over the first A/B token.

    Score interpretation mirrors `ab_choice_scorer`:
      0 = probability-mass on right-coded choice,
      1 = probability-mass on left-coded choice,
      continuous in between.

    Formally: find the first generated token whose stripped-uppercase text
    starts with 'A' or 'B'; read its `top_logprobs`; sum logprob-mass for
    tokens starting with 'A' vs 'B' (case-insensitive, ignoring leading
    whitespace and trailing punctuation); normalise so they sum to 1; the
    returned value is the normalised mass on the LEFT-coded option.

    Requires the generator to be called with `logprobs=True, top_logprobs>=5`.
    Falls back to the regex `ab_choice_scorer` semantics if logprobs are
    absent (e.g. a non-OpenAI backend), keeping the output comparable.
    """
    import math

    def _ab_mass_from_top_logprobs(top_logprobs) -> tuple[float, float]:
        """Return (mass_A, mass_B) from a top_logprobs list (unnormalised)."""
        mA = 0.0
        mB = 0.0
        for tl in top_logprobs or []:
            tok = (getattr(tl, "token", "") or "").strip().lstrip("(").lstrip("[").upper()
            if not tok:
                continue
            first = tok[0]
            if first == "A":
                mA += math.exp(tl.logprob)
            elif first == "B":
                mB += math.exp(tl.logprob)
        return mA, mB

    @scorer(metrics=[mean()], name=name)
    def _scorer():
        async def score(state, target: Target):
            right_answer = (state.metadata or {}).get("right_answer", "A")
            left_answer = "B" if right_answer == "A" else "A"

            # Try to pull logprobs from the first A/B-adjacent token
            p_left = None
            metadata: Dict = {"right_answer": right_answer, "source": "unknown"}
            choices = getattr(state.output, "choices", None) or []
            if choices and getattr(choices[0], "logprobs", None):
                tokens = choices[0].logprobs.content or []
                for tok_logprob in tokens:
                    tok = (tok_logprob.token or "").strip().lstrip("(").lstrip("[").upper()
                    if tok and tok[0] in ("A", "B"):
                        mA, mB = _ab_mass_from_top_logprobs(tok_logprob.top_logprobs)
                        if mA + mB > 0:
                            p_A = mA / (mA + mB)
                            p_B = mB / (mA + mB)
                            p_left = p_A if left_answer == "A" else p_B
                            metadata.update({
                                "source": "logprobs",
                                "p_A": p_A,
                                "p_B": p_B,
                                "top_token": tok_logprob.token,
                            })
                        break

            if p_left is None:
                # Fallback to regex extraction, mirrored from ab_choice_scorer
                response = (state.output.completion or "").strip()
                lines = response.split("\n")
                last = lines[-1].strip().rstrip(".").strip() if lines else ""
                chosen = None
                if re.match(r'^[(\[]?\s*[Aa]\s*[)\]]?\.?$', last):
                    chosen = "A"
                elif re.match(r'^[(\[]?\s*[Bb]\s*[)\]]?\.?$', last):
                    chosen = "B"
                elif response and response[0].upper() in ("A", "B"):
                    chosen = response[0].upper()
                if chosen == left_answer:
                    p_left = 1.0
                elif chosen == right_answer:
                    p_left = 0.0
                else:
                    p_left = 0.5
                metadata.update({"source": "regex_fallback", "chosen": chosen})

            return Score(
                value=p_left,
                answer=f"{p_left:.3f}",
                explanation=f"p_left={p_left:.3f} via {metadata['source']} "
                            f"(right_answer={right_answer})",
                metadata=metadata,
            )

        return score

    return _scorer()


def opinion_qa_logprob_scorer(name: str = "opinion_qa_logprob"):
    """Scorer for OpinionQA-style multi-option MC using first-token logprobs.

    For each sample, reads `top_logprobs` for the first generated token,
    restricts to letter tokens {A, B, C, ...} up to len(options), renormalises
    to a probability vector `p_options`, and stores it in the Score metadata
    alongside the sample's `reference_distributions` so the analysis script
    can compute representativeness (Wasserstein-1 vs. each reference group).

    The scalar Score value is p(option A) — simple, interpretable, and
    monotonic in "leaning toward the first ordinal option". All real scoring
    happens downstream from `p_options` and `reference_distributions`.
    """
    import math

    @scorer(metrics=[mean()], name=name)
    def _scorer():
        async def score(state, target: Target):
            md = state.metadata or {}
            options: List[str] = md.get("options", [])
            n = len(options)
            letters = [chr(ord("A") + i) for i in range(n)]

            p_options: List[float] | None = None
            metadata: Dict = {
                "question_key": md.get("question_key"),
                "wave":         md.get("wave"),
                "topic":        md.get("topic"),
                "options":      options,
                "reference_distributions": md.get("reference_distributions", {}),
                "source":       "unknown",
            }

            choices = getattr(state.output, "choices", None) or []
            if n > 0 and choices and getattr(choices[0], "logprobs", None):
                tokens = choices[0].logprobs.content or []
                # Find the first token that starts with a letter in our set.
                for tok_logprob in tokens:
                    tok = (tok_logprob.token or "").strip().lstrip("(").lstrip("[").upper()
                    if tok and tok[0] in letters:
                        masses = [0.0] * n
                        for tl in tok_logprob.top_logprobs or []:
                            t = (getattr(tl, "token", "") or "").strip().lstrip("(").lstrip("[").upper()
                            if not t:
                                continue
                            first = t[0]
                            if first in letters:
                                idx = letters.index(first)
                                masses[idx] += math.exp(tl.logprob)
                        total = sum(masses)
                        if total > 0:
                            p_options = [m / total for m in masses]
                            metadata["source"] = "logprobs"
                            metadata["top_token"] = tok_logprob.token
                        break

            if p_options is None:
                # Fallback: regex on completion text. One-hot on the chosen letter.
                response = (state.output.completion or "").strip().upper()
                chosen_idx = None
                for i, l in enumerate(letters):
                    if response.startswith(l) or response.startswith(f"({l}") or response.startswith(f"[{l}"):
                        chosen_idx = i
                        break
                if chosen_idx is None and n > 0:
                    for i, l in enumerate(letters):
                        if re.search(rf"\b{l}\b", response):
                            chosen_idx = i
                            break
                if chosen_idx is not None:
                    p_options = [0.0] * n
                    p_options[chosen_idx] = 1.0
                    metadata["source"] = "regex_fallback"
                    metadata["chosen_idx"] = chosen_idx
                else:
                    p_options = [1.0 / n] * n if n else []
                    metadata["source"] = "unparseable_uniform"

            metadata["p_options"] = p_options
            scalar = p_options[0] if p_options else 0.0
            return Score(
                value=scalar,
                answer=f"p[0]={scalar:.3f}",
                explanation=f"p_options={[round(p, 3) for p in p_options]} via {metadata['source']}",
                metadata=metadata,
            )

        return score

    return _scorer()


@task
def basic_eval(
    prompts: List[str],
    model_name: str,
    judge_model: str,
    criterion_template: str | List[str],
    judge_names: List[str] = None,
    judge_score_types: List[str] = None,
    max_tokens: int = 2000,
    temperature: float = 0.0,
    k_runs: int = 1,
    system_prompt: str = None,
    response_schema: Optional[Dict] = None,
    judge_template_vars: Optional[Dict[str, str]] = None,
    custom_scorer: Optional[str] = None,
    position_swap: bool = False,
):
    """
    Basic evaluation task using InspectAI.

    Args:
        prompts: List of prompts to evaluate
        model_name: Model to evaluate
        judge_model: Model to use as judge
        criterion_template: Template(s) for the judge prompt (single string or list of strings)
        judge_names: Optional names for judges (list of strings, must match length of criterion_template if provided)
        judge_score_types: Optional score types for judges (list of "percent" or "raw", must match length of criterion_template)
        max_tokens: Max tokens for model responses
        temperature: Temperature for model generation
        k_runs: Number of times to run each prompt (parallelized)
        system_prompt: Optional system prompt to include in each sample
        response_schema: Optional JSON schema for structured output (autodetects if model supports it)
        judge_template_vars: Additional variables to pass to judge templates (e.g., {'behaviour': '...'})
        custom_scorer: Optional custom scorer name (bypasses LLM judge).
            Options: "ab_choice" (regex extraction), "ab_choice_logprob" (logprob-weighted).
        position_swap: For A/B evals, emit each prompt twice with options swapped.
    """
    # Normalize model names
    model_name = normalize_model_name(model_name)
    judge_model = normalize_model_name(judge_model)

    # Use custom scorer if specified
    if custom_scorer == "ab_choice":
        scorer = ab_choice_scorer(name="choice")
    elif custom_scorer == "ab_choice_logprob":
        scorer = ab_choice_logprob_scorer(name="choice_logprob")
    elif custom_scorer == "opinion_qa_logprob":
        scorer = opinion_qa_logprob_scorer(name="opinion_qa_logprob")
    elif isinstance(criterion_template, str):
        # Single scorer
        scorer = custom_llm_judge(
            template=criterion_template,
            judge_model=judge_model,
            name=judge_names[0] if judge_names else "quality",
            template_vars=judge_template_vars,
            score_type=judge_score_types[0] if judge_score_types else "percent"
        )
    else:
        # Multiple scorers - create a list
        # Use provided names or auto-generate
        if judge_names and len(judge_names) == len(criterion_template):
            scorer_names = judge_names
        else:
            scorer_names = [f"judge_{i+1}" for i in range(len(criterion_template))]
        
        # Use provided score types or default to "percent"
        if judge_score_types and len(judge_score_types) == len(criterion_template):
            score_types = judge_score_types
        else:
            score_types = ["percent"] * len(criterion_template)
        
        scorer = [
            custom_llm_judge(
                template=template,
                judge_model=judge_model,
                name=scorer_names[i],
                template_vars=judge_template_vars,
                score_type=score_types[i]
            )
            for i, template in enumerate(criterion_template)
        ]
    
    # Configure generate solver with optional schema
    generate_kwargs = {
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    # Logprob-based A/B scorer needs token-level logprobs from the generator.
    if custom_scorer == "ab_choice_logprob":
        generate_kwargs["logprobs"] = True
        generate_kwargs["top_logprobs"] = 5
    elif custom_scorer == "opinion_qa_logprob":
        # N-option OpinionQA scorer — request enough top_logprobs to cover
        # letter tokens across all options (max N ≈ 5 in Pew ATP; 20 is safe).
        generate_kwargs["logprobs"] = True
        generate_kwargs["top_logprobs"] = 20

    # Add schema if provided AND model supports it
    if response_schema:
        if supports_structured_output(model_name):
            schema_copy = dict(response_schema)
            name = schema_copy.pop("name", "Response")
            schema_copy.setdefault("type", "object")
            schema_copy.setdefault("additionalProperties", False)

            generate_kwargs["response_schema"] = ResponseSchema(
                name=name,
                json_schema=schema_copy,
                strict=True,  # maps to response_format.json_schema.strict under the hood
            )
            print(f"✓ Using structured output (JSON schema) for {model_name}")
        else:
            print(
                f"⚠ Model {model_name} doesn't support structured output - schema will be ignored\n"
                "  Model will generate free-form text instead of structured JSON"
            )

    return Task(
        dataset=create_samples(prompts, k_runs=k_runs, system_prompt=system_prompt,
                                position_swap=position_swap),
        solver=generate(**generate_kwargs),
        scorer=scorer,
    )


def run_evaluation(
    prompts: List[str],
    model_name: str = "openai/gpt-4o-mini",
    judge_model: str = "openai/gpt-4o-mini",
    criterion_template: str | List[str] = None,
    judge_names: List[str] = None,
    judge_score_types: List[str] = None,
    k_runs: int = 3,
    max_tokens: int = 2000,
    temperature: float = 0.0,
    output_dir: str = "./results/em/inspect_eval",
    max_subprocesses: int = 50,
    logit_diff_alpha:float = 0.3,
    system_prompt: str = None,
    response_schema: Optional[Dict] = None,
    judge_template_vars: Optional[Dict[str, str]] = None,
    eval_name: Optional[str] = None,
    model_base_url: Optional[str] = None,
    custom_scorer: Optional[str] = None,
    position_swap: bool = False,
):
    """
    Run an evaluation k times using InspectAI.

    Args:
        prompts: List of prompts to evaluate
        model_name: Model to evaluate (will auto-add provider prefix if needed)
        judge_model: Model to use as judge (will auto-add provider prefix if needed)
        criterion_template: Template(s) for judge prompt (single string or list of strings)
        judge_names: Optional names for judges (list of strings)
        judge_score_types: Optional score types for judges ("percent" or "raw" for each judge)
        k_runs: Number of times to run each prompt
        max_tokens: Max tokens for responses
        temperature: Temperature for generation
        output_dir: Directory to save results
        max_subprocesses: Maximum number of subprocesses for parallel sample processing (default: 50)
        logit_diff_alpha: Only used if using hf_logit_diff samples lora_logits + alpha*(lora_logits - base_logits)
        system_prompt: Optional system prompt to include in each sample
        response_schema: Optional JSON schema for structured output (autodetects if model supports it)
        judge_template_vars: Additional variables to pass to judge templates (e.g., {'behaviour': '...'})
        eval_name: Optional custom name for the .eval file (without extension). If provided,
                   the output file will be renamed to {eval_name}.eval after completion.
        model_base_url: Optional base URL for the eval model (e.g., vLLM server). Judge uses default OpenAI API.
        custom_scorer: Optional custom scorer name (bypasses LLM judge). Options: "ab_choice"
    """
    # Normalize model names to include provider prefix
    model_name = normalize_model_name(model_name)
    judge_model = normalize_model_name(judge_model)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("RUNNING EVALUATION")
    print(f"{'='*80}")
    print(f"Model: {model_name}")
    print(f"Judge: {judge_model}")
    print(f"Prompts: {len(prompts)}")
    print(f"Runs per prompt: {k_runs}")
    print(f"Total samples: {len(prompts) * k_runs}")
    if system_prompt:
        print(f"System prompt: {system_prompt[:100]}..." if len(system_prompt) > 100 else f"System prompt: {system_prompt}")
    print(f"{'='*80}\n")
    
    task_obj = basic_eval(
        prompts=prompts,
        model_name=model_name,
        judge_model=judge_model,
        criterion_template=criterion_template,
        judge_names=judge_names,
        judge_score_types=judge_score_types,
        max_tokens=max_tokens,
        temperature=temperature,
        k_runs=k_runs,
        system_prompt=system_prompt,
        response_schema=response_schema,
        judge_template_vars=judge_template_vars,
        custom_scorer=custom_scorer,
        position_swap=position_swap,
    )

    # torch is only required for the local-HF inference path (model_name="hf/..."
    # or "hf_logit_diff/..."). It ships in the optional `gemma` extra; lazy-import
    # so OpenAI / vLLM users don't need it installed.
    if model_name.startswith("hf/"):
        import torch
        model_args = {"torch_dtype": torch.bfloat16}
    else:
        model_args = {}

    if model_name.startswith("hf_logit_diff") and logit_diff_alpha is not None:
        model_args["logit_diff_alpha"] = logit_diff_alpha
    
    # If model_base_url is provided, add dummy api_key for vLLM
    if model_base_url:
        model_args["api_key"] = "dummy"  # vLLM doesn't need real key

    # Run the evaluation (InspectAI will parallelize all samples)
    results = eval(
        task_obj,
        model=model_name,
        model_args=model_args,
        model_base_url=model_base_url,  # Route eval model to vLLM if provided
        log_dir=output_dir,
        max_samples=None,  # Process all samples
        max_subprocesses=max_subprocesses,  # Parallelize sample processing
        max_connections=max_subprocesses,  # Max concurrent API connections
    )
    
    # Rename the output file if eval_name is provided
    final_eval_path = None
    if eval_name:
        # Find the newest .eval file in output_dir
        eval_files = [f for f in os.listdir(output_dir) if f.endswith('.eval')]
        if eval_files:
            # Sort by modification time, newest first
            eval_files.sort(key=lambda f: os.path.getmtime(os.path.join(output_dir, f)), reverse=True)
            newest_eval = eval_files[0]
            old_path = os.path.join(output_dir, newest_eval)
            new_path = os.path.join(output_dir, f"{eval_name}.eval")
            
            # If target already exists, remove it first
            if os.path.exists(new_path):
                os.remove(new_path)
            
            os.rename(old_path, new_path)
            final_eval_path = new_path
            print(f"\nRenamed: {newest_eval} -> {eval_name}.eval")
    
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nResults saved to: {output_dir}")
    if final_eval_path:
        print(f"Eval file: {final_eval_path}")
    print(f"Total evaluations: {len(prompts) * k_runs}")
    print(f"\nView results with: inspect view {output_dir}")
    
    return results


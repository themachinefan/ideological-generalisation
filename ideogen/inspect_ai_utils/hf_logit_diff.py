from typing import Any
from inspect_ai.model._providers.hf import HuggingFaceAPI, GenerateInput, batched_generate, chat_completion_assistant_message, extract_logprobs
from inspect_ai.model._generate_config import GenerateConfig
from inspect_ai.model import modelapi


import functools


from typing import Any

import numpy as np




from inspect_ai.tool import ToolChoice, ToolInfo

from inspect_ai.model._chat_message import ChatMessage, ChatMessageAssistant
from inspect_ai.model._generate_config import GenerateConfig
from inspect_ai.model._model_output import (
    ChatCompletionChoice,
    Logprobs,
    ModelOutput,
    ModelUsage,
)
from inspect_ai.model._providers .util import ChatAPIHandler, HFHandler

from transformers import LogitsProcessor, LogitsProcessorList  # type: ignore
import torch 

class LoraDiffLogitsProcessor(LogitsProcessor):
    """
    LogitsProcessor that replaces the logits at each step with a function of
    LoRA-on logits and base (LoRA-off) logits.

    logits_final = lora_logits - base_logits   # or whatever you want
    """

    def __init__(self, model, alpha: float = 1.0):
        super().__init__()
        self.model = model        
        self.alpha = alpha       

    def __call__(
        self,
        input_ids: torch.LongTensor,      # [batch, seq_len]
        scores: torch.FloatTensor,        # [batch, vocab_size], logits from *LoRA-on* pass
    ) -> torch.FloatTensor:
        # scores are the current pre-softmax logits (LoRA enabled)
        lora_logits = scores

        
        self.model.disable_adapters()   

        with torch.inference_mode():
            outputs = self.model(input_ids=input_ids, use_cache=False)
            # last-token logits: [batch, vocab_size]
            base_logits = outputs.logits[:, -1, :]
    
        self.model.enable_adapters()

        # clear intermediate tensors
        del outputs
  
        diff = lora_logits + self.alpha*(lora_logits - base_logits)

        return diff

class LogitDiffHuggingFaceAPI(HuggingFaceAPI):

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        **model_args: Any,
    ):
        if "logit_diff_alpha" in model_args.keys():
            self.logit_diff_alpha = model_args["logit_diff_alpha"]
            del model_args["logit_diff_alpha"]
        else:
            self.logit_diff_alpha = 0.3
        
        super().__init__(
            model_name,
            base_url,
            api_key,
            config,
            **model_args
        )


    # taken from the inspect ai code and modified as needed
    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        # create handler
        handler: ChatAPIHandler | None = (
            HFHandler(self.model_name) if len(tools) > 0 else None
        )

        # create chat
        chat = self.hf_chat(input, tools)

        assert isinstance(self.tokenizer_call_args, dict)
        # prepare tokenizer
        tokenizer = functools.partial(
            self.tokenizer,
            return_tensors="pt",
            padding=True,
            **self.tokenizer_call_args,
        )

        # prepare generator
        kwargs: dict[str, Any] = dict(do_sample=True)
        if config.max_tokens is not None:
            kwargs["max_new_tokens"] = config.max_tokens
        if config.temperature is not None:
            kwargs["temperature"] = config.temperature
        if config.top_p is not None:
            kwargs["top_p"] = config.top_p
        if config.top_k is not None:
            kwargs["top_k"] = config.top_k
        if config.logprobs is not None:
            kwargs["output_logits"] = True
        if self.hidden_states is not None:
            kwargs["output_hidden_states"] = self.hidden_states
        if "return_dict_in_generate" in kwargs:
            assert kwargs["return_dict_in_generate"]
        if config.stop_seqs is not None:
            from transformers.generation import StopStringCriteria  # type: ignore

            stopping_criteria = [StopStringCriteria(self.tokenizer, config.stop_seqs)]
            kwargs["stopping_criteria"] = stopping_criteria

        kwargs["return_dict_in_generate"] = True


        # replace the logit processor with something doing logit diff
        logits_processor = kwargs.get("logits_processor", None)
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
        logits_processor.append(LoraDiffLogitsProcessor(self.model, alpha=self.logit_diff_alpha))
        kwargs["logits_processor"] = logits_processor


        generator = functools.partial(self.model.generate, **kwargs)

        # prepare decoder
        decoder = functools.partial(
            self.tokenizer.batch_decode,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        # generate (uses a queue to batch so we await)
        response = await batched_generate(
            GenerateInput(
                input=chat,
                device=self.model.device,
                tokenizer=tokenizer,
                generator=generator,
                decoder=decoder,
                batch_size=config.max_connections or self.max_connections(),
            )
        )

        torch.cuda.empty_cache()
        # gather logprobs
        final_logprobs = None
        if config.logprobs is not None:

       
            final_logprobs = extract_logprobs(
                response=response,
                top=config.top_logprobs,
                tokenizer=self.tokenizer,
            )


        # construct choice
        choice = ChatCompletionChoice(
            message=ChatMessageAssistant(
                content=response.output, model=self.model_name, source="generate"
            ),
            logprobs=(
                Logprobs(content=final_logprobs) if final_logprobs is not None else None
            ),
        )

        choice = ChatCompletionChoice(
            message=chat_completion_assistant_message(
                response, tools, handler, self.model_name
            ),
            logprobs=(
                Logprobs(content=final_logprobs) if final_logprobs is not None else None
            ),
        )

        # return output
        return ModelOutput(
            model=self.model_name,
            choices=[choice],
            usage=ModelUsage(
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                total_tokens=response.total_tokens,
            ),
            time=response.time,
            metadata={"hidden_states": response.hidden_states},
        )
@modelapi(name="hf_logit_diff")
def custom():

    return LogitDiffHuggingFaceAPI
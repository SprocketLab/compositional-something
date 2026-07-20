"""Chat-template-aware Qwen LoRA training for coding atomic examples."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from self.coding.atomic_data import AtomicExample
from self.core.training_data import CausalLMDataCollator


def chat_prefix_ids(tokenizer: Any, messages: Sequence[Mapping[str, str]]) -> List[int]:
    rendered = tokenizer.apply_chat_template(
        list(messages),
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = rendered["input_ids"] if hasattr(rendered, "keys") else rendered
    return [int(token) for token in ids]


def chat_prefix_text(tokenizer: Any, messages: Sequence[Mapping[str, str]]) -> str:
    return str(
        tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )


def im_end_token_id(tokenizer: Any) -> int:
    token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if token_id is None or token_id == tokenizer.unk_token_id:
        raise ValueError("Tokenizer does not expose the Qwen <|im_end|> token")
    return int(token_id)


class ChatTargetDataset(Dataset):
    """Tokenize chat prompts while applying loss only to the assistant target."""

    def __init__(
        self,
        examples: Sequence[AtomicExample],
        tokenizer: Any,
        *,
        max_length: int,
    ) -> None:
        self.examples = list(examples)
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self._end_id = im_end_token_id(tokenizer)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        example = self.examples[index]
        prefix = chat_prefix_ids(self.tokenizer, example.messages)
        target = list(self.tokenizer.encode(example.target, add_special_tokens=False)) + [self._end_id]
        input_ids = prefix + target
        if len(input_ids) > self.max_length:
            raise ValueError(
                f"Example {example.source_id} has {len(input_ids)} tokens, exceeding max_length={self.max_length}"
            )
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": [-100] * len(prefix) + target,
        }


def load_qwen_tokenizer(model_name: str) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=Path(model_name).exists(),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_qwen_lora_model(
    model_name: str,
    *,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.0,
) -> Any:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3.5-4B coding training requires CUDA")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=Path(model_name).exists(),
        dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=int(lora_rank),
            lora_alpha=int(lora_alpha),
            lora_dropout=float(lora_dropout),
            bias="none",
            target_modules="all-linear",
        ),
    )
    model.to(torch.device("cuda"))
    return model


def adapter_parameter_summary(model: Any) -> Dict[str, Any]:
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    target_modules = sum(1 for _name, module in model.named_modules() if hasattr(module, "lora_A"))
    return {
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_percent": 100.0 * trainable / max(total, 1),
        "lora_target_module_count": target_modules,
    }


def train_lora(
    *,
    model: Any,
    tokenizer: Any,
    examples: Sequence[AtomicExample],
    output_dir: Path,
    max_length: int,
    max_steps: int,
    learning_rate: float,
    micro_batch_size: int,
    effective_batch_size: int,
    seed: int,
) -> Dict[str, Any]:
    from transformers import Trainer, TrainingArguments

    if effective_batch_size % micro_batch_size != 0:
        raise ValueError("effective_batch_size must be divisible by micro_batch_size")
    accumulation = effective_batch_size // micro_batch_size
    dataset = ChatTargetDataset(examples, tokenizer, max_length=max_length)
    logging_steps = max(1, max_steps // 10)
    arguments = TrainingArguments(
        output_dir=str(output_dir / "trainer"),
        max_steps=int(max_steps),
        learning_rate=float(learning_rate),
        lr_scheduler_type="constant",
        warmup_steps=0,
        per_device_train_batch_size=int(micro_batch_size),
        gradient_accumulation_steps=int(accumulation),
        weight_decay=0.0,
        max_grad_norm=1.0,
        logging_steps=logging_steps,
        logging_first_step=True,
        save_strategy="no",
        eval_strategy="no",
        report_to=[],
        bf16=True,
        fp16=False,
        seed=int(seed),
        data_seed=int(seed),
        remove_unused_columns=False,
        dataloader_num_workers=0,
        optim="adamw_torch",
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=dataset,
        data_collator=CausalLMDataCollator(tokenizer),
    )
    result = trainer.train()
    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    return {
        **{key: float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else value for key, value in result.metrics.items()},
        "micro_batch_size": micro_batch_size,
        "gradient_accumulation_steps": accumulation,
        "effective_batch_size": effective_batch_size,
        "log_history": list(trainer.state.log_history),
    }


@torch.inference_mode()
def generate_predictions(
    *,
    model: Any,
    tokenizer: Any,
    examples: Sequence[AtomicExample],
    batch_size: int,
    max_new_tokens: int,
) -> List[str]:
    model.eval()
    predictions: List[str] = []
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        prompts = [chat_prefix_text(tokenizer, example.messages) for example in batch]
        encoded = tokenizer(
            prompts,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        prompt_width = int(encoded["input_ids"].shape[1])
        generated = model.generate(
            **encoded,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
        for sequence in generated:
            predictions.append(
                tokenizer.decode(sequence[prompt_width:], skip_special_tokens=True).strip()
            )
    return predictions


def load_adapter_for_evaluation(model_name: str, adapter_dir: Path) -> Tuple[Any, Any]:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    tokenizer = load_qwen_tokenizer(str(adapter_dir))
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=Path(model_name).exists(),
        dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.to(torch.device("cuda"))
    model.eval()
    return model, tokenizer


def load_adapter_for_training(model_name: str, adapter_dir: Path) -> Tuple[Any, Any]:
    """Load an existing LoRA adapter as the trainable continuation checkpoint."""

    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3.5-4B coding training requires CUDA")
    tokenizer = load_qwen_tokenizer(str(adapter_dir))
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=Path(model_name).exists(),
        dtype=torch.bfloat16,
    )
    base.config.use_cache = False
    model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=True)
    model.to(torch.device("cuda"))
    model.train()
    return model, tokenizer

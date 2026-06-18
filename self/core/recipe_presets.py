"""Self-improvement recipe presets and phase schedules."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Dict, List, Tuple


RECIPE_ALGORITHMIC_SELF_IMPROVE_V1 = "algorithmic_self_improve_v1"
RECIPE_ARITHMETIC_SELF_IMPROVE_V1 = "arithmetic_self_improve_v1"
RECIPE_MULTIPLICATION_SELF_IMPROVE_V1 = "multiplication_self_improve_v1"
SUPPORTED_SELF_IMPROVEMENT_RECIPES = (
    RECIPE_ALGORITHMIC_SELF_IMPROVE_V1,
    RECIPE_ARITHMETIC_SELF_IMPROVE_V1,
    RECIPE_MULTIPLICATION_SELF_IMPROVE_V1,
)


def recipe_enabled(recipe_name: str) -> bool:
    return recipe_name != "none"


@dataclass(frozen=True)
class RecipeTrainingPhaseConfig:
    learning_rate: float
    weight_decay: float
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    warmup_steps: int
    max_steps: int
    num_stable_steps: int
    num_decay_steps: int


@dataclass(frozen=True)
class SelfImprovementRecipePreset:
    name: str
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_hidden_layers: int
    max_position_embeddings: int
    decode_max_new_tokens: int
    seed_phase: RecipeTrainingPhaseConfig
    self_improve_phase: RecipeTrainingPhaseConfig
    min_lr_ratio: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    bf16: bool
    tf32: bool
    auto_find_batch_size: bool
    logging_steps: int = 50


def resolve_self_improvement_recipe(name: str) -> SelfImprovementRecipePreset:
    if name not in SUPPORTED_SELF_IMPROVEMENT_RECIPES:
        raise ValueError(f"Unsupported self-improvement recipe: {name!r}")
    common_kwargs = dict(
        name=name,
        hidden_size=384,
        intermediate_size=1536,
        num_attention_heads=6,
        num_hidden_layers=6,
        max_position_embeddings=1024,
        bf16=True,
        tf32=True,
        auto_find_batch_size=True,
    )

    if name == RECIPE_MULTIPLICATION_SELF_IMPROVE_V1:
        return SelfImprovementRecipePreset(
            **common_kwargs,
            decode_max_new_tokens=48,
            seed_phase=RecipeTrainingPhaseConfig(
                learning_rate=5e-5,
                weight_decay=0.01,
                adam_beta1=0.9,
                adam_beta2=0.98,
                adam_epsilon=1e-12,
                warmup_steps=1_000,
                max_steps=10_000,
                num_stable_steps=7_000,
                num_decay_steps=2_000,
            ),
            self_improve_phase=RecipeTrainingPhaseConfig(
                learning_rate=5e-5,
                weight_decay=0.01,
                adam_beta1=0.9,
                adam_beta2=0.98,
                adam_epsilon=1e-12,
                warmup_steps=0,
                max_steps=3_000,
                num_stable_steps=2_000,
                num_decay_steps=1_000,
            ),
            min_lr_ratio=0.01,
            per_device_train_batch_size=256,
            per_device_eval_batch_size=256,
            gradient_accumulation_steps=1,
        )

    return SelfImprovementRecipePreset(
        **common_kwargs,
        decode_max_new_tokens=48,
        seed_phase=RecipeTrainingPhaseConfig(
            learning_rate=5e-4,
            weight_decay=0.1,
            adam_beta1=0.9,
            adam_beta2=0.99,
            adam_epsilon=1e-12,
            warmup_steps=1_000,
            max_steps=10_000,
            num_stable_steps=8_000,
            num_decay_steps=1_000,
        ),
        self_improve_phase=RecipeTrainingPhaseConfig(
            learning_rate=5e-4,
            weight_decay=0.1,
            adam_beta1=0.9,
            adam_beta2=0.99,
            adam_epsilon=1e-12,
            warmup_steps=0,
            max_steps=3_000,
            num_stable_steps=2_000,
            num_decay_steps=1_000,
        ),
        min_lr_ratio=0.01,
        per_device_train_batch_size=1024,
        per_device_eval_batch_size=1024,
        gradient_accumulation_steps=1,
    )


AdditionRecipePreset = SelfImprovementRecipePreset


def resolve_addition_recipe(name: str) -> AdditionRecipePreset:
    if name != RECIPE_ARITHMETIC_SELF_IMPROVE_V1:
        raise ValueError(f"Unsupported addition recipe: {name!r}")
    return resolve_self_improvement_recipe(name)


def resolve_recipe_phase(
    preset: SelfImprovementRecipePreset,
    phase_name: str,
) -> RecipeTrainingPhaseConfig:
    if phase_name == "seed":
        return preset.seed_phase
    if phase_name in {"self_improve", "frontier"}:
        return preset.self_improve_phase
    raise ValueError(f"Unsupported recipe phase: {phase_name!r}")


def fit_recipe_phase_to_max_steps(
    phase: RecipeTrainingPhaseConfig,
    *,
    max_steps: int,
) -> RecipeTrainingPhaseConfig:
    """Compress the warmup/stable/decay schedule for shorter max-step budgets."""

    total_schedule_steps = int(phase.warmup_steps) + int(phase.num_stable_steps) + int(phase.num_decay_steps)
    max_steps = int(max_steps)
    if max_steps <= 0 or total_schedule_steps <= 0 or max_steps >= total_schedule_steps:
        return phase

    raw_segments = (
        ("warmup_steps", float(phase.warmup_steps)),
        ("num_stable_steps", float(phase.num_stable_steps)),
        ("num_decay_steps", float(phase.num_decay_steps)),
    )
    scale = float(max_steps) / float(total_schedule_steps)

    floors: Dict[str, int] = {}
    remainders: List[Tuple[float, str]] = []
    assigned = 0
    for name, value in raw_segments:
        scaled = value * scale
        floored = int(math.floor(scaled))
        floors[name] = floored
        assigned += floored
        remainders.append((scaled - floored, name))

    remaining = max_steps - assigned
    for _, name in sorted(remainders, reverse=True):
        if remaining <= 0:
            break
        floors[name] += 1
        remaining -= 1

    return replace(
        phase,
        warmup_steps=floors["warmup_steps"],
        num_stable_steps=floors["num_stable_steps"],
        num_decay_steps=floors["num_decay_steps"],
    )

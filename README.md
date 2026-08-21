# Math Tutor Project

A learning-focused LLM post-training project for building and evaluating a personalized math tutor.

## Current milestone: SFT v1

The first milestone focuses on a transparent supervised fine-tuning workflow:

1. Inspect and audit raw math-reasoning data.
2. Clean, filter, deduplicate, and format training examples.
3. Convert examples into chat-style SFT records.
4. Fine-tune `Qwen/Qwen2.5-0.5B` with LoRA.
5. Evaluate the base model and SFT model on fixed holdout sets.
6. Perform error analysis and iterate on data/training choices.

Reward modeling and RL/preference optimization are intentionally out of scope until the SFT milestone is complete.

## Repository layout

- `configs/` — experiment hyperparameters.
- `data/` — raw/processed local data plus tiny committed samples.
- `src/data/` — inspection, cleaning, formatting, and splitting logic.
- `src/training/` — tokenization, label masking, LoRA, and SFT training.
- `src/evaluation/` — generation, answer extraction, metrics, and error analysis.
- `tests/` — unit tests for core data/evaluation logic.
- `scripts/` — reproducible command-line entry points.

Large datasets, checkpoints, and model weights should not be committed to GitHub.

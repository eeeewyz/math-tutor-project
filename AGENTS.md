# AGENTS.md

## Project goal
Build a small, reproducible personalized math-tutor post-training project. The first milestone is SFT only: data preparation -> LoRA SFT -> evaluation -> error analysis.

## Current scope
- Base model: Qwen/Qwen2.5-0.5B
- Training method: LoRA SFT
- Primary focus: transparent data preprocessing and evaluation
- Do not add reward-model, DPO, PPO, or GRPO code until the SFT milestone is complete.

## Engineering rules
1. Keep raw and processed datasets out of git; only tiny examples go in `data/samples/`.
2. Put reusable Python code under `src/`, not notebooks.
3. Keep hyperparameters in `configs/`, not hard-coded in training scripts when practical.
4. Data-cleaning rules, answer parsing, assistant-only label masking, and evaluation behavior must be explicit and testable.
5. Do not hide core SFT logic behind high-level one-line training wrappers during the learning phase.
6. Add or update tests when changing parsing, masking, or cleaning behavior.
7. Keep generated checkpoints and large artifacts out of git.

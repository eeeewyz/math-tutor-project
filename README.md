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

## Conservative data cleaning (SFT v1)

Run from the repository root on a fixed local UTF-8 JSONL export:

```bash
python -m src.data.clean_data \
  --input data/raw/openr1_sample.jsonl \
  --output data/processed/openr1_clean.jsonl \
  --report data/processed/cleaning_report.json
python -m pytest -q
```

The same input order produces the same JSONL and report. Input, output and report
must have distinct paths. Input is never modified. Malformed JSON (including blank
lines) aborts with a line number before outputs are opened. Non-object records are
counted as `invalid_record_type`. This implementation retains cleaned rows in
memory; use a bounded local export before attempting a full dataset run.

Rules:

- `problem`, `solution`, and `answer` must be nonempty strings. Missing/null or
  whitespace-only values are rejected; non-string values are not coerced.
- Only these three texts are normalized: CRLF/CR becomes LF, surrounding whitespace
  and a leading BOM are removed. Internal spacing, LaTeX, reasoning and Unicode
  distinctions are preserved. Extra fields, generations and messages are untouched.
- Present `generations` must be a list/tuple. Present `correctness_math_verify`,
  `correctness_llama`, `is_reasoning_complete` and `finish_reasons` must be lists/
  tuples with the same length as generations. Metadata without generations, null/
  scalar sequences and length mismatches are rejected with field-specific reasons.
- Absent optional metadata and aligned empty lists are allowed. `messages` is not
  assumed to correspond to generations. Correctness values are not interpreted.
- Exact duplicates use the complete raw record (JSON key order is ignored).
  Normalized-text duplicates use the three cleaned required texts together and
  ignore metadata; the first structurally valid row wins. Different solutions to
  the same problem remain separate. Exact takes precedence over normalized counts,
  including repeated copies of an already-discarded normalized variant.
- Reports contain input/kept/discarded row counts, per-reason counts and separate
  exact/normalized duplicate counts. Multiple validation reasons may apply to one
  row, so reason counts need not sum to discarded rows. Duplicate counts are a
  subset of discard counts, not an additional loss.

### Still requires real OpenR1-Math validation

Tests use synthetic fixtures, not a downloaded OpenR1 export. Confirm whether
`solution`/`answer` are always strings, whether each listed metadata field really
is generation-indexed, whether absent metadata/empty generation lists should be
allowed, and whether first-wins deduplication across differing metadata is suitable
for the eventual training subset. Equal lengths cannot prove semantic alignment;
this cleaner never sorts, zips, repairs or selects generations based on flags.
No chat conversion, correctness filtering, tokenizer or training is added here.

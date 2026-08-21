"""Inspect OpenR1-Math data before any cleaning decisions are made.

This module deliberately separates pure summary logic from Hugging Face dataset
loading so the inspection behavior is easy to test and reuse.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any, Iterable


TEXT_FIELDS = ("problem", "solution", "answer")
SEQUENCE_FIELDS = (
    "generations",
    "is_reasoning_complete",
    "correctness_math_verify",
    "correctness_llama",
    "finish_reasons",
    "messages",
)
CATEGORICAL_FIELDS = ("problem_type", "question_type", "source")


def _is_missing_or_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, dict, set)):
        return len(value) == 0
    return False


def _length_stats(lengths: list[int]) -> dict[str, float | int] | None:
    if not lengths:
        return None
    return {
        "min": min(lengths),
        "max": max(lengths),
        "mean": round(sum(lengths) / len(lengths), 2),
    }


def summarize_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Return a compact audit summary for an iterable of dataset records."""
    rows = list(records)
    if not rows:
        return {
            "num_records": 0,
            "columns": [],
            "missing_or_empty": {},
            "text_length_chars": {},
            "sequence_length": {},
            "value_counts": {},
        }

    columns = sorted({key for row in rows for key in row.keys()})

    missing_or_empty = {
        column: sum(_is_missing_or_empty(row.get(column)) for row in rows)
        for column in columns
    }

    text_length_chars: dict[str, Any] = {}
    for field in TEXT_FIELDS:
        lengths = [
            len(row[field])
            for row in rows
            if isinstance(row.get(field), str) and row[field].strip()
        ]
        if lengths:
            text_length_chars[field] = _length_stats(lengths)

    sequence_length: dict[str, Any] = {}
    for field in SEQUENCE_FIELDS:
        lengths = [
            len(row[field])
            for row in rows
            if isinstance(row.get(field), (list, tuple))
        ]
        if lengths:
            sequence_length[field] = _length_stats(lengths)

    value_counts: dict[str, dict[str, int]] = {}
    for field in CATEGORICAL_FIELDS:
        values = [
            str(row[field])
            for row in rows
            if not _is_missing_or_empty(row.get(field))
        ]
        if values:
            value_counts[field] = dict(Counter(values).most_common())

    return {
        "num_records": len(rows),
        "columns": columns,
        "missing_or_empty": missing_or_empty,
        "text_length_chars": text_length_chars,
        "sequence_length": sequence_length,
        "value_counts": value_counts,
    }


def load_openr1_sample(sample_size: int, seed: int) -> list[dict[str, Any]]:
    """Load a reproducible sample from OpenR1-Math-220k default/train."""
    from datasets import load_dataset

    dataset = load_dataset(
        "open-r1/OpenR1-Math-220k",
        "default",
        split="train",
    )
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")

    sample_size = min(sample_size, len(dataset))
    sampled = dataset.shuffle(seed=seed).select(range(sample_size))
    return [dict(row) for row in sampled]


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect OpenR1-Math-220k before cleaning")
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    records = load_openr1_sample(args.sample_size, args.seed)
    summary = summarize_records(records)
    rendered = json.dumps(summary, indent=2, ensure_ascii=False)
    print(rendered)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(rendered + "\n")


if __name__ == "__main__":
    main()

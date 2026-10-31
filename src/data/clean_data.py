"""Conservative, deterministic preprocessing of local OpenR1-style JSONL.

Only structural checks are made. Equal lengths do not prove semantic alignment.
No correctness values are interpreted and no reasoning is rewritten.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

TEXT_FIELDS = ('problem', 'solution', 'answer')
GENERATION_METADATA = (
    'correctness_math_verify', 'correctness_llama',
    'is_reasoning_complete', 'finish_reasons',
)


def normalize_text(value: str) -> str:
    """Unify line endings and remove boundary whitespace/BOM only."""
    return value.replace('\r\n', '\n').replace('\r', '\n').strip().lstrip('\ufeff').strip()


def _fingerprint(value: Any) -> str:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()


def clean_records(records: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return independent cleaned rows and an audit summary; first valid row wins.

    Missing optional metadata is allowed. Present metadata must be a list/tuple
    aligned in length with generations. messages is deliberately not aligned.
    Reasons can overlap; num_discarded counts rows exactly once.
    Exact duplicates compare full raw records; normalized duplicates compare
    the three required texts, including solution, and ignore metadata.
    """
    kept = []
    reasons: Counter[str] = Counter()
    duplicates = {'exact': 0, 'normalized_text': 0}
    seen_raw: set[str] = set()
    seen_text: set[str] = set()
    total = 0
    for row in records:
        total += 1
        errors = []
        if not isinstance(row, dict):
            reasons['invalid_record_type'] += 1
            continue
        cleaned = deepcopy(row)
        for field in TEXT_FIELDS:
            value = row.get(field)
            if value is None:
                errors.append(f'missing_or_empty:{field}')
            elif not isinstance(value, str):
                errors.append(f'invalid_text_type:{field}')
            else:
                cleaned[field] = normalize_text(value)
                if not cleaned[field]:
                    errors.append(f'missing_or_empty:{field}')
        present = [field for field in GENERATION_METADATA if field in row]
        if present and 'generations' not in row:
            errors.append('missing_generations')
        generations = row.get('generations')
        valid_generations = isinstance(generations, (list, tuple))
        if 'generations' in row and not valid_generations:
            errors.append('invalid_sequence_type:generations')
        for field in present:
            if not isinstance(row[field], (list, tuple)):
                errors.append(f'invalid_sequence_type:{field}')
            elif valid_generations and len(row[field]) != len(generations):
                errors.append(f'sequence_length_mismatch:{field}')
        if errors:
            reasons.update(errors)
            continue
        raw_key = _fingerprint(row)
        text_key = _fingerprint([cleaned[field] for field in TEXT_FIELDS])
        if raw_key in seen_raw:
            duplicates['exact'] += 1
            reasons['duplicate_exact'] += 1
        elif text_key in seen_text:
            duplicates['normalized_text'] += 1
            reasons['duplicate_normalized_text'] += 1
        else:
            kept.append(cleaned)
        seen_raw.add(raw_key)
        seen_text.add(text_key)
    return kept, {
        'num_input': total, 'num_kept': len(kept),
        'num_discarded': total - len(kept),
        'discard_reasons': dict(sorted(reasons.items())), 'duplicates': duplicates,
    }


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding='utf-8-sig') as source:
        for line_number, line in enumerate(source, 1):
            try:
                yield json.loads(line)
            except ValueError as exc:
                raise ValueError(f'{path}:{line_number}: invalid JSON') from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    paths = [args.input, args.output] + ([args.report] if args.report else [])
    if len({p.resolve() for p in paths}) != len(paths):
        parser.error('input, output and report paths must be distinct')
    try:
        kept, report = clean_records(_read_jsonl(args.input))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    # Fully validate before opening outputs so bad input cannot truncate results.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8') as output:
        for row in kept:
            output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + '\n'
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding='utf-8')
    print(rendered, end='')


if __name__ == '__main__':
    main()

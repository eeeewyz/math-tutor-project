"""Synthetic schema fixtures; generation metadata semantics need real-data validation."""
import json
import subprocess
import sys
from copy import deepcopy

import pytest

from src.data.clean_data import clean_records


def row(**changes):
    return dict(problem='What is 1 + 1?', solution='1 + 1 = 2', answer='2', **changes)


def test_keep_normalize_without_mutation():
    original = row(generations=[' trace '], correctness_math_verify=[False], messages=[{'content': ' untouched '}])
    original['solution'] = '\ufeff  x  + y\r\n= 2  \r '
    before = deepcopy(original)
    kept, report = clean_records([original])
    assert original == before
    assert kept[0]['solution'] == 'x  + y\n= 2'
    assert kept[0]['generations'] == [' trace ']
    assert kept[0]['messages'] == original['messages']
    assert report['num_kept'] == 1


@pytest.mark.parametrize('field', ['problem', 'solution', 'answer'])
@pytest.mark.parametrize('value,reason', [(None, 'missing_or_empty'), ('', 'missing_or_empty'), (' \n\t', 'missing_or_empty'), ([], 'invalid_text_type'), (42, 'invalid_text_type')])
def test_required_text(field, value, reason):
    sample = row()
    sample[field] = value
    kept, report = clean_records([sample])
    assert kept == []
    assert report['discard_reasons'] == {f'{reason}:{field}': 1}


def test_missing_key_and_multiple_reasons_are_counted_once_per_record():
    kept, report = clean_records([{'answer': '2'}])
    assert not kept
    assert report['num_discarded'] == 1
    assert report['discard_reasons'] == {'missing_or_empty:problem': 1, 'missing_or_empty:solution': 1}


@pytest.mark.parametrize('field', ['correctness_math_verify', 'correctness_llama', 'is_reasoning_complete', 'finish_reasons'])
def test_alignment(field):
    kept, report = clean_records([row(generations=['a', 'b'], **{field: [True]})])
    assert not kept
    assert report['discard_reasons'] == {f'sequence_length_mismatch:{field}': 1}


@pytest.mark.parametrize('changes,reason', [
    ({'generations': 'a'}, 'invalid_sequence_type:generations'),
    ({'generations': None}, 'invalid_sequence_type:generations'),
    ({'correctness_llama': [True]}, 'missing_generations'),
    ({'generations': ['a'], 'correctness_llama': None}, 'invalid_sequence_type:correctness_llama'),
    ({'generations': ['a'], 'correctness_llama': True}, 'invalid_sequence_type:correctness_llama'),
])
def test_unverifiable_alignment(changes, reason):
    kept, report = clean_records([row(**changes)])
    assert not kept
    assert reason in report['discard_reasons']


def test_absent_metadata_and_aligned_empty_sequences_are_preserved():
    assert clean_records([row()])[1]['num_kept'] == 1
    assert clean_records([row(generations=[], correctness_llama=[])])[1]['num_kept'] == 1


def test_exact_and_normalized_duplicates_and_distinct_solutions():
    first = row(source='first')
    variant = row(source='second')
    variant['problem'] = '  What is 1 + 1?\r\n'
    other = row()
    other['solution'] = 'Different reasoning'
    kept, report = clean_records([first, deepcopy(first), variant, deepcopy(variant), other])
    assert kept == [first, other]
    assert report['duplicates'] == {'exact': 2, 'normalized_text': 1}
    assert report['num_input'] == report['num_kept'] + report['num_discarded']
    assert clean_records(kept)[0] == kept
    assert clean_records([])[1]['num_input'] == 0


def test_cli_reproducible(tmp_path):
    source = tmp_path / 'raw.jsonl'
    output = tmp_path / 'clean.jsonl'
    report = tmp_path / 'report.json'
    source.write_text('\n'.join(json.dumps(x) for x in [row(), row(), {}]), encoding='utf-8')
    cmd = [sys.executable, '-m', 'src.data.clean_data', '--input', str(source), '--output', str(output), '--report', str(report)]
    one = subprocess.run(cmd, check=True, capture_output=True, text=True)
    two = subprocess.run(cmd, check=True, capture_output=True, text=True)
    assert one.stdout == two.stdout == report.read_text()
    assert json.loads(one.stdout)['num_kept'] == 1
    assert len(output.read_text().splitlines()) == 1


def test_cli_rejects_input_overwrite(tmp_path):
    source = tmp_path / 'raw.jsonl'
    source.write_text(json.dumps(row()) + '\n')
    result = subprocess.run([sys.executable, '-m', 'src.data.clean_data', '--input', str(source), '--output', str(source)], capture_output=True)
    assert result.returncode != 0
    assert json.loads(source.read_text()) == row()

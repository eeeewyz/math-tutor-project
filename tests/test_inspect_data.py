from src.data.inspect_data import summarize_records


def test_summarize_records_reports_schema_missing_values_and_lengths():
    records = [
        {
            "problem": "What is 1 + 1?",
            "solution": "1 + 1 = 2",
            "answer": "2",
            "problem_type": "arithmetic",
            "source": "demo",
            "generations": ["trace one", "trace two"],
            "correctness_math_verify": [True, False],
        },
        {
            "problem": "What is 2 + 2?",
            "solution": "",
            "answer": "4",
            "problem_type": "arithmetic",
            "source": None,
            "generations": ["trace three"],
            "correctness_math_verify": [True],
        },
    ]

    summary = summarize_records(records)

    assert summary["num_records"] == 2
    assert summary["columns"] == sorted(records[0].keys())
    assert summary["missing_or_empty"]["solution"] == 1
    assert summary["missing_or_empty"]["source"] == 1
    assert summary["text_length_chars"]["problem"]["min"] == len("What is 1 + 1?")
    assert summary["sequence_length"]["generations"]["max"] == 2
    assert summary["value_counts"]["problem_type"]["arithmetic"] == 2

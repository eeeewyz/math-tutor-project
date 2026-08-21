# Data layout

- `raw/`: downloaded source datasets. Not committed to git.
- `processed/`: cleaned, formatted, tokenization-ready datasets. Not committed to git.
- `samples/`: tiny examples that document the expected schema and are safe to commit.

The first SFT milestone will use a small, quality-filtered subset of a post-2024 math-reasoning dataset. Data preparation should explicitly cover schema checks, correctness filtering, answer parsing, deduplication, length filtering, category/difficulty inspection, splitting, and final chat formatting.

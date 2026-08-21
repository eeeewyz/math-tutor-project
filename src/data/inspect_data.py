"""在进行任何数据清洗之前，对 OpenR1-Math 数据进行检查与统计。

本模块将“纯统计逻辑”和 Hugging Face 数据集加载逻辑刻意分开，
这样可以让数据检查功能更容易测试、复用和维护。
"""

from __future__ import annotations

# argparse：解析命令行参数，例如采样数量、随机种子和输出路径
import argparse
# json：将统计结果格式化为 JSON，便于终端查看或保存到文件
import json
# Counter：统计类别字段中各个取值出现的次数
from collections import Counter
# Any / Iterable：用于类型标注，提高代码可读性
from typing import Any, Iterable


# 文本类字段：主要用于统计字符串长度
TEXT_FIELDS = ("problem", "solution", "answer")

# 序列类字段：主要用于统计 list / tuple 的长度
SEQUENCE_FIELDS = (
    "generations",
    "is_reasoning_complete",
    "correctness_math_verify",
    "correctness_llama",
    "finish_reasons",
    "messages",
)

# 类别字段：用于统计不同类别值出现的次数
CATEGORICAL_FIELDS = ("problem_type", "question_type", "source")


def _is_missing_or_empty(value: Any) -> bool:
    """判断一个字段值是否应被视为“缺失或为空”。"""

    # None 直接视为缺失值
    if value is None:
        return True

    # 字符串去掉首尾空格后为空，则视为空值
    if isinstance(value, str):
        return not value.strip()

    # 常见容器类型长度为 0 时，也视为空值
    if isinstance(value, (list, tuple, dict, set)):
        return len(value) == 0

    # 其他类型默认认为存在有效值
    return False


def _length_stats(lengths: list[int]) -> dict[str, float | int] | None:
    """计算一组长度数据的最小值、最大值和平均值。"""

    # 如果没有可统计的数据，返回 None
    if not lengths:
        return None

    # 返回长度统计信息；平均值保留两位小数
    return {
        "min": min(lengths),
        "max": max(lengths),
        "mean": round(sum(lengths) / len(lengths), 2),
    }


def summarize_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """对一批数据记录进行快速审计，并返回紧凑的统计摘要。"""

    # 将可迭代对象转换为列表，方便后续多次遍历
    rows = list(records)

    # 如果没有任何数据，返回结构完整但内容为空的统计结果
    if not rows:
        return {
            "num_records": 0,
            "columns": [],
            "missing_or_empty": {},
            "text_length_chars": {},
            "sequence_length": {},
            "value_counts": {},
        }

    # 汇总所有样本中出现过的字段，并按字母顺序排序
    columns = sorted({key for row in rows for key in row.keys()})

    # 对每一个字段统计缺失值或空值的数量
    missing_or_empty = {
        column: sum(_is_missing_or_empty(row.get(column)) for row in rows)
        for column in columns
    }

    # ---------------- 文本字段长度统计 ----------------
    # 用于观察 problem / solution / answer 等文本字段的大致长度分布
    text_length_chars: dict[str, Any] = {}
    for field in TEXT_FIELDS:
        lengths = [
            len(row[field])
            for row in rows
            # 只统计非空字符串
            if isinstance(row.get(field), str) and row[field].strip()
        ]
        if lengths:
            text_length_chars[field] = _length_stats(lengths)

    # ---------------- 序列字段长度统计 ----------------
    # 用于观察 generations、messages 等列表字段通常包含多少个元素
    sequence_length: dict[str, Any] = {}
    for field in SEQUENCE_FIELDS:
        lengths = [
            len(row[field])
            for row in rows
            # 只统计 list 或 tuple 类型的字段
            if isinstance(row.get(field), (list, tuple))
        ]
        if lengths:
            sequence_length[field] = _length_stats(lengths)

    # ---------------- 类别字段分布统计 ----------------
    # 例如统计 problem_type、question_type、source 的取值分布
    value_counts: dict[str, dict[str, int]] = {}
    for field in CATEGORICAL_FIELDS:
        values = [
            str(row[field])
            for row in rows
            # 忽略缺失或空值，只统计实际存在的类别
            if not _is_missing_or_empty(row.get(field))
        ]
        if values:
            # Counter.most_common() 会按出现次数从高到低排列
            value_counts[field] = dict(Counter(values).most_common())

    # 汇总并返回整个数据检查结果
    return {
        "num_records": len(rows),
        "columns": columns,
        "missing_or_empty": missing_or_empty,
        "text_length_chars": text_length_chars,
        "sequence_length": sequence_length,
        "value_counts": value_counts,
    }


def load_openr1_sample(sample_size: int, seed: int) -> list[dict[str, Any]]:
    """从 OpenR1-Math-220k 的默认训练集加载一个可复现的随机样本。"""

    # 在函数内部导入 datasets，只有真正加载数据集时才需要该依赖
    from datasets import load_dataset

    # 从 Hugging Face 加载 OpenR1-Math-220k 的 default/train split
    dataset = load_dataset(
        "open-r1/OpenR1-Math-220k",
        "default",
        split="train",
    )

    # sample_size 必须大于 0，避免无意义的采样请求
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")

    # 防止用户指定的采样数量超过数据集实际大小
    sample_size = min(sample_size, len(dataset))

    # 先根据固定随机种子打乱，再选取前 sample_size 条记录
    # 使用固定 seed 可以保证实验具有可复现性
    sampled = dataset.shuffle(seed=seed).select(range(sample_size))

    # 将 Hugging Face Dataset 中的每条记录转换为普通 Python 字典
    return [dict(row) for row in sampled]


def main() -> None:
    """命令行入口：加载样本、生成统计摘要，并按需保存结果。"""

    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="Inspect OpenR1-Math-220k before cleaning")

    # --sample-size：需要随机抽取的数据数量，默认 1000 条
    parser.add_argument("--sample-size", type=int, default=1000)

    # --seed：随机种子，用于保证每次采样结果一致
    parser.add_argument("--seed", type=int, default=42)

    # --output：可选输出文件路径；如果不指定，则只打印到终端
    parser.add_argument("--output", type=str, default=None)

    # 解析命令行参数
    args = parser.parse_args()

    # 第一步：从原始 OpenR1-Math 数据集中抽取样本
    records = load_openr1_sample(args.sample_size, args.seed)

    # 第二步：对样本执行字段、缺失值、长度和类别分布统计
    summary = summarize_records(records)

    # 第三步：将统计结果格式化为可读的 JSON 字符串
    rendered = json.dumps(summary, indent=2, ensure_ascii=False)

    # 默认将结果打印到终端
    print(rendered)

    # 如果用户指定了输出路径，则同时保存到文件
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(rendered + "\n")


# 只有直接运行本文件时才执行 main()；
# 如果该文件被其他模块 import，则不会自动执行命令行逻辑
if __name__ == "__main__":
    main()

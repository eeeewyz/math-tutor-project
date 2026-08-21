# 导入数据检查函数，用于分析数据集结构和质量
from src.data.inspect_data import summarize_records


# 测试 summarize_records 是否能够正确报告：
# 1. 数据字段(schema)
# 2. 缺失值/空值数量
# 3. 文本长度统计
# 4. 序列(list)长度统计
# 5. 类别(value)分布统计
def test_summarize_records_reports_schema_missing_values_and_lengths():

    # 构造一组模拟训练数据，用于测试数据检查功能
    # 模拟 Math Tutor 项目中的数学题数据格式
    records = [

        # 第一条数据：完整样本
        {
            "problem": "What is 1 + 1?",
            "solution": "1 + 1 = 2",
            "answer": "2",
            "problem_type": "arithmetic",
            "source": "demo",
            "generations": ["trace one", "trace two"],
            "correctness_math_verify": [True, False],
        },

        # 第二条数据：故意加入缺失情况
        # 用于测试函数是否能发现空 solution 和 None source
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


    # 调用数据分析函数，对整个 dataset 进行统计
    summary = summarize_records(records)


    # -------- 验证统计结果是否正确 --------


    # 检查数据总数量是否正确
    assert summary["num_records"] == 2


    # 检查函数是否正确识别所有字段(schema)
    # 并按照字母排序返回
    assert summary["columns"] == sorted(records[0].keys())


    # 检查缺失值检测：
    # solution 有一个为空字符串，因此缺失数量应该为1
    assert summary["missing_or_empty"]["solution"] == 1


    # source 有一个 None，因此缺失数量应该为1
    assert summary["missing_or_empty"]["source"] == 1


    # 检查文本长度统计功能：
    # problem字段的最短文本长度应该等于第一条problem长度
    assert summary["text_length_chars"]["problem"]["min"] == len("What is 1 + 1?")


    # 检查列表长度统计功能：
    # generations字段：
    # 第一条长度=2，第二条长度=1
    # 最大长度应该为2
    assert summary["sequence_length"]["generations"]["max"] == 2


    # 检查类别统计功能：
    # problem_type中 arithmetic 出现两次
    assert summary["value_counts"]["problem_type"]["arithmetic"] == 2

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

import pytest
from evaluation import RagasEvaluator  # 假设源代码在 src 目录下

@pytest.mark.asyncio
async def test_evaluate_single():
    # 初始化评估器
    evaluator = RagasEvaluator()
    
    # 准备测试数据
    test_data = {
        "question": "什么是机器学习？",
        "answer": "机器学习是人工智能的一个分支，它使用数据和算法来模仿人类的学习方式，逐步提高准确性。",
        "contexts": [
            "机器学习是人工智能的一个重要分支，通过算法使计算机系统能够从数据中学习和改进。",
            "机器学习使用统计学方法，通过数据训练来提高性能和做出预测。"
        ],
        "reference": [
            "https://example.com/ml1",
            "https://example.com/ml2"
        ]
    }
    
    # 执行评估
    result = await evaluator.evaluate_single(**test_data)
    
    # 基本验证
    assert result is not None
    assert result.question == test_data["question"]
    assert result.answer == test_data["answer"]
    assert result.contexts == test_data["contexts"]
    assert result.reference == test_data["reference"]
    
    # 验证评分结果
    assert isinstance(result.scores, dict)
    
    # 打印实际返回的评分键，以便调试
    print("实际返回的评分键:", result.scores.keys())
    
    # 检查是否至少包含一个评分结果
    assert len(result.scores) > 0
    
    # 检查评分值是否在合理范围内
    for metric, score in result.scores.items():
        if isinstance(score, (int, float)):  # 只检查数值类型的评分
            assert 0 <= score <= 1, f"{metric} 的评分 {score} 超出有效范围"
        else:
            print(f"非数值类型的评分: {metric}: {score}")

@pytest.mark.asyncio
async def test_evaluate_single_with_error():
    # 初始化评估器
    evaluator = RagasEvaluator()
    
    # 准备无效的测试数据
    test_data = {
        "question": "",  # 空问题应该触发错误
        "answer": "这是一个测试回答",
        "contexts": [],  # 空上下文
        "reference": []
    }
    
    # 执行评估
    result = await evaluator.evaluate_single(**test_data)
    
    # 验证结果
    assert result is not None
    assert isinstance(result.scores, dict)
    
    # 如果返回错误信息
    if "错误" in result.scores:
        assert isinstance(result.scores["错误"], str)
        print("错误信息:", result.scores["错误"])
    # 如果返回评分结果
    else:
        for metric, score in result.scores.items():
            if isinstance(score, (int, float)):
                assert 0 <= score <= 1

if __name__ == "__main__":
    pytest.main(["-v"]) 
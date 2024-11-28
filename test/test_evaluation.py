import pytest
from evaluation import RagasEvaluator

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
        "context_urls": [
            "https://example.com/ml1",
            "https://example.com/ml2"
        ]
    }
    
    # 执行评估
    result = await evaluator.evaluate_single(
        question=test_data["question"],
        answer=test_data["answer"],
        contexts=test_data["contexts"],
        context_urls=test_data["context_urls"]
    )
    
    # 验证结果
    assert result is not None
    assert result.question == test_data["question"]
    assert result.answer == test_data["answer"]
    assert result.contexts == test_data["contexts"]
    assert result.context_urls == test_data["context_urls"]
    
    # 验证评分结果
    assert isinstance(result.scores, dict)
    expected_metrics = {
        "回答相关性",
        "上下文精确度",
        "忠实度",
        "上下文召回"
    }
    
    # 检查是否包含所有预期的评估指标
    actual_metrics = set(result.scores.keys())
    assert expected_metrics.issubset(actual_metrics)
    
    # 验证评分值是否在合理范围内 (0-1)
    for score in result.scores.values():
        if isinstance(score, float):  # 排除可能的错误消息
            assert 0 <= score <= 1

@pytest.mark.asyncio
async def test_evaluate_single_error_handling():
    # 初始化评估器
    evaluator = RagasEvaluator()
    
    # 准备无效的测试数据
    test_data = {
        "question": "",  # 空问题应该触发错误
        "answer": "这是一个测试回答",
        "contexts": [],  # 空上下文
        "context_urls": []
    }
    
    # 执行评估
    result = await evaluator.evaluate_single(
        question=test_data["question"],
        answer=test_data["answer"],
        contexts=test_data["contexts"],
        context_urls=test_data["context_urls"]
    )
    
    # 验证错误处理
    assert result is not None
    assert "错误" in result.scores  # 应该包含错误信息
    assert isinstance(result.scores["错误"], str)  # 错误信息应该是字符串

if __name__ == "__main__":
    pytest.main(["-v"]) 
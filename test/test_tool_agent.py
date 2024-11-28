import pytest
from tool_agent import create_workflow, AgentState
from langchain_core.messages import HumanMessage

@pytest.mark.asyncio
async def test_workflow_execution():
    # 测试数据
    query = "我需要自己做河豚鱼刺身，具体应该怎么做，有什么步骤和重点注意"
    
    # 创建工作流和状态
    workflow = create_workflow()
    state = AgentState(
        messages=[HumanMessage(content=query)],
        current_agent="master",
        next_agent=None,
        final_answer=None,
        tools_output={}
    )
    
    # 执行工作流
    result = await workflow.ainvoke(state)
    
    # 基本验证
    assert result is not None
    assert isinstance(result, dict)
    
    # 验证工具输出格式
    tools_output = result.get('tools_output', {}).get('result', {})
    assert isinstance(tools_output, dict)
    
    # 验证过滤后的结果
    filtered_results = tools_output.get('filtered_results', [])
    assert isinstance(filtered_results, list)
    
    # 如果有搜索结果，验证结果格式
    if filtered_results:
        for item in filtered_results:
            assert 'title' in item
            assert 'content' in item
            assert 'url' in item
            assert 'relevance_check' in item
            
            # 验证相关度检查
            relevance_check = item.get('relevance_check', {})
            assert isinstance(relevance_check.get('confidence', 0), (int, float))
    
    # 验证统计信息
    assert isinstance(tools_output.get('original_count', 0), int)
    assert isinstance(tools_output.get('filtered_count', 0), int)


@pytest.mark.asyncio
async def test_table_data_formatting():
    # 模拟搜索结果数据
    mock_filtered_results = [
        {
            'title': '河豚刺身制作指南',
            'content': '这是一个示例内容...' * 10,  # 创建长内容
            'url': 'http://example.com',
            'relevance_check': {'confidence': 0.95}
        }
    ]
    
    # 格式化表格数据
    table_data = [
        [
            result.get('title', 'N/A'),
            result.get('content', '')[:200] + '...',
            f"{result.get('relevance_check', {}).get('confidence', 0):.2f}",
            result.get('url', 'N/A')
        ]
        for result in mock_filtered_results
    ]
    
    # 验证表格数据格式
    assert len(table_data) > 0
    for row in table_data:
        assert len(row) == 4  # 验证列数
        assert isinstance(row[0], str)  # 标题
        assert isinstance(row[1], str)  # 内容摘要
        assert len(row[1]) <= 203  # 内容长度（200 + '...'）
        assert isinstance(row[2], str)  # 相关度
        assert isinstance(row[3], str)  # URL

@pytest.mark.asyncio
async def test_stats_formatting():
    # 模拟工具输出数据
    mock_tools_output = {
        'original_count': 10,
        'filtered_count': 5,
        'quality_check': {'score': 0.85}
    }
    
    # 生成统计信息
    stats = f"""
### 搜索统计
- 原始结果数: {mock_tools_output.get('original_count', 0)}
- 筛选后结果数: {mock_tools_output.get('filtered_count', 0)}
- 结果质量评分: {mock_tools_output.get('quality_check', {}).get('score', 'N/A')}
    """
    
    # 验证统计信息格式
    assert '搜索统计' in stats
    assert '原始结果数: 10' in stats
    assert '筛选后结果数: 5' in stats
    assert '结果质量评分: 0.85' in stats

if __name__ == "__main__":
    pytest.main(["-v"]) 
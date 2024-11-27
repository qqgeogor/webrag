import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from abc import ABC, abstractmethod
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict

from enum import Enum

import asyncio
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import Graph, StateGraph,END
from langgraph.prebuilt import ToolExecutor
from pydantic import BaseModel, Field

from utils import WebContentExtractor
from logging_config import log_tool_usage
from text_chunker import TextChunker
from agent import WebSearchAgent, CalculatorAgent, ArxivAgent, MasterAgent,AgentState,ResultFormatter
import gradio as gr


# 6. 工作流定义
def create_workflow() -> Graph:
    workflow = StateGraph(AgentState)
    
    # 初始化代理
    master = MasterAgent()
    calculator = CalculatorAgent()
    websearch = WebSearchAgent()
    arxiv = ArxivAgent()
    formatter = ResultFormatter()
    
    # 添加节点
    workflow.add_node("master", master.invoke)
    
    # workflow.add_edge(
    #     "master",
    #     END
    # )

    
    workflow.add_node("websearch_agent", websearch.invoke)
    workflow.add_node("calculator_agent", calculator.invoke)
    workflow.add_node("arxiv_agent", arxiv.invoke)
    workflow.add_node("result_formatter", formatter.invoke)
    
    # 定义条件路由函数
    def router(state: AgentState) -> bool:
        if state["next_agent"]:
            return state["next_agent"]
        else:
            return 'end'
    
    
    # 添加条件边
    workflow.add_conditional_edges(
        "master",router,
        {
            'calculator_agent': "calculator_agent",
            'arxiv_agent': "arxiv_agent",
            'websearch_agent': "websearch_agent",
            'end': END
        }
    )
    
    

    workflow.add_edge(
        "calculator_agent",
        "result_formatter",
    )
    
    workflow.add_edge(
        "arxiv_agent",
        "result_formatter",
    )
    workflow.add_edge(
        "result_formatter",
        END
    )

    workflow.add_edge(
        "websearch_agent",
        END
    )
    
    
    workflow.set_entry_point("master")
    
    return workflow.compile()

# 修改主函数为异步处理单个查询
async def process_query(query: str) -> str:
    workflow = create_workflow()
    state = AgentState(
        messages=[HumanMessage(content=query)],
        current_agent="master",
        next_agent=None,
        final_answer=None,
        tools_output={}
    )
    try:
        result = await workflow.ainvoke(state)
        return result['final_answer']
    except Exception as e:
        return f"发生错误: {str(e)}"

# 添加 Gradio 接口处理函数
def gradio_interface(query: str) -> str:
    return asyncio.run(process_query(query))

# 主函数改为启动 Gradio
def main():
    async def process_query_with_details(query: str) -> tuple:
        workflow = create_workflow()
        state = AgentState(
            messages=[HumanMessage(content=query)],
            current_agent="master",
            next_agent=None,
            final_answer=None,
            tools_output={}
        )
        
        try:
            result = await workflow.ainvoke(state)
            
            # 提取搜索结果
            tools_output = result.get('tools_output', {}).get('result', {})
            filtered_results = tools_output.get('filtered_results', [])
            
            # 格式化搜索结果为表格数据
            table_data = [
                [
                    result.get('title', 'N/A'),
                    result.get('content', '')[:200] + '...',  # 内容摘要
                    f"{result.get('relevance_check', {}).get('confidence', 0):.2f}",
                    result.get('url', 'N/A')
                ]
                for result in filtered_results
            ]
            
            # 生成统计信息
            stats = f"""
### 搜索统计
- 原始结果数: {tools_output.get('original_count', 0)}
- 筛选后结果数: {tools_output.get('filtered_count', 0)}
- 结果质量评分: {tools_output.get('quality_check', {}).get('score', 'N/A')}
            """
            
            return (
                result['final_answer'],  # 主要回答
                table_data,              # 搜索结果表格
                stats                    # 统计信息
            )
            
        except Exception as e:
            return f"发生错误: {str(e)}", [], "处理出错"

    def gradio_interface(query: str) -> tuple:
        return asyncio.run(process_query_with_details(query))

    demo = gr.Blocks()
    
    with demo:
        gr.Markdown("# AI 助手")
        gr.Markdown("我可以帮您搜索信息、进行计算等。请输入您的问题。")
        
        with gr.Row():
            with gr.Column():
                # 输入框
                query_input = gr.Textbox(
                    lines=2, 
                    placeholder="请输入您的问题...",
                    label="问题"
                )
                
                # 提交按钮
                submit_btn = gr.Button("提交")
                
            with gr.Column():
                # 主要回答
                answer_output = gr.Textbox(
                    lines=4,
                    label="AI 回答"
                )
                
                # 搜索结果展示
                with gr.Accordion("参考资料", open=False):
                    results_output = gr.Dataframe(
                        headers=["标题", "内容摘要", "相关度", "来源链接"],
                        label="搜索结果",
                        wrap=True
                    )
                    
                    stats_output = gr.Markdown(
                        label="统计信息"
                    )

        # 设置提交事件
        submit_btn.click(
            fn=gradio_interface,
            inputs=[query_input],
            outputs=[answer_output, results_output, stats_output]
        )
        
        # 修改示例添加方式
        gr.Examples(
            examples=[
                ["请帮我计算 123 加 456"],
                ["慈禧太后是什么人，罗列具体的重大事件"],
                ["请帮我搜索关于 transformer architecture 的最新论文，并提供摘要"],
                ['怎么样才能做出好吃的蛋炒饭'],
                ['我需要自己做河豚鱼刺身，具体应该怎么做，有什么步骤和重点注意']
            ],
            inputs=query_input,
            outputs=[answer_output, results_output, stats_output]
        )
        
        # 修改 launch() 调用，移除 examples 参数
        demo.launch(share=False)  # 直接调用 launch()，不需要任何示例参数

if __name__ == "__main__":
    main()
    
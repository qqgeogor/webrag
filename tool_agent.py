import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from abc import ABC, abstractmethod
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict

import operator
from enum import Enum
import arxiv
import httpx
import json
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
        "websearch_agent",
        "result_formatter",
    )
    
    workflow.add_edge(
        "result_formatter",
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
    # 创建 Gradio 界面
    iface = gr.Interface(
        fn=gradio_interface,
        inputs=gr.Textbox(
            lines=2, 
            placeholder="请输入您的问题...",
            label="问题"
        ),
        outputs=gr.Textbox(
            lines=4,
            label="回答"
        ),
        title="AI 助手",
        description="我可以帮您搜索信息、进行计算等。请输入您的问题。",
        examples=[
            ["请帮我计算 123 加 456"],
            ["慈禧太后是什么人"],
            ["请帮我搜索关于 transformer architecture 的最新论文"],
        ]
    )
    
    # 启动界面
    iface.launch(share=False)

if __name__ == "__main__":
    main()
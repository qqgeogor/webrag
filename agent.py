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

from prompt import prompt_master
from utils import WebContentExtractor
from logging_config import log_tool_usage
from text_chunker import TextChunker
from tools import ArxivTool, CalculatorTool, WebSearchTool
from prompt import prompt_master
MASTER_PROMPT = prompt_master


# 1. 添加新的Pydantic模型定义
class CalculatorParams(BaseModel):
    a: float
    b: float

class SearchParams(BaseModel):
    keyword: str
    max_results: int = Field(default=10)

class AgentDecision(BaseModel):
    next_agent: str = Field(
        ...,  # 表示必填字段
        description="下一个要执行的代理名称",
    )
    params: CalculatorParams | SearchParams




# 2. 状态定义
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    current_agent: str
    next_agent: Optional[str]
    final_answer: Optional[str]
    tools_output: Dict[str, Any]



# 4. 添加 LLM 配置类
class LLMConfig:
    @staticmethod
    def create_llm() -> ChatOpenAI:
        return ChatOpenAI(
            model='deepseek-chat', 
            openai_api_key='sk-9693411e1fcb4176ab62ed97f98c68f3', 
            openai_api_base='https://api.deepseek.com',
            temperature=0,
            max_tokens=4096,
        )


# 添加基础Agent类
class BaseAgent(ABC):
    
    @abstractmethod
    async def invoke(self, state: AgentState) -> AgentState:
        """基础invoke方法，子类需要重写这个方法"""
        raise NotImplementedError("子类必须实现invoke方法")


# 添加基础Agent类
class DeepSeekAgent(BaseAgent):
    
    def __init__(self):
        self.llm = LLMConfig.create_llm()


    async def invoke(self, state: AgentState) -> AgentState:
        pass



# 5. Agent 定义
class MasterAgent(DeepSeekAgent):
    def __init__(self):
        super().__init__()
        self.system_prompt = MASTER_PROMPT

    @log_tool_usage
    async def invoke(self, state: AgentState) -> AgentState:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": state["messages"][-1].content}
        ]
        
        response = await self.llm.ainvoke(messages)
        
        
        try:
            # 使用Pydantic模型解析和验证响应
            raw_decision = json.loads(response.content)
            decision = AgentDecision.model_validate(raw_decision)
            
            state["next_agent"] = decision.next_agent
            state["tools_output"]["params"] = decision.params.model_dump()
            return state
            
        except Exception as e:
            state["final_answer"] = response.content
            state["next_agent"] = None
            return state

class CalculatorAgent(DeepSeekAgent):
    def __init__(self):
        super().__init__()
        self.tool = CalculatorTool()
        
    @log_tool_usage
    async def invoke(self, state: AgentState) -> AgentState:
        params = state["tools_output"]["params"]
        result = self.tool.add(params["a"], params["b"])
        state["tools_output"]["result"] = result

        return state

class ArxivAgent(DeepSeekAgent):
    def __init__(self):
        super().__init__()
        self.tool = ArxivTool()

    @log_tool_usage    
    async def invoke(self, state: AgentState) -> AgentState:
        params = state["tools_output"]["params"]
        papers = await self.tool.search_papers(
            keyword=params["keyword"],
            max_results=params.get("max_results", 10)
        )
        state["tools_output"]["result"] = papers
        
        return state
    
class KeyPoint(BaseModel):
    point: str = Field(..., description="关键观点")
    importance: int = Field(default=1, description="重要性评分 1-5")

class WebSearchAgent(DeepSeekAgent):
    def __init__(self):
        super().__init__()
        self.tool = WebSearchTool()
    

    @log_tool_usage
    async def extract_keypoints(self, query: str) -> List[KeyPoint]:
        messages = [
            {"role": "system", "content": """请分析输入的查询文本,提取关键观点,转换为查询意图。
每个关键观点需包含:
- point: 具体的查询意图
- importance: 重要性评分(1-5)

以JSON数组格式返回,例如:
[
    {"point": "查询意图1", "importance": 5},
    {"point": "查询意图2", "importance": 3}
]"""},
            {"role": "user", "content": query}
        ]
        
        response = await self.llm.ainvoke(messages)
        keypoints = json.loads(response.content)
        
        return [KeyPoint.model_validate(kp) for kp in keypoints]
    

    @log_tool_usage
    async def invoke(self, state: AgentState) -> AgentState:
        # 从state中获取messages
        messages = state["messages"]
        
        # 提取最后一条用户消息作为查询
        last_message = messages[-1]
        if isinstance(last_message, HumanMessage):
            query = last_message.content
        else:
            raise ValueError("最后一条消息必须是用户消息")
            
        # 提取关键观点
        keypoints = await self.extract_keypoints(query)
        
        # 对每个关键观点进行搜索
        all_results = []
        for kp in sorted(keypoints, key=lambda x: x.importance, reverse=True):
            results = await self.tool.search(
                query=kp.point,
                max_results=1  # 每个关键点最多5个结果
            )
            all_results.extend(results)
            
        # 更新state
        state["tools_output"]["result"] = all_results[:10]  # 最多返回10个结果
        state["next_agent"] = 'websearch_agent'  # 搜索完成后结束
        
        return state
    



class ResultFormatter(DeepSeekAgent):
    
    @log_tool_usage
    async def invoke(self, state: AgentState) -> AgentState:
        result = state["tools_output"]["result"]
        messages = [
            {"role": "system", "content": "请将执行结果转换为用户友好的自然语言形式"},
            {"role": "user", "content": f"结果是: {result}"}
        ]
        
        response = await self.llm.ainvoke(messages)
        state["final_answer"] = response.content
        return state
    
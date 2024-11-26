import os

# 设置环境变量来控制线程数
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

from abc import ABC, abstractmethod
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict, Union

import operator
from enum import Enum
import json
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field

from logging_config import log_tool_usage
from tools import ArxivTool, CalculatorTool, WebSearchTool,LLMConfig,GeneratorTool
from tools import HallucinationCheckTool,ChunkAbstractTool
from tools import HallucinationCheckResult
from utils import LLMConfig
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
class SearchResult(TypedDict):
    original_count: int
    filtered_count: int
    filtered_results: List[Dict[str, Any]]
    generated_answer: str
    quality_check: Dict[str, Any]

class ToolsOutput(TypedDict):
    result: Union[SearchResult, Any]  # 可以是SearchResult或其他工具的结果
    params: Optional[Dict[str, Any]]

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    current_agent: str
    next_agent: Optional[str]
    final_answer: Optional[str]
    tools_output: ToolsOutput




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
        # self.relevance_checker = RelevanceCheckTool()
        self.generator = GeneratorTool()
        self.hallucination_checker = HallucinationCheckTool()
        self.chunk_abstracter = ChunkAbstractTool()
        self.max_results = 10
    

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
                max_results=self.max_results  # 每个关键点最多5个结果
            )
            all_results.extend(results)
            
            results = all_results
            
            # # 相关性检查
            # filtered_results = []
            # for result in results:
            #     relevance_check = await self.relevance_checker.check_relevance(
            #         query=state["messages"],
            #         document=result["content"]
            #     )
            
            #     if relevance_check.is_relevant and relevance_check.confidence > 0.6:
            #         result["relevance_check"] = relevance_check.model_dump()
            #         filtered_results.append(result)
            
            # # 在相关性检查后，为每个相关文档生成摘要
            # filtered_results_with_abstract = []
            # for result in filtered_results:
            #     abstract_result = await self.chunk_abstracter.create_abstract(
            #         query=state["messages"],
            #         chunk_content=result["content"],
            #         max_length=200
            #     )
            
            #     result["abstract"] = abstract_result.model_dump()
            #     # 使用摘要替换原始内容用于后续处理
            #     result["original_content"] = result["content"]
            #     result["content"] = abstract_result.abstract
            #     filtered_results_with_abstract.append(result)
            
            
            
            # 生成答案
            if all_results:
                answer = await self.generator.generate_answer(
                    query=state["messages"],
                    relevant_docs=all_results
                )
                
                # 检查答案质量
                quality_check = await self.hallucination_checker.check_answer(
                    query=state["messages"],
                    answer=answer,
                    relevant_docs=all_results
                )
                
                if not quality_check.is_valid or quality_check.confidence < 0.7:
                    # 如果答案质量不够好，尝试重新生成
                    print("首次生成的答案质量不足，尝试重新生成...")
                    answer = await self.generator.generate_answer(
                        query=state["messages"],
                        relevant_docs=all_results,
                        max_length=1500  # 增加长度限制以获得更详细的答案
                    )
                    # 再次检查质量
                    quality_check = await self.hallucination_checker.check_answer(
                        query=state["messages"],
                        answer=answer,
                        relevant_docs=all_results
                    )
            else:
                answer = "未找到相关的文档内容来回答该问题。"
                quality_check = HallucinationCheckResult(
                    is_valid=False,
                    confidence=0.0,
                    issues=["没有找到相关文档"],
                    suggestions=["尝试使用不同的关键词搜索"]
                )


        # 更新state
        state["tools_output"]["result"] = {
                "original_count": len(results),
                "filtered_count": len(all_results),
                "filtered_results": all_results,
                "generated_answer": answer,
                "quality_check": quality_check.model_dump()
            }
        
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
    


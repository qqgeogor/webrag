import os

# 设置环境变量来控制线程数
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

from abc import ABC, abstractmethod
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict, Union

import hashlib
import operator
from enum import Enum
import json
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field
from itertools import islice
import asyncio

from logging_config import log_tool_usage
from tools import ArxivTool, CalculatorTool, WebSearchTool,LLMConfig,GeneratorTool,RelevanceCheckTool
from tools import HallucinationCheckTool,ChunkAbstractTool
from tools import HallucinationCheckResult
from evaluation import RagasEvaluator
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
        self._llm = LLMConfig.create_llm()

    def _get_llm(self):
        return self._llm
    
    @property
    def llm(self):
        return self._get_llm()  
    
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

        state["tools_output"]["result"] = {
                "original_count": 0,
                "filtered_count": 0,
                "filtered_results": [],  # 使用包含 URL 的结果
                "generated_answer": result,
                "quality_check": {}
            }
        
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
        
        # state["tools_output"]["result"] = 
        state["tools_output"]["result"] = {
                "original_count": 0,
                "filtered_count": 0,
                "filtered_results": [],  # 使用包含 URL 的结果
                "generated_answer": papers,
                "quality_check": {}
            }
        
        
        return state
    
class KeyPoint(BaseModel):
    point: str = Field(..., description="关键观点")
    importance: int = Field(default=1, description="重要性评分 1-5")

class WebSearchAgent(DeepSeekAgent):
    def __init__(self):
        super().__init__()
        self.tool = WebSearchTool()
        self.relevance_checker = RelevanceCheckTool()
        self.generator = GeneratorTool()
        self.hallucination_checker = HallucinationCheckTool()
        self.chunk_abstracter = ChunkAbstractTool()
        self.max_results = 5
        self.similarity_threshold = 0.75  # 相似度阈值
        self.max_concurrent = 16  # 最大并发数
        # self.semaphore = asyncio.Semaphore(self.max_concurrent)
        # self.search_semaphore = asyncio.Semaphore(self.max_concurrent)
        self.ragas_evaluator = RagasEvaluator()

    @log_tool_usage
    async def extract_keypoints(self, query: str) -> List[KeyPoint]:
        messages = [
            {"role": "system", "content": """请分析输入的查询文本,提取关键且互不重复的查询意图。要求：

1. 每个意图必须表达不同的查询方向
2. 避免提取语义重复或高度相似的意图
3. 合并相近的查询意图为一个更全面的表述
4. 确保意图之间的区分度

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

    async def calculate_minhash_similarity(self, text1: str, text2: str, num_perm: int = 128) -> float:
        """使用MinHash计算两段文本的相似度"""
        def get_shingles(text: str, k: int = 3):
            """获取文本的k-shingles"""
            text = text.lower()
            return set(' '.join(text[i:i+k]) for i in range(len(text)-k+1))
            
        def create_minhash(shingles: set, num_perm: int):
            """创建MinHash签名"""
            minhash = [float('inf')] * num_perm
            for shingle in shingles:
                hash_val = int(hashlib.md5(shingle.encode()).hexdigest(), 16)
                for i in range(num_perm):
                    minhash[i] = min(minhash[i], (hash_val * (i + 1)) % (2**32))
            return minhash
            
        # 获取shingles
        shingles1 = get_shingles(text1)
        shingles2 = get_shingles(text2)
        
        # 创建MinHash签名
        minhash1 = create_minhash(shingles1, num_perm)
        minhash2 = create_minhash(shingles2, num_perm)
        
        # 计算相似度
        same_count = sum(1 for i in range(num_perm) if minhash1[i] == minhash2[i])
        similarity = same_count / num_perm
        
        return similarity

    async def merge_similar_keypoints(self, keypoints: List[KeyPoint]) -> List[KeyPoint]:
        """合并相似的关键点"""
        if not keypoints:
            return keypoints

        merged_points = []
        used_indices = set()

        for i, kp1 in enumerate(keypoints):
            if i in used_indices:
                continue
                
            similar_points = []
            for j, kp2 in enumerate(keypoints[i+1:], i+1):
                if j in used_indices:
                    continue
                
                similarity = await self.calculate_minhash_similarity(kp1.point, kp2.point)
                if similarity > self.similarity_threshold:
                    similar_points.append(kp2)
                    used_indices.add(j)

            if similar_points:
                # 合并相似的关键点
                messages = [
                    {"role": "system", "content": "请将以下相似的查询意图合并为一个更全面的表述，保留最高的重要性分数。"},
                    {"role": "user", "content": f"意图列表：\n" + "\n".join([kp1.point] + [kp.point for kp in similar_points])}
                ]
                response = await self.llm.ainvoke(messages)
                merged_point = KeyPoint(
                    point=response.content.strip(),
                    importance=max([kp1.importance] + [kp.importance for kp in similar_points])
                )
                merged_points.append(merged_point)
            else:
                merged_points.append(kp1)

        return merged_points

    @log_tool_usage
    async def check_single_relevance(self, query: str, result: dict) -> Optional[dict]:
        """检查单个文档的相关性"""
        # async with self.semaphore:  # 使用信号量控制并发
        try:
            relevance_check = await self.relevance_checker.check_relevance(
                query=query,
                document=result["content"]
            )
            
            if relevance_check.is_relevant and relevance_check.confidence > 0.6:
                result["relevance_check"] = relevance_check.model_dump()
                return result
            return None
        except Exception as e:
            print(f"Error checking relevance: {e}")
            return None
    
    async def batch_check_relevance(self, query: str, results: List[dict], batch_size: int = 20) -> List[dict]:
        """批量检查文档相关性"""
        filtered_results = []
        
        # 将结果分批处理
        for i in range(0, len(results), batch_size):
            batch = list(islice(results, i, i + batch_size))
            
            # 创建当前批次的任务
            tasks = [
                self.check_single_relevance(query, result)
                for result in batch
            ]
            
            # 并发执行当前批次的任务
            batch_results = await asyncio.gather(*tasks)
            
            # 过滤掉None结果
            filtered_results.extend([r for r in batch_results if r is not None])
            
        return filtered_results
    
    @log_tool_usage
    async def search_single_keypoint(self, keypoint: KeyPoint) -> List[dict]:
        """搜索单个关键点"""
        # async with self.search_semaphore:
        try:
            results = await self.tool.search(
                query=keypoint.point,
                max_results=self.max_results
            )
            # 添加重要性信息到结果中，方便后续处理
            for result in results:
                result['importance'] = keypoint.importance
            return results
        except Exception as e:
            print(f"Error searching for keypoint {keypoint.point}: {e}")
            return []

    async def parallel_search(self, keypoints: List[KeyPoint]) -> List[dict]:
        """并行执行多个关键点的搜索"""
        # 按重要性排序关键点
        sorted_keypoints = sorted(keypoints, key=lambda x: x.importance, reverse=True)
        
        # 创建搜索任务
        search_tasks = [
            self.search_single_keypoint(kp)
            for kp in sorted_keypoints
        ]
        
        # 并行执行搜索
        results_list = await asyncio.gather(*search_tasks)
        
        # 合并结果，保持重要性顺序
        all_results = []
        for results in results_list:
            all_results.extend(results)
            
        return all_results
    

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
            
        # 提取关键观点后添加相似度检查和合并
        keypoints = await self.extract_keypoints(query)
        merged_keypoints = await self.merge_similar_keypoints(keypoints)
        
        # 并行执行搜索
        all_results = await self.parallel_search(merged_keypoints)
            
        # 根据 URL 去重
        unique_results = {}
        for result in all_results:
            url = result.get("url")
            if url and url not in unique_results:
                unique_results[url] = result

        # 将去重后的结果转换为列表
        all_results = list(unique_results.values())
        
        # 相关性检查
        filtered_results = await self.batch_check_relevance(
            query=query,
            results=all_results,
            batch_size=2  # 每批处理20个文档
        )

        # 生成答案
        if filtered_results:
            answer = await self.generator.generate_answer(
                query=state["messages"],
                relevant_docs=filtered_results
            )
            
            # 检查答案质量
            quality_check = await self.hallucination_checker.check_answer(
                query=state["messages"],
                answer=answer,
                relevant_docs=filtered_results
            )
            
            if not quality_check.is_valid or quality_check.confidence < 0.7:
                # 如果答案质量不够好，尝试重新生成
                print("首次生成的答案质量不足，尝试重新生成...")
                answer = await self.generator.generate_answer(
                    query=state["messages"],
                    relevant_docs=filtered_results,
                    max_length=1500  # 增加长度限制以获得更详细的答案
                )
                # 再次检查质量
                quality_check = await self.hallucination_checker.check_answer(
                    query=state["messages"],
                    answer=answer,
                    relevant_docs=filtered_results
                )
        else:
            answer = "未找到相关的文档内容来回答该问题。"
            quality_check = HallucinationCheckResult(
                is_valid=False,
                confidence=0.0,
                issues=["没有找到相关文档"],
                suggestions=["尝试使用不同的关键词搜索"]
            )

        # 更新 state 时添加 url
        filtered_results_with_urls = []
        for result in filtered_results:
            result_with_url = {
                "content": result["content"],
                "url": result["url"],  # 添加 URL
                "relevance_check": result.get("relevance_check", {}),
                "importance": result.get("importance", 1)
            }
            filtered_results_with_urls.append(result_with_url)

        # 更新 state
        state["tools_output"]["result"] = {
                "original_count": len(all_results),
                "filtered_count": len(filtered_results),
                "filtered_results": filtered_results_with_urls,  # 使用包含 URL 的结果
                "generated_answer": answer,
                "quality_check": quality_check.model_dump()
            }
        

        # # 评估答案
        # eval_result = await self.ragas_evaluator.evaluate(
        #     question=query,
        #     answer=answer,
        #     contexts=filtered_results_with_urls
        # )

        # state["eval_result"] = eval_result


        state["next_agent"] = 'websearch_agent'
        state["final_answer"] = answer
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
    
    


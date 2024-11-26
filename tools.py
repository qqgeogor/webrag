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
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions



# 1. 基础模型定义
class PaperInfo(BaseModel):
    title: str
    abstract: str
    pdf_url: str
    authors: List[str]


# 3. 工具定义
class ArxivTool:
    def __init__(self):
        self.name = "arxiv_search"
        self.description = "搜索arxiv论文的工具"

    async def search_papers(self, keyword: str, max_results: int = 10) -> List[PaperInfo]:
        client = arxiv.Client()
        search = arxiv.Search(
            query=keyword,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )

        papers = []
        async with httpx.AsyncClient() as http_client:
            for result in client.results(search):
                paper = PaperInfo(
                    title=result.title,
                    abstract=result.summary,
                    pdf_url=result.pdf_url,
                    authors=[author.name for author in result.authors]
                )
                papers.append(paper)
        return papers

class CalculatorTool:
    def __init__(self):
        self.name = "calculator"
        self.description = "执行基础数学运算"

    def add(self, a: float, b: float) -> float:
        return a + b
    

class WebSearchParams(BaseModel):
    query: str = Field(..., description="搜索关键词")
    max_results: int = Field(default=5, description="最大返回结果数")

class WebSearchTool:
    def __init__(self):
        self.name = "web_search"
        self.description = "使用Serper.dev搜索网页内容的工具"
        # 替换为你的 Serper API key
        self.api_key = "6d8a07a4c3dbecf40509ae64a9461b38c2c1b99f"
        self.base_url = "https://google.serper.dev/search"
        self.extractor = WebContentExtractor()

    async def fetch_webpage_content(self, url: str) -> str:
        """获取网页内容"""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, timeout=10.0)
                response.raise_for_status()
                return await self.extractor.extract_content(response.text)
            except Exception as e:
                return f"无法获取网页内容: {str(e)}"

    @log_tool_usage
    async def search(self, query: str, max_results: int = 5) -> List[dict]:
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }
        
        payload = {
            "q": query,
            "num": max_results
        }
        
        results = []
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.base_url,
                headers=headers,
                json=payload
            )
            
            if response.status_code != 200:
                raise Exception(f"搜索API调用失败: {response.text}")
                
            data = response.json()
            organic_results = data.get("organic", [])
            
            for result in organic_results[:max_results]:
                content = await self.fetch_webpage_content(result.get("link", ""))
                results.append({
                    "title": result.get("title", ""),
                    "content": content,
                    "link": result.get("link", ""),
                    "position": result.get("position", 0)
                })
            
            return results

class ChromaEmbeddingTool:
    def __init__(self, collection_name: str = "default_collection"):
        self.name = "embedding_tool"
        self.description = "用于文档的embedding存储和检索的工具"
        
        # 初始化ChromaDB客户端
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="./chroma_db"
        ))
        
        # 使用OpenAI的embedding函数
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key="your_openai_api_key",
            model_name="text-embedding-ada-002"
        )
        
        # 获取或创建collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )

    @log_tool_usage
    async def add_documents(self, texts: List[str], metadata: Optional[List[dict]] = None) -> List[str]:
        """
        添加文档到向量数据库
        
        Args:
            texts: 要添加的文本列表
            metadata: 对应的元数据列表
        Returns:
            文档ID列表
        """
        # 生成唯一ID
        ids = [f"doc_{i}_{hash(text)}" for i, text in enumerate(texts)]
        
        # 如果没有提供metadata，创建空的metadata
        if metadata is None:
            metadata = [{} for _ in texts]
            
        # 添加文档
        self.collection.add(
            documents=texts,
            metadatas=metadata,
            ids=ids
        )
        
        return ids

    @log_tool_usage
    async def query_similar(self, 
                          query_text: str, 
                          n_results: int = 5,
                          where: Optional[dict] = None) -> List[dict]:
        """
        查询相似文档
        
        Args:
            query_text: 查询文本
            n_results: 返回结果数量
            where: 过滤条件
        Returns:
            相似文档列表
        """
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where
        )
        
        # 格式化返回结果
        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i],
                'id': results['ids'][0][i]
            })
            
        return formatted_results

    @log_tool_usage
    async def delete_documents(self, ids: List[str]) -> None:
        """删除指定ID的文档"""
        self.collection.delete(ids=ids)

    @log_tool_usage
    async def update_documents(self, 
                             ids: List[str], 
                             texts: List[str], 
                             metadata: Optional[List[dict]] = None) -> None:
        """更新指定ID的文档"""
        if metadata is None:
            metadata = [{} for _ in texts]
            
        self.collection.update(
            ids=ids,
            documents=texts,
            metadatas=metadata
        )

    def get_collection_stats(self) -> dict:
        """获取collection的统计信息"""
        return {
            "count": self.collection.count(),
            "name": self.collection.name
        }
from abc import ABC, abstractmethod
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict, Literal

import chromadb
import arxiv
import httpx
from pydantic import BaseModel, Field

from prompt import prompt_master
from db_helper import DBHelper,StorageType
from utils import WebContentExtractor
from logging_config import log_tool_usage


from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import networkx as nx
import re
import json
import hashlib
from utils import LLMConfig


class SemanticChunk(BaseModel):
    content: str
    start_idx: int
    end_idx: int
    coherence_score: float
    sentences: List[str]
    url: str

class SemanticChunker:
    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        min_chunk_size: int = 512,
        max_chunk_size: int = 1024,
        overlap_size: int = 50,
        embedding_model: SentenceTransformer = None
    ):
        if embedding_model is None:
            self.model = SentenceTransformer(model_name,cache_folder='./cache')
        else:
            self.model = embedding_model
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        
    def _split_into_sentences(self, text: str) -> List[str]:
        """将文本分割为句子"""
        # 处理中英文混合的情况
        text = re.sub(r'([。！？\?!])\s*', r'\1\n', text)
        sentences = [s.strip() for s in text.split('\n') if s.strip()]
        return sentences
        
    def _calculate_coherence(
        self,
        embeddings: np.ndarray,
        start_idx: int,
        end_idx: int
    ) -> float:
        """计算文本块的语义连贯性"""
        if end_idx - start_idx < 2:
            return 1.0
            
        chunk_embeddings = embeddings[start_idx:end_idx]
        similarities = np.inner(chunk_embeddings, chunk_embeddings)
        # 计算相邻句子的平均相似度
        coherence = np.mean([
            similarities[i][i+1] 
            for i in range(len(similarities)-1)
        ])
        return float(coherence)

    async def chunk_by_semantic_similarity(
        self,
        text: str,
        url: str
    ) -> List[SemanticChunk]:
        """基于语义相似度的动态分块"""
        sentences = self._split_into_sentences(text)
        if not sentences:
            return []
            
        # 计算句子嵌入
        embeddings = self.model.encode(sentences)
        
        # 使用层次聚类
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0.5,
            linkage='ward'
        )
        
        
        # 重塑embeddings为2D数组
        embeddings = embeddings.reshape(-1, embeddings.shape[-1])
        
        # 检查embeddings维度
        if len(embeddings.shape) <= 1:
            # 如果只有一个句子,直接返回单个chunk
            return [SemanticChunk(
                content=text,
                start_idx=0,
                end_idx=1,
                coherence_score=1.0,
                sentences=sentences,
                url=url
            )]
            
        clusters = clustering.fit_predict(embeddings)
        
        # 根据聚类结果生成chunks
        chunks = []
        current_chunk = []
        current_start = 0
        
        for i, (sentence, cluster) in enumerate(zip(sentences, clusters)):
            current_chunk.append(sentence)
            chunk_text = ' '.join(current_chunk)
            
            # 检查是否需要切分
            if (len(chunk_text) >= self.max_chunk_size or  # 达到最大长度
                (i < len(sentences)-1 and clusters[i] != clusters[i+1] and  # 聚类边界
                 len(chunk_text) >= self.min_chunk_size)):  # 达到最小长度
                
                coherence = self._calculate_coherence(
                    embeddings,
                    current_start,
                    i + 1
                )
                
                chunks.append(SemanticChunk(
                    content=chunk_text,
                    start_idx=current_start,
                    end_idx=i + 1,
                    coherence_score=coherence,
                    sentences=current_chunk,
                    url=url
                ))
                
                current_chunk = []
                current_start = i + 1
        
        # 处理最后一个chunk
        if current_chunk:
            coherence = self._calculate_coherence(
                embeddings,
                current_start,
                len(sentences)
            )

            chunks.append(SemanticChunk(
                content=' '.join(current_chunk),
                start_idx=current_start,
                end_idx=len(sentences),
                coherence_score=coherence,
                sentences=current_chunk,
                url=url
            ))
        
        return chunks


    async def chunk_by_topic_segmentation(
        self,
        text: str,
        url: Optional[str] = None
    ) -> List[SemanticChunk]:
        """基于主题分割的动态分块"""
        sentences = self._split_into_sentences(text)
        if not sentences:
            return []
            
        # 计算句子嵌入
        embeddings = self.model.encode(sentences)
        
        # 构建相似度图
        similarity_matrix = np.inner(embeddings, embeddings)
        G = nx.Graph()
        
        # 添加边，权重为相似度
        for i in range(len(sentences)):
            for j in range(i + 1, len(sentences)):
                similarity = similarity_matrix[i][j]
                if similarity > 0.5:  # 相似度阈值
                    G.add_edge(i, j, weight=similarity)
        
        # 使用社区检测算法
        communities = nx.community.louvain_communities(G)
        
        # 根据社区划分生成chunks
        chunks = []
        for community in communities:
            community = sorted(community)
            if not community:
                continue
                
            # 获取社区内的句子
            community_sentences = [sentences[i] for i in community]
            chunk_text = ' '.join(community_sentences)
            
            # 如果chunk太大，进行二次划分
            if len(chunk_text) > self.max_chunk_size:
                sub_chunks = self._split_large_chunk(
                    community_sentences,
                    embeddings[community],
                    url,
                )
                chunks.extend(sub_chunks)
            else:
                coherence = self._calculate_coherence(
                    embeddings,
                    community[0],
                    community[-1] + 1
                )
                
                chunks.append(SemanticChunk(
                    content=chunk_text,
                    start_idx=community[0],
                    end_idx=community[-1] + 1,
                    coherence_score=coherence,
                    sentences=community_sentences,
                    url=url
                ))
        
        return sorted(chunks, key=lambda x: x.start_idx)

    def _split_large_chunk(
        self,
        sentences: List[str],
        embeddings: np.ndarray,
        url: Optional[str] = None
    ) -> List[SemanticChunk]:
        """将大块文本进行二次划分"""
        chunks = []
        current_chunk = []
        current_start = 0
        current_length = 0
        
        for i, sentence in enumerate(sentences):
            current_chunk.append(sentence)
            current_length += len(sentence)
            
            if current_length >= self.max_chunk_size:
                coherence = self._calculate_coherence(
                    embeddings,
                    current_start,
                    i + 1
                )
                
                chunks.append(SemanticChunk(
                    content=' '.join(current_chunk),
                    start_idx=current_start,
                    end_idx=i + 1,
                    coherence_score=coherence,
                    sentences=current_chunk,
                    url=url
                ))
                
                current_chunk = []
                current_start = i + 1
                current_length = 0
        
        # 处理剩余部分
        if current_chunk:
            coherence = self._calculate_coherence(
                embeddings,
                current_start,
                len(sentences)
            )
            
            chunks.append(SemanticChunk(
                content=' '.join(current_chunk),
                start_idx=current_start,
                end_idx=len(sentences),
                coherence_score=coherence,
                sentences=current_chunk,
                url=url
            ))
        
        return chunks

# PaperInfo基础模型定义
class PaperInfo(BaseModel):
    title: str
    abstract: str
    pdf_url: str
    authors: List[str]


# ArxivTool工具定义
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
        for result in client.results(search):
            paper = PaperInfo(
                title=result.title,
                abstract=result.summary,
                pdf_url=result.pdf_url,
                authors=[author.name for author in result.authors]
            )
            papers.append(paper)

        return papers


#  Calculator Tool,用于执行基础数学运算
class CalculatorTool:
    def __init__(self):
        self.name = "calculator"
        self.description = "执行基础数学运算"

    def add(self, a: float, b: float) -> float:
        return a + b
    


# Add new models for unified search results
class SearchDocument(BaseModel):
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    distance: Optional[float] = None
    url: Optional[str] = None

class SearchResult(BaseModel):
    documents: List[SearchDocument]
    total: int

# Modify HybridSearch class
class HybridSearch:
    def __init__(
        self,
        storage_type: StorageType = StorageType.MYSCALE,
        collection_name: str = "web_search_content",
        embedding_model=None
    ):
        self.storage_type = storage_type
        self.collection_name = collection_name

        if storage_type == StorageType.CHROMA:
            self._chroma_client = chromadb.PersistentClient(path="./chroma_db")
            self._collection = self._chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        else:
            self.db_helper = DBHelper(storage_type)
  

        
        self.min_results = 3  # 最少期望结果数
        self.initial_threshold = 0.7  # 初始相似度阈值
        
        if embedding_model is None:
            model_name = 'all-MiniLM-L6-v2'
            self.model = SentenceTransformer(model_name,cache_folder='./cache')
        else:
            self.model = embedding_model

    def _get_embedding(self, text: str) -> np.ndarray:
        """获取文本的embedding向量"""
        return self.model.encode(text)


    @classmethod
    async def create(
        cls,
        storage_type: StorageType = StorageType.MYSCALE,
        collection_name: str = "web_search_content",
        embedding_model=None
    ):
        instance = cls(storage_type, collection_name, embedding_model)
        
        instance.db_helper.create_table(collection_name)
        
        return instance

    def _get_collection(self):
        return self._collection if self.storage_type == StorageType.CHROMA else self._myscale_client

    @property
    def collection(self):
        return self._get_collection()

    async def hybrid_search(
        self, 
        query: str, 
        top_k: int = 5, 
        alpha: float = 0.9
    ) -> SearchResult:
        try:
            query_embedding = self._get_embedding(query)
            
            if self.storage_type == StorageType.CHROMA:
                raw_results = self.collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=top_k
                )
                
                documents = [
                    SearchDocument(
                        id=doc_id,
                        content=content,
                        metadata=metadata,
                        distance=distance,
                        url=metadata.get("url")
                    )
                    for doc_id, content, metadata, distance in zip(
                        raw_results['ids'][0],
                        raw_results['documents'][0],
                        raw_results['metadatas'][0],
                        raw_results['distances'][0]
                    )
                ]
                
            else:  # MyScale
                
                
                results = self.db_helper.query_hybrid(query,query_embedding,self.collection_name,top_k)
                
                documents = [
                    SearchDocument(
                        id=row['id'],
                        content=row['content'],
                        metadata=json.loads(row['metadata']),
                        distance=float(row['distance']),
                        url=row['url']
                    )
                    for row in results
                ]
            
            # Sort by score and apply threshold
            documents.sort(key=lambda x: 1 - (x.distance or 0), reverse=True)
            
            threshold = self.initial_threshold
            while threshold > 0:
                filtered_docs = [
                    doc for doc in documents 
                    if (1 - (doc.distance or 0)) >= threshold
                ]
                
                if len(filtered_docs) >= self.min_results:
                    return SearchResult(
                        documents=filtered_docs[:top_k],
                        total=len(filtered_docs)
                    )
                    
                threshold -= 0.1

            return SearchResult(
                documents=documents[:top_k],
                total=len(documents)
            )
            
        except Exception as e:
            print(f"检索过程中出错: {str(e)}")
            return SearchResult(documents=[], total=0)

class WebSearchParams(BaseModel):
    query: str = Field(..., description="搜索关键词")
    max_results: int = Field(default=5, description="最大返回结果数")

class WebSearchTool:
    def __init__(self, storage_type: StorageType = StorageType.MYSCALE):
        self.name = "web_search"
        self.description = "使用Serper.dev搜索网页内容的工具"
        self.api_key = "6d8a07a4c3dbecf40509ae64a9461b38c2c1b99f"
        self.base_url = "https://google.serper.dev/search"
        self.extractor = WebContentExtractor()
        self.model_name: str = 'all-MiniLM-L6-v2'
        
        self.embedding_model = SentenceTransformer(self.model_name,cache_folder='./cache')
        self.semantic_chunker = SemanticChunker(
            min_chunk_size=100,
            max_chunk_size=1000,
            overlap_size=50,
            embedding_model = self.embedding_model
        )
        
        self._client = httpx.AsyncClient()
        self.storage_type = storage_type

    @classmethod
    async def create(cls, storage_type: StorageType = StorageType.MYSCALE):
        instance = cls(storage_type)
        instance.hybrid_search = await HybridSearch.create(
            storage_type=storage_type,
            collection_name="web_search_content",
            embedding_model=instance.embedding_model
        )
        return instance

    @classmethod
    async def search(cls, query: str, max_results: int = 5) -> List[dict]:
        instance = await cls.create()
        return await instance._search(query, max_results)

    async def _search(self, query: str, max_results: int = 5) -> List[dict]:
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }
        
        payload = {
            "q": query,
            "num": max_results
        }
        
        results = []

        response = await self.client.post(
            self.base_url,
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            raise Exception(f"搜索API调用失败: {response.text}")
                
        data = response.json()
        organic_results = data.get("organic", [])
        
        for result in organic_results[:max_results]:
            chunks = await self.fetch_and_process_content(
                result.get("link", ""),
                result.get("title", "")
            )
            
            if chunks:
                results.extend(chunks)
        
        search_results = await self.hybrid_search.hybrid_search(query, max_results)
        
        results = [
            {
                "content": doc.content,
                "url": doc.url or doc.metadata.get("url", "无可用URL"),
                "metadata": doc.metadata
            }
            for doc in search_results.documents
        ]
        
        return results

    def _get_client(self):
        return self._client 
    
    @property
    def client(self):
        return self._get_client()   

    async def fetch_and_process_content(self, url: str, title: str) -> List[dict]:
        """获取网页内容并进行语义分块"""
        try:
            # 获取原始内容
            response = await self.client.get(url, timeout=10.0)
            response.raise_for_status()
            content = await self.extractor.extract_content(response.text)
                
            # 使用 SemanticChunker 进行分块，传入URL
            chunks = await self.semantic_chunker.chunk_by_semantic_similarity(content, url)
            
            # 准备数据
            documents = []
            metadatas = []
            ids = []
            
            for i, chunk in enumerate(chunks):
                chunk_id = hashlib.md5(f"{url}_{i}".encode()).hexdigest()
                documents.append(chunk.content)
                metadatas.append({
                    "url": url,
                    "title": title,
                    "chunk_index": i,
                    "coherence_score": chunk.coherence_score
                })
                ids.append(chunk_id)
            
            # 使用 SentenceTransformer 生成 embeddings
            if documents:
                embeddings = self.semantic_chunker.model.encode(documents).tolist()  # 转换为列表以便序列化
                
                if self.storage_type == StorageType.CHROMA:
                    self.collection.add(
                        documents=documents,
                        metadatas=metadatas,
                        ids=ids,
                        embeddings=embeddings
                    )
                else:  # MyScale
                    # 批量插入数据
                    try:
                        self.hybrid_search.db_helper.insert(ids, documents, metadatas, embeddings)
                    except Exception as e:
                        print(f"MyScale插入数据失败: {str(e)}")
                
                return [{"id": id, "content": doc, "metadata": meta, "url": url} 
                       for id, doc, meta in zip(ids, documents, metadatas)]
                   
        except Exception as e:
            print(f"处理网页内容时出错: {str(e)}")
            return []

class RelevanceCheckResult(BaseModel):
    is_relevant: bool
    confidence: float
    reason: str

class RelevanceCheckTool:
    def __init__(self):
        self.llm = LLMConfig.create_llm()
        self.system_prompt = """你是一个专门判断文档相关性的助手。你需要判断给定的文档片段是否与用户查询相关。
        
请以JSON格式返回结果，包含以下字段：
{
    "is_relevant": true/false,  # 布尔值，表示是否相关
    "confidence": 0.0-1.0,     # 浮点数，表示置信度
    "reason": "string"         # 字符串，说明判断理由
}

判断标准：
1. 内容相关性：文档内容是否直接回答或涉及查询主题
2. 主题匹配：关键概念是否对应
3. 信息价值：内容是否提供有价值的信息

只有当内容确实与查询高度相关时才返回true。"""

    def _clean_json_response(self, content: str) -> str:
        """清理LLM响应中的JSON内容"""
        # 移除可能的markdown代码块标记
        content = content.strip()
        if content.startswith('```json'):
            content = content[7:]
        elif content.startswith('```'):
            content = content[3:]
            
        if content.endswith('```'):
            content = content[:-3]
            
        # 清理额外的换行和空格
        content = content.strip()
        
        return content

    async def check_relevance(
        self, 
        query: str, 
        document: str
    ) -> RelevanceCheckResult:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"""
请判断以下文档片段是否与查询相关：

查询：{query}

文档片段：{document}

请以JSON格式返回判断结果。"""}
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            content = self._clean_json_response(response.content)
            
            try:
                result = RelevanceCheckResult.model_validate_json(content)
                return result
            except Exception as e:
                print(f"Pydantic解析错误: {str(e)}")
                print(f"清理后的响应: {content}")
                return RelevanceCheckResult(
                    is_relevant=False,
                    confidence=0.0,
                    reason=f"解析失败: {str(e)}"
                )
            
        except Exception as e:
            print(f"相关性检查出错: {str(e)}")
            return RelevanceCheckResult(
                is_relevant=False,
                confidence=0.0,
                reason=f"处理出错: {str(e)}"
            )

class GeneratorTool:
    def __init__(self):
        self.llm = LLMConfig.create_llm()
        self.system_prompt = """你是一个专业的学术助手。你的任务是基于检索到的相关文档片段，为用户的查询生成准确、全面的回答。

要求：
1. 综合所有相关文档的信息
2. 保持答案的准确性和学术性
3. 如果文档中包含相互矛盾的信息，请指出并解释
4. 如果信息不足以完全回答问题，请明确指出
5. 适当引用原文内容，并标注来源

请确保回答：
- 直接针对用户的问题
- 结构清晰
- 逻辑连贯
- 有理有据"""

    async def generate_answer(
        self,
        query: str,
        relevant_docs: List[Dict],
        max_length: int = 1000
    ) -> str:
        # 构建上下文
        context = "\n\n".join([
            f"文档 {i+1} (\n{doc['content']}"
            for i, doc in enumerate(relevant_docs)
        ])
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"""
用户查询：{query}

相关文档内容：
{context}

请基于以上信息生成回答。回答长度不要超过{max_length}字。"""}
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            return response.content
        except Exception as e:
            print(f"生成答案时出错: {str(e)}")
            return f"生成答案时发生错误: {str(e)}"

class HallucinationCheckResult(BaseModel):
    is_valid: bool = Field(description="答案是否有效回答了问题")
    confidence: float = Field(ge=0.0, le=1.0, description="判断的置信度")
    issues: List[str] = Field(default_factory=list, description="发现的问题列表")
    suggestions: List[str] = Field(default_factory=list, description="改进建议")

class HallucinationCheckTool:
    def __init__(self):
        self.llm = LLMConfig.create_llm()
        self.system_prompt = """你是一个专门检测答案质量的助手。你需要判断生成的答案是真实可靠地回答了用户的问题。

请严格按照以下JSON格式返回结果：
{
    "is_valid": true/false,      # 答案是否有效
    "confidence": 0.0-1.0,       # 判断的置信度
    "issues": ["问题1", "问题2"], # 发现的具体问题
    "suggestions": ["建议1", "建议2"] # 改进建议
}

评估标准：
1. 答案是否直接回答问题
2. 内容是否完全基于提供的文档
3. 是否存在未经证实的推测
4. 是否存在与文档矛盾的内容
5. 答案的完整性和准确性"""

    def _clean_json_response(self, content: str) -> str:
        """清理LLM响应中的JSON内容"""
        content = content.strip()
        if content.startswith('```json'):
            content = content[7:]
        elif content.startswith('```'):
            content = content[3:]
        if content.endswith('```'):
            content = content[:-3]
        return content.strip()

    async def check_answer(
        self,
        query: str,
        answer: str,
        relevant_docs: List[Dict]
    ) -> HallucinationCheckResult:
        # 构建上下文
        context = "\n\n".join([
            f"文档 {i+1}:\n{doc['content']}"
            for i, doc in enumerate(relevant_docs)
        ])
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"""
用户问题：{query}

生成的答案：{answer}

参考文档：
{context}

请评估这个答案的质量并返回JSON格式的结果。"""}
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            content = self._clean_json_response(response.content)
            
            try:
                result = HallucinationCheckResult.model_validate_json(content)
                return result
            except Exception as e:
                print(f"Pydantic解析错误: {str(e)}")
                print(f"清理后的响应: {content}")
                return HallucinationCheckResult(
                    is_valid=False,
                    confidence=0.0,
                    issues=[f"解析失败: {str(e)}"],
                    suggestions=["需要重新生成答案"]
                )
                
        except Exception as e:
            print(f"答案质量检查出错: {str(e)}")
            return HallucinationCheckResult(
                is_valid=False,
                confidence=0.0,
                issues=[f"处理出错: {str(e)}"],
                suggestions=["需要重新生成答案"]
            )

class ChunkAbstractResult(BaseModel):
    abstract: str = Field(description="提炼的核心内容")
    key_points: List[str] = Field(description="关键要点列表")
    relevance_aspects: List[str] = Field(description="与查询相关的方面")
    original_length: int = Field(description="原文长度")
    abstract_length: int = Field(description="摘要长度")

class ChunkAbstractTool():
    def __init__(self):
        self.llm = LLMConfig.create_llm()
        self.system_prompt = """你是一个专门提炼文档核心内容的助手。你需要将长文本片段压缩为简短但信息丰富的摘要。

请严格按照以下JSON格式返回结果：
{
    "abstract": "提炼的核心内容，保持简洁但包含关键信息",
    "key_points": ["关键点1", "关键点2", ...],
    "relevance_aspects": ["与查询相关的方面1", "相关方面2", ...],
    "original_length": 原文字数,
    "abstract_length": 摘要字数
}

提炼要：
1. 保留最重要的信息和关键概念
2. 确保与用户查询相关的内容被优先保留
3. 去除冗余和次要信息
4. 保持逻辑连贯性
5. 控制摘要长度在原文的1/3以内"""

    def _clean_json_response(self, content: str) -> str:
        """清理LLM响应中的JSON内容"""
        content = content.strip()
        if content.startswith('```json'):
            content = content[7:]
        elif content.startswith('```'):
            content = content[3:]
        if content.endswith('```'):
            content = content[:-3]
        return content.strip()

    async def create_abstract(
        self,
        query: str,
        chunk_content: str,
        max_length: int = 200
    ) -> ChunkAbstractResult:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"""
请为以下文档片段生成摘要，需要考虑用户查询的上下文：

用户查询：{query}

文档内容：{chunk_content}

请提炼核心内容，生成不超过{max_length}字的摘要，并以JSON格式返回结果。"""}
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            content = self._clean_json_response(response.content)
            
            try:
                result = ChunkAbstractResult.model_validate_json(content)
                return result
            except Exception as e:
                print(f"Pydantic解析错误: {str(e)}")
                print(f"清理后的响应: {content}")
                return ChunkAbstractResult(
                    abstract="解析错误，无法生成摘要",
                    key_points=["解析失败"],
                    relevance_aspects=["无法确定相关性"],
                    original_length=len(chunk_content),
                    abstract_length=0
                )
                
        except Exception as e:
            print(f"生成摘要时出错: {str(e)}")
            return ChunkAbstractResult(
                abstract=f"处理出错: {str(e)}",
                key_points=["处理失败"],
                relevance_aspects=["无法确定相关性"],
                original_length=len(chunk_content),
                abstract_length=0
            )
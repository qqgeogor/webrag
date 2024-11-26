import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import Graph, StateGraph,END
from langgraph.prebuilt import ToolExecutor
from pydantic import BaseModel, Field
import operator
from enum import Enum
import arxiv
import httpx
import json
import asyncio
from pypdf import PdfReader
import io
import requests
from chromadb import Client, Settings
from datasketch import MinHash, MinHashLSH
import hashlib
import tempfile
import re


from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import networkx as nx
from nltk.tokenize import sent_tokenize
import re

@dataclass
class SemanticChunk:
    content: str
    start_idx: int
    end_idx: int
    coherence_score: float
    sentences: List[str]

class SemanticChunker:
    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        min_chunk_size: int = 100,
        max_chunk_size: int = 1000,
        overlap_size: int = 50
    ):
        self.model = SentenceTransformer(model_name)
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
        text: str
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
                    sentences=current_chunk
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
                sentences=current_chunk
            ))
        
        return chunks

    async def chunk_by_topic_segmentation(
        self,
        text: str
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
                    embeddings[community]
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
                    sentences=community_sentences
                ))
        
        return sorted(chunks, key=lambda x: x.start_idx)

    def _split_large_chunk(
        self,
        sentences: List[str],
        embeddings: np.ndarray
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
                    sentences=current_chunk
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
                sentences=current_chunk
            ))
        
        return chunks


# 设置环境变量来控制线程数
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

class PDFChunk(BaseModel):
    chunk_id: str
    content: str
    chunk_index: int

class PDFContent(BaseModel):
    title: str
    chunks: List[PDFChunk]
    pdf_url: str
    hash_id: str
    is_duplicate: bool = False
    duplicate_hash: Optional[str] = None

class TextChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        # 按段落分割文本
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            para_size = len(para)
            
            if current_size + para_size > self.chunk_size:
                # 如果当前块已满，保存它
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                # 保留最后一部分作为重叠
                overlap_size = 0
                overlap_chunks = []
                for chunk in reversed(current_chunk):
                    if overlap_size + len(chunk) <= self.chunk_overlap:
                        overlap_chunks.insert(0, chunk)
                        overlap_size += len(chunk)
                    else:
                        break
                current_chunk = overlap_chunks
                current_size = overlap_size
            
            current_chunk.append(para)
            current_size += para_size
        
        # 添加最后一个块
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks

class RAGTool:
    def __init__(self):
        # 设置 ChromaDB 的配置
        settings = Settings(
            persist_directory="./chroma_db",
            anonymized_telemetry=False,
            allow_reset=True,
            is_persistent=True
        )
        
        # 创建客户端
        self.chroma_client = Client(settings)
        self.collection = self.chroma_client.get_or_create_collection(
            name="paper_chunks",
            metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
        )
        
        self.lsh = MinHashLSH(threshold=0.8, num_perm=128)
        self.chunker = TextChunker()
        
        # 修改 embedding 模型初始化
        try:
            self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        except:
            # 如果上面的失败，尝试不同的模型名称
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
        # 用于存储BM25索引的属性
        self.bm25_index = None
        self.doc_store = []  # 存储文档原文
        
    def _create_bm25_index(self, documents: List[str]):
        """创建BM25索引"""
        # 对文档进行分词
        tokenized_docs = [doc.split() for doc in documents]
        self.bm25_index = BM25Okapi(tokenized_docs)
        self.doc_store = documents
        
    def _get_embedding(self, text: str) -> np.ndarray:
        """获取文本的embedding向量"""
        return self.embedding_model.encode(text)
    
    async def hybrid_search(
        self, 
        query: str, 
        top_k: int = 5, 
        alpha: float = 0.5
    ) -> List[Dict]:
        """
        混合检索方法
        """
        try:
            # 1. Embedding 检索
            query_embedding = self._get_embedding(query)
            
            # 获取集合中的文档数量
            collection_size = len(self.collection.get()["ids"])
            actual_top_k = min(top_k * 2, collection_size)  # 确保不超过实际文档数
            
            embedding_results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=actual_top_k
            )
            
            # 2. BM25 检索
            if self.bm25_index is None:
                all_docs = self.collection.get()["documents"]
                self._create_bm25_index(all_docs)
            
            bm25_scores = self.bm25_index.get_scores(query.split())
            
            # 3. 合并结果
            combined_results = []
            seen_docs = set()
            
            # 处理embedding结果
            for i, (doc_id, distance) in enumerate(zip(
                embedding_results['ids'][0], 
                embedding_results['distances'][0]
            )):
                if doc_id not in seen_docs:
                    # 获取文档内容和元数据
                    doc_data = self.collection.get(ids=[doc_id])
                    doc_content = doc_data["documents"][0]
                    doc_metadata = doc_data["metadatas"][0]
                    
                    # 获取文档在原始列表中的索引
                    try:
                        doc_index = self.doc_store.index(doc_content)
                        bm25_score = bm25_scores[doc_index]
                    except (ValueError, IndexError):
                        bm25_score = 0
                    
                    # 归一化分数
                    embedding_score = 1 - distance  # 转换距离为相似度
                    normalized_bm25 = bm25_score / max(bm25_scores) if max(bm25_scores) > 0 else 0
                    
                    # 计算混合分数
                    final_score = alpha * embedding_score + (1 - alpha) * normalized_bm25
                    
                    combined_results.append({
                        "id": doc_id,
                        "content": doc_content,
                        "metadata": doc_metadata,
                        "score": final_score,
                        "embedding_score": embedding_score,
                        "bm25_score": normalized_bm25
                    })
                    seen_docs.add(doc_id)
            
            # 按最终分数排序
            combined_results.sort(key=lambda x: x["score"], reverse=True)
            return combined_results[:top_k]
            
        except Exception as e:
            print(f"检索过程中出错: {str(e)}")
            return []
    
    async def process_pdf(self, pdf_url: str, title: str) -> Optional[PDFContent]:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(pdf_url)
                with tempfile.NamedTemporaryFile(suffix='.pdf') as tmp_file:
                    tmp_file.write(response.content)
                    tmp_file.flush()
                    
                    reader = PdfReader(tmp_file.name)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
                    
                    # 分块处理
                    chunks = self.chunker.split_text(text)
                    pdf_chunks = []
                    for i, chunk in enumerate(chunks):
                        chunk_id = hashlib.md5(f"{text}{i}".encode()).hexdigest()
                        pdf_chunks.append(PDFChunk(
                            chunk_id=chunk_id,
                            content=chunk,
                            chunk_index=i
                        ))
                    
            # 创建整个文档的 MinHash
            minhash = MinHash(num_perm=128)
            for d in text.split():
                minhash.update(d.encode('utf-8'))
            
            # 生成文档唯一标识
            hash_id = hashlib.md5(text.encode()).hexdigest()
            
            # 检查重复
            duplicate_docs = self.lsh.query(minhash)
            if not duplicate_docs:
                self.lsh.insert(hash_id, minhash)
                return PDFContent(
                    title=title,
                    chunks=pdf_chunks,
                    pdf_url=pdf_url,
                    hash_id=hash_id,
                    is_duplicate=False
                )
            else:
                return PDFContent(
                    title=title,
                    chunks=pdf_chunks,
                    pdf_url=pdf_url,
                    hash_id=hash_id,
                    is_duplicate=True,
                    duplicate_hash=duplicate_docs[0]
                )
            
        except Exception as e:
            print(f"处理PDF时出错: {str(e)}")
            return None
            
    def store_in_chroma(self, pdf_content: PDFContent):
        """存储文档到ChromaDB，同时更新BM25索引"""
        documents = []
        metadatas = []
        ids = []
        embeddings = []
        
        for chunk in pdf_content.chunks:
            documents.append(chunk.content)
            metadatas.append({
                "title": pdf_content.title,
                "pdf_url": pdf_content.pdf_url,
                "doc_hash_id": pdf_content.hash_id,
                "chunk_index": chunk.chunk_index
            })
            ids.append(chunk.chunk_id)
            embeddings.append(self._get_embedding(chunk.content).tolist())
        
        # 存储到ChromaDB
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings
        )
        
        # 更新BM25索引和文档存储
        all_docs = self.collection.get()["documents"]
        self._create_bm25_index(all_docs)
        self.doc_store = all_docs

# 1. 基础模型定义
class PaperInfo(BaseModel):
    title: str
    abstract: str
    pdf_url: str
    authors: List[str]

# 2. 状态定义
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    current_agent: str
    next_agent: Optional[str]
    final_answer: Optional[str]
    tools_output: Dict[str, Any]

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

# 1. 添加新的Pydantic模型定义
class CalculatorParams(BaseModel):
    a: float
    b: float

class ArxivParams(BaseModel):
    keyword: str
    max_results: int = Field(default=10)

class AgentDecision(BaseModel):
    next_agent: str = Field(
        ...,  # 表示必填字段
        description="下一个要执行的代理名称",
    )
    params: CalculatorParams | ArxivParams

# 5. Agent 定义
class MasterAgent:
    def __init__(self):
        self.llm = LLMConfig.create_llm()
        self.system_prompt = """你是一个主控制代理。分析用户请求并决定使用哪个子程序，如果不存在匹配的子程序，直接返回结果。

当用户需要计算时，返回：
{
    "next_agent": "calculator_agent",
    "params": {
        "a": 数字1,
        "b": 数字2
    }
}

当用户需要搜索论文时，返回：
{
    "next_agent": "arxiv_agent",
    "params": {
        "keyword": "搜索关键词",
        "max_results": 搜索数量
    }
}

注意：next_agent 必须严格是 "calculator_agent" 或 "arxiv_agent" 之一。
"""

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

class CalculatorAgent:
    def __init__(self):
        self.llm = LLMConfig.create_llm()
        self.tool = CalculatorTool()
        
    async def invoke(self, state: AgentState) -> AgentState:
        params = state["tools_output"]["params"]
        result = self.tool.add(params["a"], params["b"])
        state["tools_output"]["result"] = result

        return state

class RelevanceCheckResult(BaseModel):
    is_relevant: bool
    confidence: float
    reason: str

class RelevanceCheckAgent:
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

class GeneratorAgent:
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
            f"文档 {i+1} (置信度: {doc['relevance_check']['confidence']}):\n{doc['content']}"
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

class HallucinationCheckAgent:
    def __init__(self):
        self.llm = LLMConfig.create_llm()
        self.system_prompt = """你是一个专门检测答案质量的助手。你需要判断生成的答案是否真实可靠地回答了用户的问题。

请严格按照以下JSON格式返回结果：
{
    "is_valid": true/false,      # 答案是否有效
    "confidence": 0.0-1.0,       # 判断的置信度
    "issues": ["问题1", "问题2"], # 发现的具体问题
    "suggestions": ["建议1", "建议2"] # 改进建议
}

评估标准：
1. 答案是否直接回答了问题
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

class ChunkAbstractAgent:
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

提炼要求：
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

# 修改 ArxivAgent 以包含 GeneratorAgent
class ArxivAgent:
    def __init__(self):
        self.llm = LLMConfig.create_llm()
        self.tool = ArxivTool()
        self.rag_tool = RAGTool()
        self.relevance_checker = RelevanceCheckAgent()
        self.generator = GeneratorAgent()
        self.hallucination_checker = HallucinationCheckAgent()
        self.top_k = 5
        self.chunk_abstracter = ChunkAbstractAgent()
        
    async def invoke(self, state: AgentState) -> AgentState:
        try:
            params = state["tools_output"]["params"]
            papers = await self.tool.search_papers(
                keyword=params["keyword"],
                max_results=params.get("max_results", 10)
            )
            
            # 处理PDF并存储
            processed_papers = []
            for paper in papers:
                pdf_content = await self.rag_tool.process_pdf(paper.pdf_url, paper.title)
                if pdf_content and not pdf_content.is_duplicate:
                    self.rag_tool.store_in_chroma(pdf_content)
                    processed_papers.append(paper)
            
            # 混合检索
            results = await self.rag_tool.hybrid_search(
                query=params["keyword"],
                top_k=self.top_k,
                alpha=0.7
            )
            
            # 相关性检查
            filtered_results = []
            for result in results:
                relevance_check = await self.relevance_checker.check_relevance(
                    query=params["keyword"],
                    document=result["content"]
                )
                
                if relevance_check.is_relevant and relevance_check.confidence > 0.6:
                    result["relevance_check"] = relevance_check.model_dump()
                    filtered_results.append(result)
            
            # 在相关性检查后，为每个相关文档生成摘要
            filtered_results_with_abstract = []
            for result in filtered_results:
                abstract_result = await self.chunk_abstracter.create_abstract(
                    query=params["keyword"],
                    chunk_content=result["content"],
                    max_length=200
                )
                
                result["abstract"] = abstract_result.model_dump()
                # 使用摘要替换原始内容用于后续处理
                result["original_content"] = result["content"]
                result["content"] = abstract_result.abstract
                filtered_results_with_abstract.append(result)
            
            # 使用摘要生成最终答案
            if filtered_results_with_abstract:
                answer = await self.generator.generate_answer(
                    query=params["keyword"],
                    relevant_docs=filtered_results_with_abstract
                )
                
                # 检查答案质量
                quality_check = await self.hallucination_checker.check_answer(
                    query=params["keyword"],
                    answer=answer,
                    relevant_docs=filtered_results_with_abstract
                )
                
                if not quality_check.is_valid or quality_check.confidence < 0.7:
                    # 如果答案质量不够好，使用原始内容重新生成
                    print("使用原始内容重新生成答案...")
                    for result in filtered_results_with_abstract:
                        result["content"] = result["original_content"]
                    
                    answer = await self.generator.generate_answer(
                        query=params["keyword"],
                        relevant_docs=filtered_results_with_abstract,
                        max_length=1500
                    )
                    quality_check = await self.hallucination_checker.check_answer(
                        query=params["keyword"],
                        answer=answer,
                        relevant_docs=filtered_results_with_abstract
                    )
            
            # 生成答案
            if filtered_results_with_abstract:
                answer = await self.generator.generate_answer(
                    query=params["keyword"],
                    relevant_docs=filtered_results_with_abstract
                )
                
                # 检查答案质量
                quality_check = await self.hallucination_checker.check_answer(
                    query=params["keyword"],
                    answer=answer,
                    relevant_docs=filtered_results_with_abstract
                )
                
                if not quality_check.is_valid or quality_check.confidence < 0.7:
                    # 如果答案质量不够好，尝试重新生成
                    print("首次生成的答案质量不足，尝试重新生成...")
                    answer = await self.generator.generate_answer(
                        query=params["keyword"],
                        relevant_docs=filtered_results_with_abstract,
                        max_length=1500  # 增加长度限制以获得更详细的答案
                    )
                    # 再次检查质量
                    quality_check = await self.hallucination_checker.check_answer(
                        query=params["keyword"],
                        answer=answer,
                        relevant_docs=filtered_results_with_abstract
                    )
            else:
                answer = "未找到相关的文档内容来回答该问题。"
                quality_check = HallucinationCheckResult(
                    is_valid=False,
                    confidence=0.0,
                    issues=["没有找到相关文档"],
                    suggestions=["尝试使用不同的关键词搜索"]
                )
            
            state["tools_output"]["result"] = {
                "original_count": len(results),
                "filtered_count": len(filtered_results_with_abstract),
                "filtered_results": filtered_results_with_abstract,
                "generated_answer": answer,
                "quality_check": quality_check.model_dump()
            }
            return state
            
        except Exception as e:
            print(f"ArxivAgent处理出错: {str(e)}")
            state["tools_output"]["result"] = {
                "original_count": 0,
                "filtered_count": 0,
                "filtered_results": [],
                "generated_answer": f"处理过程中发生错误: {str(e)}",
                "quality_check": None,
                "error": str(e)
            }
            return state

class ResultFormatter:
    def __init__(self):
        self.llm = LLMConfig.create_llm()
        self.system_prompt = """你是一个结果格式化助手。请将检索和生成结果转换为用户友好的格式。
        
格式要求：
1. 首先展示生成的答案
2. 如果有质量问题，说明具体问题和建议
3. 然后简要说明检索和过滤的统计信息
4. 保持简洁清晰"""
        
    async def invoke(self, state: AgentState) -> AgentState:
        result = state["tools_output"]["result"]
        quality_check = result.get("quality_check", {})
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"""
生成的答案：
{result.get('generated_answer', '未生成答案')}

答案质量检查：
- 有效性: {quality_check.get('is_valid', False)}
- 置信度: {quality_check.get('confidence', 0.0)}
- 发现的问题: {', '.join(quality_check.get('issues', []))}
- 改进建议: {', '.join(quality_check.get('suggestions', []))}

检索统计：
- 原始检索结果数量：{result['original_count']}
- 相关性过滤后数量：{result['filtered_count']}

请将以上信息转换为用户友好的格式。"""}
        ]
        
        response = await self.llm.ainvoke(messages)
        state["final_answer"] = response.content
        return state

# 6. 工作流定义
def create_workflow() -> Graph:
    workflow = StateGraph(AgentState)
    
    # 初始化代理
    master = MasterAgent()
    calculator = CalculatorAgent()
    arxiv = ArxivAgent()
    formatter = ResultFormatter()
    
    # 添加节点
    workflow.add_node("master", master.invoke)
    
    
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
    
    workflow.set_entry_point("master")
    
    return workflow.compile()

# 7. 主函数
async def main():
    workflow = create_workflow()
    
    queries = [
        # "请帮我计算 123 加 456",
        "请帮我搜索关于 transformer architecture 的最新论文",
        # '慈禧太后是什么人',
    ]
    
    for query in queries:
        print(f"\n用户: {query}")
        state = AgentState(
            messages=[HumanMessage(content=query)],
            current_agent="master",
            next_agent=None,
            final_answer=None,
            tools_output={}
        )
        
        try:
            result = await workflow.ainvoke(state)
            print(f"助手: {result['final_answer']}")
        except Exception as e:
            print(f"错误: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
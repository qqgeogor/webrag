from typing import List, Optional
import re
import jieba
from dataclasses import dataclass
from pydantic import BaseModel
import networkx as nx
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer
import numpy as np

@dataclass
class TextChunk:
    content: str
    index: int
    total_chunks: int

class TextChunker:
    def __init__(
        self,
        max_chunk_size: int = 2000,
        min_chunk_size: int = 100,
        overlap_size: int = 50
    ):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap_size = overlap_size
        
    @staticmethod
    def is_chinese(text: str) -> bool:
        """判断文本是否主要为中文"""
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        return chinese_chars / len(text) > 0.5 if text else False
    
    def split_sentences(self, text: str) -> List[str]:
        """智能分句，支持中英文"""
        if self.is_chinese(text):
            # 中文分句
            text = re.sub(r'\n+', '\n', text)  # 统一换行符
            sentences = re.split(r'([。！？\n])', text)
            # 将分句符号加回到句子末尾
            sentences = [''.join(i) for i in zip(sentences[0::2], sentences[1::2] + [''])]
        else:
            # 英文分句
            text = re.sub(r'\s+', ' ', text)  # 统一空格
            sentences = re.split(r'([.!?\n])\s+', text)
            # 将分句符号加回到句子末尾
            sentences = [''.join(i) for i in zip(sentences[0::2], sentences[1::2] + [''])]
            
        return [s.strip() for s in sentences if s.strip()]
    
    def create_chunks(self, text: str) -> List[TextChunk]:
        """将文本分块，保持句子完整性"""
        if not text:
            return []
            
        sentences = self.split_sentences(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # 如果单个句子超过最大长度，需要进行分词再分块
            if sentence_length > self.max_chunk_size:
                # 处理当前累积的内容
                if current_chunk:
                    chunks.append(''.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # 对超长句子进行分词处理
                if self.is_chinese(sentence):
                    words = list(jieba.cut(sentence))
                else:
                    words = sentence.split()
                
                temp_chunk = []
                temp_length = 0
                
                for word in words:
                    word_length = len(word)
                    if temp_length + word_length > self.max_chunk_size:
                        if temp_chunk:
                            chunks.append(''.join(temp_chunk))
                        temp_chunk = [word]
                        temp_length = word_length
                    else:
                        temp_chunk.append(word)
                        temp_length += word_length
                
                if temp_chunk:
                    chunks.append(''.join(temp_chunk))
                continue
            
            # 检查是否需要开始新的块
            if current_length + sentence_length > self.max_chunk_size:
                if current_chunk:
                    chunks.append(''.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # 处理最后剩余的内容
        if current_chunk:
            chunks.append(''.join(current_chunk))
        
        # 添加重叠部分
        final_chunks = []
        total_chunks = len(chunks)
        
        for i, chunk in enumerate(chunks):
            # 添加前一块的结尾
            prefix = ""
            if i > 0:
                prev_chunk = chunks[i-1]
                prefix = prev_chunk[-self.overlap_size:] if len(prev_chunk) > self.overlap_size else prev_chunk
            
            # 添加后一块的开头
            suffix = ""
            if i < total_chunks - 1:
                next_chunk = chunks[i+1]
                suffix = next_chunk[:self.overlap_size] if len(next_chunk) > self.overlap_size else next_chunk
            
            final_content = f"{prefix}{chunk}{suffix}".strip()
            final_chunks.append(
                TextChunk(
                    content=final_content,
                    index=i + 1,
                    total_chunks=total_chunks
                )
            )
        
        return final_chunks

    async def chunk_and_process(
        self,
        text: str,
        processor: callable,
        **kwargs
    ) -> List[str]:
        """分块处理文本"""
        chunks = self.create_chunks(text)
        results = []
        
        for chunk in chunks:
            # 添加块的上下文信息
            context = f"[这是第 {chunk.index}/{chunk.total_chunks} 块文本]\n\n{chunk.content}"
            result = await processor(context, **kwargs)
            results.append(result)
            
        return results 
    


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
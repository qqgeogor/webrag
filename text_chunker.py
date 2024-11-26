from typing import List, Optional
import re
import jieba
from dataclasses import dataclass

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
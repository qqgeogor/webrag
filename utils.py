# 添加新的imports
from bs4 import BeautifulSoup
import trafilatura
import re
from langchain_openai import ChatOpenAI

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
    

class WebContentExtractor:
    """网页内容提取工具"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text)
        # 移除空行
        text = re.sub(r'\n\s*\n', '\n', text)
        return text.strip()
    
    @staticmethod
    async def extract_content(html: str) -> str:
        """
        从HTML中提取主要内容
        优先使用trafilatura，如果失败则回退到BeautifulSoup
        """
        # 首先尝试使用trafilatura
        content = trafilatura.extract(html)
        
        if content:
            return WebContentExtractor.clean_text(content)
            
        # 如果trafilatura失败，使用BeautifulSoup作为后备方案
        soup = BeautifulSoup(html, 'html.parser')
        
        # 移除script和style标签
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
            
        # 获取文本
        text = soup.get_text()
        return WebContentExtractor.clean_text(text)

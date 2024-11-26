# 添加新的imports
from bs4 import BeautifulSoup
import trafilatura
import re


    

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

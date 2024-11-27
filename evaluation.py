from ragas import evaluate
from ragas.metrics import (
    Faithfulness,  # 改用类而不是函数
    AnswerRelevancy,
    ContextRecall,
    ContextPrecision
)
from langchain.chat_models import ChatOpenAI
from datasets import Dataset
import pandas as pd
from typing import List, Dict
from dataclasses import dataclass
import os

from datasets import Dataset
import pandas as pd
from typing import List, Dict
from dataclasses import dataclass
import os
from utils import LLMConfig

@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: List[str]
    context_urls: List[str]
    scores: Dict[str, float]

class RagasEvaluator:
    def __init__(self, api_base: str = None, api_key: str = None):
        # 配置OpenAI客户端
        llm = LLMConfig.create_llm()
        
        # 使用当前支持的metrics
        self.metrics = [
            Faithfulness(llm=llm),
            AnswerRelevancy(llm=llm),
            ContextRecall(llm=llm),
            ContextPrecision(llm=llm)
        ]
    
    async def evaluate_single(self, 
                            question: str, 
                            answer: str, 
                            contexts: List[str],
                            context_urls: List[str]) -> EvalResult:
        """评估单个问答对"""
        try:
            dataset = Dataset.from_dict({
                "question": [question],
                "answer": [answer],
                "contexts": [contexts],
                "context_urls": [context_urls]
            })
            
            # 运行评估
            scores = evaluate(
                dataset=dataset,
                metrics=self.metrics
            )
            
            # 转换评分名称为中文
            chinese_names = {
                "answer_relevancy": "回答相关性",
                "context_precision": "上下文精确度",
                "faithfulness": "忠实度",
                "context_recall": "上下文召回"
            }
            
            scores_dict = {
                chinese_names.get(k, k): float(v) 
                for k, v in scores.items()
            }
            
            return EvalResult(
                question=question,
                answer=answer,
                contexts=contexts,
                context_urls=context_urls,
                scores=scores_dict
            )
            
        except Exception as e:
            print(f"评估过程中出错: {str(e)}")
            return EvalResult(
                question=question,
                answer=answer,
                contexts=contexts,
                context_urls=context_urls,
                scores={"错误": str(e)}
            )

    def format_results(self, results: List[EvalResult]) -> pd.DataFrame:
        """将评估结果格式化为DataFrame"""
        data = []
        for result in results:
            row = {
                "问题": result.question,
                "回答": result.answer[:100] + "...",  # 截断显示
                "上下文数量": len(result.contexts),
                "来源URL数量": len(result.context_urls)
            }
            row.update({f"评分_{k}": v for k, v in result.scores.items()})
            data.append(row)
        
        return pd.DataFrame(data)

from ragas import evaluate
from ragas.metrics import (
    Faithfulness,  # 改用类而不是函数
    AnswerRelevancy,
    ContextRecall,
    ContextPrecision
)
from ragas.llms import LangchainLLMWrapper


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
    reference: List[str]
    scores: Dict[str, float]

class RagasEvaluator:
    def __init__(self, api_base: str = None, api_key: str = None):
        # 配置OpenAI客户端
        llm = LLMConfig.create_llm()
        llm_wrapper = LangchainLLMWrapper(llm)

        # 使用当前支持的metrics
        self.metrics = [
            Faithfulness(llm=llm_wrapper),
            AnswerRelevancy(llm=llm_wrapper),
            # ContextRecall(llm=llm),
            # ContextPrecision(llm=llm)
        ]
    
    async def evaluate_single(self, 
                            question: str, 
                            answer: str, 
                            contexts: List[str],
                            reference: List[str]) -> EvalResult:
        """评估单个问答对"""
        try:
            dataset = Dataset.from_dict({
                "question": [question],
                "answer": [answer],
                "contexts": [contexts],
                "reference": [reference]
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
                reference=reference,
                scores=scores_dict
            )
            
        except Exception as e:
            print(f"评估过程中出错: {str(e)}")
            return EvalResult(
                question=question,
                answer=answer,
                contexts=contexts,
                reference=reference,
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
                "来源reference数量": len(result.reference)
            }
            row.update({f"评分_{k}": v for k, v in result.scores.items()})
            data.append(row)
        
        return pd.DataFrame(data)

    def format_for_gradio(self, result: EvalResult) -> Dict:
        """将单个评估结果格式化为Gradio友好的展示格式"""
        # 格式化上下文和reference
        context_display = "\n\n".join([
            f"上下文 {i+1}:\n{ctx}" 
            for i, ctx in enumerate(result.contexts)
        ])
        
        reference_display = "\n".join([
            f"- {reference}" 
            for reference in result.reference
        ])
        
        # 格式化评分
        scores_display = "\n".join([
            f"- {metric}: {score:.3f}" 
            for metric, score in result.scores.items()
        ])
        
        return {
            "问题": result.question,
            "回答": result.answer,
            "评分详情": scores_display,
            "参考上下文": context_display,
            "来源reference": reference_display
        }

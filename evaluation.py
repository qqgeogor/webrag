from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextRecall,
    ContextPrecision,
)
from ragas.llms import LangchainLLMWrapper
import asyncio

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
from pydantic import BaseModel


class EvalResult(BaseModel):
    question: str
    answer: str
    contexts: List[str]
    scores: Dict[str, float]
    interpretations: Dict[str, str]

class RagasEvaluator:
    def __init__(self, api_base: str = None, api_key: str = None):
        llm = LLMConfig.create_llm()
        llm_wrapper = LangchainLLMWrapper(langchain_llm=llm)
        
        # Initialize all metrics
        faithfulness = Faithfulness(llm=llm_wrapper)
        answer_relevancy = AnswerRelevancy(llm=llm_wrapper)
        context_recall = ContextRecall(llm=llm_wrapper)
        context_precision = ContextPrecision(llm=llm_wrapper)
        
        # Use all available metrics
        self.metrics = [
            faithfulness,
            # answer_relevancy,
            # context_relevancy,
            # context_recall,
            # context_precision
        ]
    
    async def evaluate_single(self, 
                            question: str, 
                            answer: str, 
                            contexts: List[str],
                            ) -> EvalResult:
        """Evaluate a single QA pair with multiple metrics"""
        try:
            # 创建数据集
            dataset = Dataset.from_dict({
                "question": [question],
                "answer": [answer],
                "contexts": [contexts],
            })
            
            # 在同步上下文中运行评估
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: evaluate(
                    dataset=dataset,
                    metrics=self.metrics
                )
            )
            
            # 转换结果
            result_df = result.to_pandas()
            result_dict = result_df.iloc[0].to_dict()
            
            scores_dict = {}
            scores_dict['faithfulness'] = result_dict['faithfulness']

            interpretations = self._interpret_scores(scores_dict)
            
            return EvalResult(
                question=question,
                answer=answer,
                contexts=contexts,
                scores=scores_dict,
                interpretations=interpretations,
            )
            
        except Exception as e:
            print(f"Evaluation error: {str(e)}")
            return EvalResult(
                question=question,
                answer=answer,
                contexts=contexts,
                scores={"error": str(e)},
                interpretations={},
            )

    def _interpret_scores(self, scores: Dict[str, float]) -> Dict[str, str]:
        """Interpret the meaning of each score"""
        interpretations = {}
        
        for metric, score in scores.items():
            if metric == "faithfulness":
                interpretations[metric] = (
                    "High: Answer is well-supported by context" if score > 0.7
                    else "Low: Answer may contain hallucinations"
                )
            elif metric == "answer_relevancy":
                interpretations[metric] = (
                    "High: Answer is relevant to question" if score > 0.7
                    else "Low: Answer may be off-topic"
                )
            elif metric == "context_relevancy":
                interpretations[metric] = (
                    "High: Retrieved contexts are relevant" if score > 0.7
                    else "Low: Contexts may be off-topic"
                )
            # Add other metric interpretations...
            
        return interpretations

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

if __name__ == "__main__":
    # 使用环境变量
    os.environ.setdefault('OPENAI_API_KEY', 'sk-9693411e1fcb4176ab62ed97f98c68f3')
    os.environ.setdefault('OPENAI_API_BASE', 'https://api.deepseek.com')

    evaluator = RagasEvaluator()
    
    # 准备测试数据
    test_data = {
        "question": "什么是机器学习？",
        "answer": "机器学习是人工智能的一个分支，它使用数据和算法来模仿人类的学习方式，逐步提高准确性。",
        "contexts": [
            "机器学习是人工智能的一个重要分支，通过算法使计算机系统能够从数据中学习和改进。",
            "机器学习使用统计学方法，通过数据训练来提高性能和做出预测。"
        ],
        
    }
    
    result = asyncio.run(evaluator.evaluate_single(**test_data))  
    print(result)
    


    
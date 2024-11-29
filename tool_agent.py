import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['OPENAI_API_KEY'] = 'sk-9693411e1fcb4176ab62ed97f98c68f3'
os.environ['OPENAI_API_BASE'] = 'https://api.deepseek.com'

from abc import ABC, abstractmethod
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict

import asyncio
from langchain_core.messages import HumanMessage
from langgraph.graph import Graph, StateGraph,END
import gradio as gr


from agent import WebSearchAgent, CalculatorAgent, ArxivAgent, MasterAgent,AgentState,ResultFormatter
from evaluation import RagasEvaluator  # 假设源代码在 src 目录下



# 6. 工作流定义
def create_workflow() -> Graph:
    workflow = StateGraph(AgentState)
    
    # 初始化代理
    master = MasterAgent()
    calculator = CalculatorAgent()
    websearch = WebSearchAgent()
    arxiv = ArxivAgent()
    formatter = ResultFormatter()
    
    # 添加节点
    workflow.add_node("master", master.invoke)
    

    
    workflow.add_node("websearch_agent", websearch.invoke)
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
            'websearch_agent': "websearch_agent",
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

    workflow.add_edge(
        "websearch_agent",
        END
    )
    
    
    workflow.set_entry_point("master")
    
    return workflow.compile()

# 修改主函数为异步处理单个查询
async def process_query(query: str) -> str:
    workflow = create_workflow()
    state = AgentState(
        messages=[HumanMessage(content=query)],
        current_agent="master",
        next_agent=None,
        final_answer=None,
        tools_output={}
    )
    try:
        result = await workflow.ainvoke(state)
        return result['final_answer']
    except Exception as e:
        return f"发生错误: {str(e)}"

# 添加 Gradio 接口处理函数
def gradio_interface(query: str) -> str:
    return asyncio.run(process_query(query))

# 主函数改为启动 Gradio
def main():
    async def process_query_with_details(query: str) -> tuple:
        workflow = create_workflow()
        state = AgentState(
            messages=[HumanMessage(content=query)],
            current_agent="master",
            next_agent=None,
            final_answer=None,
            tools_output={}
        )
        
        try:
            result = await workflow.ainvoke(state)
            
            # 提取搜索结果
            tools_output = result.get('tools_output', {}).get('result', {})
            filtered_results = tools_output.get('filtered_results', [])
            
            # 格式化搜索结果为表格数据
            table_data = [
                [
                    result.get('title', 'N/A'),
                    result.get('content', '')[:200] + '...',  # 内容摘要
                    f"{result.get('relevance_check', {}).get('confidence', 0):.2f}",
                    result.get('url', 'N/A')
                ]
                for result in filtered_results
            ]
            
            # 生成统计信息
            stats = f"""
### 搜索统计
- 原始结果数: {tools_output.get('original_count', 0)}
- 筛选后结果数: {tools_output.get('filtered_count', 0)}
- 结果质量评分: {tools_output.get('quality_check', {}).get('score', 'N/A')}
            """
            
            # 提取上下文内容
            contexts = []
            try:
                for fr in filtered_results:
                    if isinstance(fr.get('content'), str):
                        contexts.append(fr['content'])
                    elif isinstance(fr.get('content'), dict):
                        contexts.append(str(fr['content']))  # 转换字典为字符串
                    else:
                        contexts.append('')  # 如果无法获取内容，添加空字符串
            except Exception as e:
                contexts = ['']  # 确保至少有一个空上下文
            
            # # 评估结果
            # try:
            #     evaluator = RagasEvaluator()
            #     eval_result = await evaluator.evaluate_single(
            #         question=query,
            #         answer=result['final_answer'],
            #         contexts=contexts,
            #         reference='N/A',
            #     )
                
            #     formatted_eval_result = evaluator.format_for_gradio(eval_result)
            # except Exception as e:
            #     formatted_eval_result = {
            #         "error": f"评估过程出错: {str(e)}",
            #         "scores": {},
            #         "contexts": contexts,
            #         "reference": 'N/A'
            #     }

            return (
                result['final_answer'],  # 主要回答
                table_data,              # 搜索结果表格
                stats,                   # 统计信息
                # formatted_eval_result    # 评估结果
            )
            
        except Exception as e:
            return f"发生错误: {str(e)}", [], "处理出错"

    def gradio_interface(query: str) -> tuple:
        return asyncio.run(process_query_with_details(query))

    demo = gr.Blocks()
    
    with demo:
        gr.Markdown("# AI 助手")
        gr.Markdown("我可以帮您搜索信息、进行计算等。请输入您的问题。")
        
        with gr.Row():
            with gr.Column():
                # 输入框
                query_input = gr.Textbox(
                    lines=2, 
                    placeholder="请输入您的问题...",
                    label="问题"
                )
                
                # 提交按钮
                submit_btn = gr.Button("提交")
                
            with gr.Column():
                # 主要回答
                answer_output = gr.Textbox(
                    lines=4,
                    label="AI 回答"
                )
                
                # 搜索结果展示
                with gr.Accordion("参考资料", open=False):
                    results_output = gr.Dataframe(
                        headers=["标题", "内容摘要", "相关度", "来源链接"],
                        label="搜索结果",
                        wrap=True
                    )
                    
                    stats_output = gr.Markdown(
                        label="统计信息"
                    )
                    
                # # 添加评估结果展示
                # with gr.Accordion("评估结果", open=False):
                #     eval_output = gr.JSON(label="质量评估")
                
        # 更新提交事件的输出
        submit_btn.click(
            fn=gradio_interface,
            inputs=[query_input],
            outputs=[
                answer_output, 
                results_output, 
                stats_output, 
                # eval_output
            ]
        )
        
        # 更新示例输出
        gr.Examples(
            examples=[
                ["请帮我计算 123 加 456"],
                ["慈禧太后是什么人，罗列具体的重大事件"],
                ["请帮我搜索关于 transformer architecture 的最新论文，并提供摘要"],
                ['怎么样才能做出好吃的蛋炒饭'],
                ['我需要自己做河豚鱼刺身，具体应该怎么做，有什么步骤和重点注意'],
                ['学习python要准备一些什么工作'],
            ],
            inputs=query_input,
            outputs=[
                answer_output, 
                results_output, 
                stats_output, 
                # eval_output
            ]
        )
        
        # 修改 launch() 调用，移除 examples 参数
        demo.launch(share=False)  # 直接调用 launch()，不需要任何示例参数

if __name__ == "__main__":
    main()
    
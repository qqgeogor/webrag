import dspy
import os
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
# Configure DSPy with DeepSeek
api_key = os.getenv("DEEPSEEK_API_KEY")


# # Create a custom DeepSeek predictor for DSPy
# class DeepSeekPredictor(dspy.Predictor):
#     def __init__(self, api_key):
#         self.model = ChatOpenAI(
#             model='deepseek-chat', 
#             openai_api_key='sk-33a74c7a4aa94c7eb28a28a98852d0ac', 
#             openai_api_base='https://api.deepseek.com/v1',
#             temperature=0.7,
#         )
    
#     def __call__(self, prompt, **kwargs):
#         try:
#             messages = [
#                 HumanMessage(content=prompt)
#             ]
            
#             response = self.model.invoke(messages).content
                
#             return response.content
#         except Exception as e:
#             print(f"Error calling DeepSeek API: {e}")
#             return ""

# # Configure DSPy to use DeepSeek
# predictor = DeepSeekPredictor(api_key)
# dspy.configure(predictor=predictor)

lm = dspy.LM('deepseek-chat', api_key='sk-33a74c7a4aa94c7eb28a28a98852d0ac',api_base='https://api.deepseek.com/v1')
dspy.configure(lm=lm)


class ZeroShotAnswer(dspy.Signature):
    problem: str = dspy.InputField()
    answer: str = dspy.OutputField()


class CritiqueAnswer(dspy.Signature):
    problem: str = dspy.InputField()
    current_answer: str = dspy.InputField()
    critique: str = dspy.OutputField()


class RefineAnswer(dspy.Signature):
    """[[ ## proposed_instruction ## ]] Given a mathematical problem, a current answer, and a critique of that answer,
    refine the current answer to provide a more accurate and well-reasoned solution. Begin by carefully analyzing the
    problem and the critique, then think step by step to derive the correct answer. Ensure that your reasoning is clear
    and logical, and that the final answer is justified by the steps taken.

    [[ ## completed ## ]]
    """

    problem: str = dspy.InputField()
    current_answer: str = dspy.InputField()
    critique: str = dspy.InputField()
    answer: str = dspy.OutputField()


class ZeroShotCoT(dspy.Module):
    def __init__(self):
        self.cot = dspy.TypedChainOfThought(ZeroShotAnswer)

    def forward(self, problem) -> dspy.Prediction:
        return dspy.Prediction(answer=self.cot(problem=problem).answer)
    

# Define output types for structured responses
class MathSolution(dspy.Signature):
    reasoning: str = dspy.OutputField()
    answer: str = dspy.OutputField()

# Create a basic zero-shot chain of thought solver
class MathProblemSolver(dspy.Module):
    def __init__(self, num_turns: int = 1):
        super().__init__()
        self.zero_shot = dspy.TypedChainOfThought(ZeroShotAnswer)
        self.critique = dspy.TypedChainOfThought(CritiqueAnswer)
        self.refine = dspy.TypedChainOfThought(RefineAnswer)
        self.num_turns = num_turns
    
    def forward(self, question):
        # Get initial answer using zero-shot
        current_answer = self.zero_shot(problem=question).answer
        reasoning = []
        
        # Refine the answer multiple times
        for i in range(self.num_turns):
            print(f"Refinement {i+1}: {current_answer}")
            # Get critique of current answer
            critique_result = self.critique(
                problem=question,
                current_answer=current_answer
            )
            reasoning.append(f"Critique {i+1}: {critique_result.critique}")
            
            # Refine based on critique
            refined_result = self.refine(
                problem=question,
                current_answer=current_answer,
                critique=critique_result.critique
            )
            
            current_answer = refined_result.answer
            reasoning.append(f"Refinement {i+1}: {current_answer}")
        
        return dspy.Prediction(
            reasoning="\n".join(reasoning),
            answer=current_answer
        )

def main():
    # Initialize the solver with 2 refinement turns
    solver = MathProblemSolver(num_turns=2)
    
    # Test problems
    problems = [
        """Why Langevin dynamic has similar form to gradient ascent?
    How to connect Langevin dynamic with gradient ascent and explain how they are similar?
    How can I use Langevin dynamic to explain why edm diffusion works so good?"""
    ]
    
    # Solve each problem
    for i, problem in enumerate(problems, 1):
        print(f"\nProblem {i}:")
        print(f"Question: {problem}")
        
        solution = solver(problem)
        
        print("\nReasoning:")
        print(solution.reasoning)
        print("\nFinal Answer:")
        print(solution.answer)
        print("-" * 50)

if __name__ == "__main__":
    main()

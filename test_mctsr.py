import dspy
from mcts_llm.mctsr import MCTSr


problem = """Why Langevin dynamic has similar form to gradient ascent?
    How to connect Langevin dynamic with gradient ascent and explain how they are similar?
    How can I use Langevin dynamic to explain why edm diffusion works so good?"""


lm = dspy.LM('deepseek-chat', api_key='sk-33a74c7a4aa94c7eb28a28a98852d0ac',api_base='https://api.deepseek.com/v1')
dspy.configure(lm=lm)

mctsr = MCTSr()
mctsr_answer = mctsr(problem).answer
print(f"MCStr answer: {mctsr_answer}")
from typing import Dict, List, Optional, Tuple
import json
from dataclasses import dataclass
import random
from copy import deepcopy
import math
from langchain_openai import ChatOpenAI
import re
from tqdm import tqdm
from difflib import SequenceMatcher

# Initialize LLM models with different temperatures
solver_model = ChatOpenAI(
    model='deepseek-chat', 
    openai_api_key='sk-9693411e1fcb4176ab62ed97f98c68f3', 
    openai_api_base='https://api.deepseek.com/v1',
    temperature=0.7,
)

evaluator_model = ChatOpenAI(
    model='deepseek-chat', 
    openai_api_key='sk-9693411e1fcb4176ab62ed97f98c68f3', 
    openai_api_base='https://api.deepseek.com/v1',
    temperature=0.7,
)

@dataclass
class MathState:
    problem: str
    steps: List[str]
    current_step: str
    is_solved: bool = False
    
    def to_dict(self) -> Dict:
        return {
            "problem": self.problem,
            "steps": self.steps,
            "current_step": self.current_step,
            "is_solved": self.is_solved
        }

class MCTSNode:
    def __init__(self, state: MathState, parent: Optional['MCTSNode'] = None):
        self.state = state
        self.parent = parent
        self.children: List['MCTSNode'] = []
        self.visits = 0
        self.value = 0.0
        self.depth = 0 if parent is None else parent.depth + 1  # Add depth tracking
        
    def is_terminal(self) -> bool:
        """Check if this node represents a solution"""
        return self.state.is_solved
    
    def is_fully_expanded(self) -> bool:
        """Check if all possible actions have been tried"""
        return len(self.children) >= 3  # Limit branching factor
    
    def get_uct_value(self, exploration_weight: float) -> float:
        """Calculate UCT value for node selection"""
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.value / self.visits
        exploration = exploration_weight * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

    def visualize(self) -> str:
        """Generate a text representation of this node and its children"""
        indent = "  " * self.depth
        stats = f"[visits={self.visits}, value={self.value:.2f}]"
        
        # Show abbreviated state info
        current_step = self.state.current_step[:50]+'...' if len(self.state.current_step) > 50 else self.state.current_step
        node_info = f"{indent}├─ {current_step} {stats}"
        
        # Recursively visualize children
        child_info = []
        for child in self.children:
            child_info.append(child.visualize())
            
        return "\n".join([node_info] + child_info)

class MathMCTS:
    def __init__(self, exploration_weight: float = 1.414):
        self.exploration_weight = exploration_weight
    
    def select_node(self, node: MCTSNode) -> MCTSNode:
        """Select a promising node to explore"""
        current = node
        while not current.is_terminal() and current.is_fully_expanded():
            current = max(current.children, 
                         key=lambda n: n.get_uct_value(self.exploration_weight))
        return current
    
    def expand_node(self, node: MCTSNode) -> MCTSNode:
        # Create prompt for next step

        previous_steps = []
        while node.parent:
            previous_steps.append(node.parent.state.current_step)
            node = node.parent
        previous_steps.reverse()

        # Modify prompt to encourage diverse steps
        existing_steps = [child.state.current_step for child in node.children]
        prompt = f"""Given this math problem and current progress, suggest ONE next step in the solution.
Problem: {node.state.problem}
Previous steps: {' ->=========== '.join(previous_steps)}
Current expression: {node.state.current_step}

Already tried steps at this level: {', '.join(existing_steps)}
Please provide a different approach than the existing steps.

Provide only the next mathematical step, no explanations."""

        # Get next step from model and ensure it's different from siblings
        max_attempts = 3
        for _ in range(max_attempts):
            response = solver_model.invoke([
                {"role": "system", "content": "You are a problem solver. Provide only the next step."},
                {"role": "user", "content": prompt}
            ])
            
            next_step = response.content.strip()
            
            # Check if this step is sufficiently different from existing steps
            if not any(self._steps_are_similar(next_step, existing) for existing in existing_steps):
                break
        
        # Create new state with updated steps
        new_state = MathState(
            problem=node.state.problem,
            steps=node.state.steps + [next_step],
            current_step=next_step,
            # is_solved=self._check_if_solved(node.state.problem, next_step,previous_steps)
        )
        
        # Create and link new child node
        child = MCTSNode(new_state, parent=node)
        node.children.append(child)
        
        return child
    
    def simulate(self, node: MCTSNode) -> float:
        """Simulate to end and evaluate result"""
        if node.state.is_solved:
            return self._evaluate_solution(node.state)
        
        # Try to solve in one shot
        prompt = f"""Solve this problem in one step if possible:
Problem: {node.state.problem}
Current progress: {' -> '.join(node.state.steps)}
Current expression: {node.state.current_step}

Show only the final answer, no explanation."""
        
        response = solver_model.invoke([
            {"role": "system", "content": "You are a problem solver."},
            {"role": "user", "content": prompt}
        ])
        
        solution = response.content.strip()
        # Check both the solution and the path taken to get there

        return self._evaluate_solution_attempt(node.state.problem, solution,node.state.steps)
    
    def backpropagate(self, node: MCTSNode, value: float):
        """Backpropagate the evaluation value"""
        current = node
        while current is not None:
            current.visits += 1
            current.value += value
            current = current.parent
    
    

    def _check_if_solved(self, problem: str, current_step: str,steps: List[str]) -> bool:
        """Check if the solution is correct and the steps are valid"""

        prompt = f"""Is this solution path correct for the given problem?
Problem: {problem}
Steps: {' -> '.join(steps)}
Current step: {current_step}

Verify that:
1. The solution is correct
2. The solution is mathematically sound
3. The solution is correct
Answer only 'yes' or 'no'."""
      
        
        response = evaluator_model.invoke([
            {"role": "system", "content": "You are a solution validator."},
            {"role": "user", "content": prompt}
        ])
        
        return 'yes' in response.content.lower()
    
    def _evaluate_solution(self, state: MathState) -> float:
        """Evaluate a complete solution"""
        prompt = f"""Rate this solution from 0 to 1:
Problem: {state.problem}
Solution steps: {' -> '.join(state.steps)}

Consider:
1. Correctness
2. Efficiency (fewer steps is better)
3. Clarity

Return only the numeric score."""
        
        response = evaluator_model.invoke([
            {"role": "system", "content": "You are a solution evaluator."},
            {"role": "user", "content": prompt}
        ])
        
        try:
            return float(response.content.strip())
        except:
            return 0.0
    
    def _evaluate_solution_attempt(self, problem: str, attempt: str,steps: List[str]) -> float:
        """Evaluate a solution attempt"""
        if self._check_if_solved(problem, attempt,steps):
            return 1.0
        return 0.5  # Partial credit for reasonable attempts
    
    def search(self, initial_state: MathState, num_iterations: int = 100) -> MathState:
        """Perform MCTS search"""
        root = MCTSNode(initial_state)
        
        for i in tqdm(range(num_iterations)):
            node = self.select_node(root)
            if not node.is_terminal() and not node.is_fully_expanded():
                node = self.expand_node(node)
            value = self.simulate(node)
            self.backpropagate(node, value)
            
            # Visualize tree every 10 iterations
            if (i + 1) % 1 == 0:
                print(f"\nMCTS Tree at iteration {i + 1}:")
                print(root.visualize())
                print("\n" + "="*80 + "\n")
            
            if node.state.is_solved:
                return node.state
        
        # Return best found solution
        best_child = max(root.children, key=lambda n: n.value / n.visits if n.visits > 0 else 0)
        return best_child.state

    def _steps_are_similar(self, step1: str, step2: str) -> bool:
        """Compare two steps to determine if they are too similar"""
        # Remove whitespace and convert to lowercase for comparison
        s1 = ''.join(step1.lower().split())
        s2 = ''.join(step2.lower().split())
        
        # If steps are exactly the same
        if s1 == s2:
            return True
        
        # If one step is contained within the other
        if s1 in s2 or s2 in s1:
            return True
        
        # Calculate similarity ratio
        similarity = SequenceMatcher(None, s1, s2).ratio()
        return similarity > 0.8  # Adjust threshold as needed

def solve_math_problem(problem: str) -> Tuple[MathState, float]:
    """Solve a math problem using MCTS with LLM"""
    # Initialize state
    initial_state = MathState(
        problem=problem,
        steps=[],
        current_step=problem
    )
    
    # Create MCTS solver
    mcts = MathMCTS()
    
    # Search for solution
    solution_state = mcts.search(initial_state)
    
    # Evaluate final solution
    score = mcts._evaluate_solution(solution_state)
    
    return solution_state, score

def main():
    # Complex math problems
    problems = [
        """A cylindrical water tank has a radius of 3 meters and a height of 8 meters. 
        If water is being pumped into the tank at a rate of 12 cubic meters per hour, 
        how long will it take to fill 75% of the tank? Give your answer in hours and minutes.""",
        
        """In a probability experiment, a fair coin is tossed three times, followed by 
        rolling a fair six-sided die if at least two heads appear. What is the probability 
        of getting a number greater than 4 on the die, given that the experiment reaches 
        the die-rolling stage?""",
        
        """Two trains depart from stations A and B, which are 450 kilometers apart. 
        Train A travels at 90 km/h and departs at 9:00 AM. Train B travels at 75 km/h 
        and departs at 8:15 AM heading towards A. Assuming constant speeds:
        1. At what time will they meet?
        2. How far from station A will their meeting point be?"""
    ]

    # problems = [
    #     """
    #     Why Langevin dynamic has similar form to gradient ascent?
    #     How to connect Langevin dynamic with gradient ascent and explain how they are similar?
    #     How can I use Langevin dynamic to explain why edm diffusion works so good?
    #     """,
    # ]
    problems = [
    """A space station is orbiting Earth in an elliptical orbit. At its closest point (perigee), 
    it's 400 km above Earth's surface, and at its farthest point (apogee), it's 900 km above. 
    The station releases a spherical satellite that deploys a solar sail.

    Given:
    - Earth's radius = 6371 km
    - Solar sail area = 100 m²
    - Solar radiation pressure = 9.08 × 10⁻⁶ N/m²
    - Initial satellite mass = 50 kg
    - The satellite is released at perigee
    - Solar sail is oriented at 45° to the sun's rays
    
    Calculate:
    1. The orbital period of the space station (use Kepler's Third Law)
    2. The initial acceleration of the satellite due to the solar sail
    3. Assuming the sail maintains optimal orientation, what's the probability the satellite 
       will escape Earth's orbit within 3 orbits if there's a 15% chance of micrometeoroid 
       damage reducing sail efficiency by 30% during each orbit?
    
    Express final probability as a percentage rounded to 2 decimal places."""
    ]

    for i, problem in enumerate(problems, 1):
        print(f"\n{'='*80}")
        print(f"Problem {i}:")
        print(problem)
        print(f"{'='*80}\n")
        
        solution_state, score = solve_math_problem(problem)
        
        print("\nSolution path:")
        for j, step in enumerate(solution_state.steps, 1):
            print(f"{j}. {step}")
            print('======================')

        print(f"Solution quality score: {score:.2f}")
        
        # Validate solution with a different prompt
        validation_prompt = f"""Verify this solution:
Problem: {problem}
Steps: {' -> '.join(solution_state.steps)}

Explain if the solution is correct and show the correct calculation if there are any errors."""
        
        validation = evaluator_model.invoke([
            {"role": "system", "content": "You are a solution validator."},
            {"role": "user", "content": validation_prompt}
        ])
        
        print("\nValidation:")
        print(validation.content.strip())
        
        input("\nPress Enter to continue to next problem...")

if __name__ == "__main__":
    main()

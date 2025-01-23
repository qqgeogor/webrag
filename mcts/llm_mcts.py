from base import BaseMCTS
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import random
from copy import deepcopy
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import os
import json
from datetime import datetime

@dataclass(frozen=True)  # Make it immutable and hashable
class LLMState:
    original_prompt: str  # The original question to solve
    step_prompts: Tuple[str, ...] = field(default_factory=tuple)  # Use tuple instead of list
    responses: Tuple[str, ...] = field(default_factory=tuple)     # Use tuple instead of list
    depth: int = 0

    def __hash__(self):
        return hash((self.original_prompt, self.step_prompts, self.responses, self.depth))

    def __eq__(self, other):
        if not isinstance(other, LLMState):
            return NotImplemented
        return (self.original_prompt == other.original_prompt and 
                self.step_prompts == other.step_prompts and 
                self.responses == other.responses and 
                self.depth == other.depth)

    def __copy__(self):
        return LLMState(
            original_prompt=self.original_prompt,
            step_prompts=self.step_prompts,
            responses=self.responses,
            depth=self.depth
        )
    
    def __deepcopy__(self, memo):
        return LLMState(
            original_prompt=deepcopy(self.original_prompt, memo),
            step_prompts=tuple(deepcopy(p, memo) for p in self.step_prompts),
            responses=tuple(deepcopy(r, memo) for r in self.responses),
            depth=self.depth
        )

    def get_trajectory(self) -> str:
        """Get a string representation of the current trajectory"""
        if not self.responses:
            return "Empty trajectory"
            
        return "\n".join([
            f"Step {i+1}: {prompt}\nResponse: {response}"
            for i, (prompt, response) in enumerate(zip(self.step_prompts, self.responses))
        ])

class LLMMCTS(BaseMCTS):
    def __init__(self, max_depth: int = 5, temperature: float = 0.7):
        super().__init__()
        self.max_depth = max_depth
        self.model = ChatOpenAI(
            model='deepseek-chat', 
            openai_api_key='sk-33a74c7a4aa94c7eb28a28a98852d0ac', 
            openai_api_base='https://api.deepseek.com/v1',
            temperature=temperature,
        )
        self.system_prompt = SystemMessage(content="""You are a helpful AI assistant focused on breaking down problems into solvable steps.""")

    def get_possible_moves(self, state: LLMState) -> List[str]:
        """Generate possible next step prompts as moves"""
        if state.depth >= self.max_depth:
            return []
        
        # Get the current context from previous steps
        context = "\n".join([
            f"Step {i+1}: {prompt}\nResponse: {response}"
            for i, (prompt, response) in enumerate(zip(state.step_prompts, state.responses))
        ])
        
        # Ask LLM for next possible steps
        step_prompt = f"""Given this question: {state.original_prompt}

Previous steps taken:
{context if context else "No steps taken yet."}

Generate 3 different possible next steps to help solve this question.
Each step should be a specific prompt or instruction that will get us closer to the answer.
Return exactly 3 steps, one per line, without numbering or additional text."""
        
        messages = [
            self.system_prompt,
            HumanMessage(content=step_prompt)
        ]
        
        try:
            response = self.model.invoke(messages).content
            # Add debug print to see raw response
            print("Raw LLM response:", response)
            
            # More robust response parsing
            steps = []
            for line in response.strip().split('\n'):
                line = line.strip()
                # Filter out empty lines and common prefixes
                if line and not line.startswith(('Step', '-', '•', '*', '1.', '2.', '3.')):
                    steps.append(line)
            
            # Ensure we have exactly 3 steps
            while len(steps) < 3:
                steps.append(f"Summarize findings about {state.original_prompt}")
            
            print("Parsed steps:", steps[:3])  # Debug print
            return steps[:3]
        except Exception as e:
            print(f"Error generating steps: {e}")
            return [
                "Analyze the key components of the question",
                "Generate a partial solution based on current understanding",
                "Synthesize previous findings into a coherent response"
            ]

    def apply_move(self, state: LLMState, move: str) -> LLMState:
        """Apply the step prompt and get response from LLM"""
        # Get context from previous steps
        context = "\n".join([
            f"Step {i+1}: {prompt}\nResponse: {response}"
            for i, (prompt, response) in enumerate(zip(state.step_prompts, state.responses))
        ])
        
        # Get response for this step
        exec_prompt = f"""Question: {state.original_prompt}

Previous steps:
{context if context else "No previous steps."}

Current step: {move}
Provide a concise response for this step."""

        messages = [
            self.system_prompt,
            HumanMessage(content=exec_prompt)
        ]
        
        try:
            response = self.model.invoke(messages).content
        except Exception as e:
            print(f"Error executing step: {e}")
            response = "Unable to complete this step"

        # Create new state with updated steps and responses
        new_state = LLMState(
            original_prompt=state.original_prompt,
            step_prompts=state.step_prompts + (move,),
            responses=state.responses + (response,),
            depth=state.depth + 1
        )
        return new_state

    def evaluate_state(self, state: LLMState) -> float:
        """Evaluate how well the steps taken answer the original question"""
        if not state.responses:
            return 0.0

        # Combine all responses into a final answer
        solution = "\n".join([
            f"Step {i+1}: {prompt}\nResponse: {response}"
            for i, (prompt, response) in enumerate(zip(state.step_prompts, state.responses))
        ])
        print('solution', solution)
        
        eval_prompt = f"""Question: {state.original_prompt}

Solution steps and responses:
{solution}

Rate how well this solution answers the original question on a scale of 0.0 to 1.0, considering:
- Completeness of the answer
- Logical progression of steps
- Accuracy of the information
- Clarity of explanation

Return only the numerical score between 0.0 and 1.0."""

        messages = [
            self.system_prompt,
            HumanMessage(content=eval_prompt)
        ]
        
        try:
            score_str = self.model.invoke(messages).content
            score = float(score_str)
            return max(0.0, min(1.0, score))
        except:
            return 0.5

    def save_trajectories_to_file(self, original_prompt: str, filename: Optional[str] = None):
        """Save all trajectories to a JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mcts_trajectories_{timestamp}.json"
            
        trajectory_data = {
            "original_prompt": original_prompt,
            "trajectories": []
        }
        
        for trajectory, value in self.trajectories:
            trajectory_info = {
                "steps": [],
                "value": value
            }
            
            for state, visits, node_value in trajectory:
                if isinstance(state, LLMState):
                    step_info = {
                        "prompt": state.original_prompt,
                        "step_prompts": list(state.step_prompts),
                        "responses": list(state.responses),
                        "depth": state.depth,
                        "visits": visits,
                        "value": node_value
                    }
                    trajectory_info["steps"].append(step_info)
                    
            trajectory_data["trajectories"].append(trajectory_info)
            
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(trajectory_data, f, indent=2, ensure_ascii=False)
            
        return filename

    def get_best_response(self, prompt: str, num_iterations: int = 50) -> str:
        """Get the best solution path and return combined response"""
        initial_state = LLMState(original_prompt=prompt)
        final_state = self.search(initial_state, num_iterations)
        
        # Save trajectories after search
        self.save_trajectories_to_file(prompt)
        
        if not final_state.responses:
            return "No solution found"
            
        solution = "\n\n".join([
            f"Step {i+1}: {prompt}\n{response}"
            for i, (prompt, response) in enumerate(zip(final_state.step_prompts, final_state.responses))
        ])
        
        return solution

    def is_terminal(self, state: LLMState) -> bool:
        """Check if we've reached max depth or no more valid moves"""
        return state.depth >= self.max_depth

    def select_random_move(self, possible_moves: List[str]) -> str:
        """Randomly select a move from possible moves"""
        if not possible_moves:
            raise ValueError("No possible moves available")
        return random.choice(possible_moves)

    def iterative_search(self, prompt: str, num_runs: int = 3, iterations_per_run: int = 50) -> str:
        """
        Perform multiple MCTS searches, using results from previous runs to inform the next run.
        
        Args:
            prompt: The original question to solve
            num_runs: Number of complete MCTS searches to perform
            iterations_per_run: Number of iterations for each MCTS search
        
        Returns:
            str: The best solution found across all runs
        """
        current_prompt = prompt
        all_solutions = []
        
        for run in range(num_runs):
            print(f"\nRun {run + 1}/{num_runs}")
            
            # Reset trajectories for new run
            self.trajectories = []
            
            # Perform MCTS search with updated prompt
            final_state = self.search(LLMState(original_prompt=current_prompt), iterations_per_run)
            
            # Evaluate the final state
            score = self.evaluate_state(final_state)
            
            # Save trajectories with run number in filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mcts_trajectories_run{run+1}_{timestamp}.json"
            self.save_trajectories_to_file(current_prompt, filename)
            
            print(f"Run {run + 1} score: {score}")
            
            # Store this run's solution
            all_solutions.append((final_state, score))
            
            # Update prompt for next iteration using the results from this run
            if run < num_runs - 1:  # Don't update after the last run
                current_prompt = self._create_next_prompt(final_state, score)
        
        # Get the best solution
        # best_solution, best_score = max(all_solutions, key=lambda x: x[1])
        best_solution, best_score = all_solutions[-1]
        # Format the best solution
        solution = "\n\n".join([
            f"Step {i+1}: {step_prompt}\n{response}"
            for i, (step_prompt, response) in enumerate(zip(best_solution.step_prompts, best_solution.responses))
        ])
        
        return f"Best solution (score: {best_score}):\n\n{solution}"

    def _create_next_prompt(self, current_state: LLMState, current_score: float) -> str:
        """
        Create a prompt for the next iteration based on current results.
        """
        # Combine all responses from the current run
        current_solution = "\n".join([
            f"Step {i+1}: {prompt}\nResponse: {response}"
            for i, (prompt, response) in enumerate(zip(current_state.step_prompts, current_state.responses))
        ])
        
        next_prompt = f"""Original question: {current_state.original_prompt}

Previous iteration results (score: {current_score}):
{current_solution}

Based on these results, let's improve the solution further. Focus on:
1. Areas that need more detailed explanation
2. Any gaps in the current reasoning
3. Potential improvements or alternative approaches

Please provide a more refined answer to the original question."""

        return next_prompt

if __name__ == "__main__":
    mcts = LLMMCTS()
    prompt = """
    Why Langevin dynamic has similar form to gradient ascent?
    How to connect Langevin dynamic with gradient ascent and explain how they are similar?
    How can I use Langevin dynamic to explain why edm diffusion works so good?
    """
    
    print("Original Question:", prompt)
    print("\nSearching for best solution across multiple runs...")
    response = mcts.iterative_search(prompt, num_runs=3, iterations_per_run=5)
    print(response)



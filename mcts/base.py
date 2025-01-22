from abc import ABC, abstractmethod
import math
from typing import Any, List, Optional
from tqdm import tqdm
from copy import deepcopy


class Node:
    def __init__(self, state: Any, parent: Optional['Node'] = None):
        self.state = state
        self.parent = parent
        self.children: List['Node'] = []
        self.visits = 0  # Number of times this node has been visited
        self.value = 0.0  # Total value/score accumulated through this node
        self.untried_moves: List[Any] = []  # List of moves not yet expanded
        
    def add_child(self, child_state: Any) -> 'Node':
        """Add a child node with the given state"""
        child = Node(child_state, parent=self)
        self.children.append(child)
        return child
        
    def get_ucb1_score(self, exploration_constant: float) -> float:
        """Calculate the UCB1 score for this node"""
        if self.visits == 0:
            return float('inf')
        exploitation = self.value / self.visits
        exploration = exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration
        
    @property
    def is_fully_expanded(self) -> bool:
        """Check if all possible moves have been tried"""
        return len(self.untried_moves) == 0
        
    @property
    def is_leaf(self) -> bool:
        """Check if this node is a leaf node (has no children)"""
        return len(self.children) == 0
    
    
class BaseMCTS(ABC):
    

    def __init__(self, exploration_constant: float = 1.41):
        self.root: Optional[Node] = None
        self.exploration_constant = exploration_constant
        self.trajectories = []  # Store all trajectories

    def select_node(self, node: Node) -> Node:
        """Select a node using UCB1 formula"""
        while not self.is_terminal(node.state):
            if not node.is_fully_expanded:
                return node
            if node.is_leaf:
                return node
            
            # Use Node's UCB1 calculation method
            node = max(
                node.children,
                key=lambda n: n.get_ucb1_score(self.exploration_constant)
            )
        return node

    def expand_node(self, node: Node) -> Node:
        """Expand the node by adding a child node with an untried move"""
        if not node.untried_moves:
            node.untried_moves = self.get_possible_moves(node.state)
            # Here we compare states to filter out duplicates
            existing_states = {child.state for child in node.children}
            node.untried_moves = [move for move in node.untried_moves 
                                if self.apply_move(node.state, move) not in existing_states]
        
        if not node.untried_moves:
            return node
            
        move = node.untried_moves.pop()
        new_state = self.apply_move(node.state, move)
        return node.add_child(new_state)

    def save_trajectory(self, node: Node, value: float):
        """Save the trajectory from root to this node"""
        trajectory = []
        current = node
        while current is not None:
            trajectory.append((current.state, current.visits, current.value))
            current = current.parent
        trajectory.reverse()
        self.trajectories.append((trajectory, value))

    def simulate(self, node: Node) -> float:
        """Simulate a random playout from the node until terminal state"""
        current_state = deepcopy(node.state)
        trajectory_states = [current_state]  # Track states in this simulation
        
        while not self.is_terminal(current_state):
            possible_moves = self.get_possible_moves(current_state)
            if not possible_moves:
                break
            move = self.select_random_move(possible_moves)
            current_state = self.apply_move(current_state, move)
            trajectory_states.append(current_state)
            
        value = self.evaluate_state(current_state)
        self.save_trajectory(node, value)  # Save the trajectory
        return value

    def backpropagate(self, node: Node, value: float) -> None:
        """Update node statistics going up to the root"""
        while node is not None:
            node.visits += 1
            node.value += value
            node = node.parent

    def search(self, initial_state: Any, num_iterations: int) -> Any:
        """Main MCTS loop"""
        if num_iterations <= 0:
            raise ValueError("Number of iterations must be positive")
            
        self.root = Node(initial_state)
        self.root.untried_moves = self.get_possible_moves(initial_state)
        self.trajectories = []  # Reset trajectories
        
        for _ in tqdm(range(num_iterations)):
            node = self.select_node(self.root)
            if not self.is_terminal(node.state) and not node.is_fully_expanded:
                node = self.expand_node(node)
            value = self.simulate(node)
            self.backpropagate(node, value)

        if not self.root.children:
            raise RuntimeError("No moves were explored")
            
        return max(self.root.children, key=lambda c: c.visits).state

    @abstractmethod
    def get_possible_moves(self, state: Any) -> List[Any]:
        """Return list of possible moves from given state"""
        pass

    @abstractmethod
    def apply_move(self, state: Any, move: Any) -> Any:
        """Apply move to state and return new state"""
        pass

    @abstractmethod
    def is_terminal(self, state: Any) -> bool:
        """Check if state is terminal"""
        pass

    @abstractmethod
    def evaluate_state(self, state: Any) -> float:
        """Return the value of terminal state (e.g., 1 for win, 0 for loss)"""
        pass

    @abstractmethod
    def select_random_move(self, possible_moves: List[Any]) -> Any:
        """Select a random move from possible moves"""
        pass


if __name__ == "__main__":  
    # Example usage would go here, but BaseMCTS cannot be instantiated directly
    pass

"""
Autonomous Response System

Implements autonomous action selection and decision-making based on:
- Expected free energy minimization
- Policy optimization
- Goal-directed behavior
- Epistemic foraging (information-seeking behavior)

No external input needed - the system autonomously decides and acts
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum


class ActionType(Enum):
    """Types of actions the system can take"""
    EXPLORE = "explore"  # Information-seeking
    EXPLOIT = "exploit"  # Goal-directed
    REFINE = "refine"    # Improve current beliefs
    ATTEND = "attend"    # Shift attention
    ADAPT = "adapt"      # Update internal model
    COMMUNICATE = "communicate"  # Express findings


@dataclass
class Action:
    """Represents a possible action"""
    action_type: ActionType
    parameters: Dict[str, Any]
    expected_free_energy: float = float('inf')
    epistemic_value: float = 0.0
    pragmatic_value: float = 0.0
    
    def __lt__(self, other):
        """For sorting by expected free energy (lower is better)"""
        return self.expected_free_energy < other.expected_free_energy


@dataclass
class Policy:
    """Sequence of actions (policy)"""
    actions: List[Action]
    expected_free_energy: float = float('inf')
    horizon: int = 1  # Number of steps ahead
    
    def compute_expected_free_energy(self):
        """Compute total expected free energy for this policy"""
        if not self.actions:
            self.expected_free_energy = float('inf')
        else:
            self.expected_free_energy = sum(a.expected_free_energy for a in self.actions)
        return self.expected_free_energy


class AutonomousResponseSystem:
    """
    Autonomous Response System for self-directed action selection
    
    Key principle: Minimize expected free energy
    - Epistemic value: reduce uncertainty (explore)
    - Pragmatic value: achieve goals (exploit)
    
    The system autonomously:
    1. Generates possible actions
    2. Evaluates expected free energy
    3. Selects optimal action
    4. Executes and monitors results
    """
    
    def __init__(self, horizon: int = 3, n_policies: int = 10):
        """
        Initialize Autonomous Response System
        
        Args:
            horizon: Planning horizon (steps ahead)
            n_policies: Number of policies to evaluate
        """
        self.horizon = horizon
        self.n_policies = n_policies
        
        # Action history
        self.action_history: List[Action] = []
        
        # Current goals
        self.goals: Dict[str, float] = {}
        
        # Action outcomes tracking (for learning)
        self.action_outcomes: Dict[str, List[float]] = {}
        
        # Autonomy parameters
        self.exploration_bonus = 0.5  # Intrinsic motivation for exploration
        self.confidence_threshold = 0.7  # Threshold for exploitation
        
    def generate_action_candidates(self, 
                                   current_state: Dict[str, Any],
                                   uncertainty: float) -> List[Action]:
        """
        Generate candidate actions based on current state
        
        Args:
            current_state: Current system state
            uncertainty: Current belief uncertainty
            
        Returns:
            List of candidate actions
        """
        candidates = []
        
        # Exploration action (epistemic foraging)
        if uncertainty > 0.3:
            candidates.append(Action(
                action_type=ActionType.EXPLORE,
                parameters={'exploration_strategy': 'maximize_information_gain'},
                epistemic_value=uncertainty * self.exploration_bonus
            ))
        
        # Exploitation action (goal pursuit)
        if self.goals:
            for goal_name, goal_value in self.goals.items():
                candidates.append(Action(
                    action_type=ActionType.EXPLOIT,
                    parameters={'goal': goal_name, 'intensity': goal_value},
                    pragmatic_value=goal_value
                ))
        
        # Refine beliefs (improve precision)
        prediction_error = current_state.get('prediction_error', 0.5)
        if prediction_error > 0.3:
            candidates.append(Action(
                action_type=ActionType.REFINE,
                parameters={'target': 'beliefs', 'iterations': 5},
                epistemic_value=prediction_error
            ))
        
        # Attention shift (reallocate attention)
        attention_entropy = current_state.get('attention_entropy', 0.0)
        if attention_entropy > 0.5:
            candidates.append(Action(
                action_type=ActionType.ATTEND,
                parameters={'strategy': 'shift_to_salient'},
                epistemic_value=attention_entropy * 0.5
            ))
        
        # Adapt model (learning)
        learning_opportunity = current_state.get('learning_opportunity', 0.0)
        if learning_opportunity > 0.4:
            candidates.append(Action(
                action_type=ActionType.ADAPT,
                parameters={'adaptation_type': 'model_update'},
                epistemic_value=learning_opportunity * 0.7
            ))
        
        # Communicate (share state)
        confidence = current_state.get('confidence', 0.5)
        if confidence > self.confidence_threshold:
            candidates.append(Action(
                action_type=ActionType.COMMUNICATE,
                parameters={'content': 'current_understanding'},
                pragmatic_value=confidence
            ))
        
        return candidates
    
    def compute_expected_free_energy(self, 
                                    action: Action,
                                    current_state: Dict[str, Any],
                                    predicted_next_state: Dict[str, Any]) -> float:
        """
        Compute expected free energy for an action
        
        Expected Free Energy = Ambiguity - Epistemic Value - Pragmatic Value
        
        Where:
        - Ambiguity: Expected uncertainty after action
        - Epistemic Value: Expected information gain
        - Pragmatic Value: Expected goal achievement
        
        Lower is better (minimize expected free energy)
        
        Args:
            action: Action to evaluate
            current_state: Current state
            predicted_next_state: Predicted state after action
            
        Returns:
            Expected free energy
        """
        # Estimate ambiguity (predicted uncertainty)
        current_uncertainty = current_state.get('uncertainty', 0.5)
        predicted_uncertainty = predicted_next_state.get('uncertainty', 0.5)
        
        # Epistemic value (information gain)
        epistemic = action.epistemic_value
        
        # Pragmatic value (goal achievement)
        pragmatic = action.pragmatic_value
        
        # Expected free energy formula
        # Lower values are better (minimize)
        expected_free_energy = (
            predicted_uncertainty  # Ambiguity
            - epistemic           # Information gain (negative = good)
            - pragmatic           # Utility (negative = good)
        )
        
        return expected_free_energy
    
    def select_action(self, 
                     current_state: Dict[str, Any],
                     predicted_next_state: Optional[Dict[str, Any]] = None) -> Action:
        """
        Autonomously select optimal action
        
        Args:
            current_state: Current system state
            predicted_next_state: Optional predicted next state
            
        Returns:
            Selected action
        """
        if predicted_next_state is None:
            predicted_next_state = current_state  # Assume no change if not provided
        
        # Generate candidate actions
        uncertainty = current_state.get('uncertainty', 0.5)
        candidates = self.generate_action_candidates(current_state, uncertainty)
        
        if not candidates:
            # Default to exploration if no candidates
            return Action(
                action_type=ActionType.EXPLORE,
                parameters={'strategy': 'random'},
                epistemic_value=1.0
            )
        
        # Evaluate expected free energy for each candidate
        for action in candidates:
            efe = self.compute_expected_free_energy(
                action, current_state, predicted_next_state
            )
            action.expected_free_energy = efe
        
        # Select action with minimum expected free energy
        optimal_action = min(candidates, key=lambda a: a.expected_free_energy)
        
        # Record action
        self.action_history.append(optimal_action)
        
        return optimal_action
    
    def generate_policy(self, current_state: Dict[str, Any]) -> Policy:
        """
        Generate optimal policy (sequence of actions)
        
        Args:
            current_state: Current system state
            
        Returns:
            Optimal policy
        """
        policies = []
        
        for _ in range(self.n_policies):
            actions = []
            simulated_state = current_state.copy()
            
            # Generate action sequence
            for step in range(self.horizon):
                # Select action for this step
                action = self.select_action(simulated_state)
                actions.append(action)
                
                # Simulate next state (simple prediction)
                simulated_state = self._simulate_next_state(simulated_state, action)
            
            # Create policy
            policy = Policy(actions=actions, horizon=self.horizon)
            policy.compute_expected_free_energy()
            policies.append(policy)
        
        # Select policy with minimum expected free energy
        optimal_policy = min(policies, key=lambda p: p.expected_free_energy)
        
        return optimal_policy
    
    def execute_action(self, action: Action) -> Dict[str, Any]:
        """
        Execute an action and return results
        
        Args:
            action: Action to execute
            
        Returns:
            Action results/effects
        """
        # This would interface with actual system components
        # For now, return a structure describing the action
        
        result = {
            'action_type': action.action_type.value,
            'parameters': action.parameters,
            'expected_free_energy': action.expected_free_energy,
            'timestamp': len(self.action_history),
        }
        
        # Track outcomes for learning
        action_key = action.action_type.value
        if action_key not in self.action_outcomes:
            self.action_outcomes[action_key] = []
        self.action_outcomes[action_key].append(action.expected_free_energy)
        
        return result
    
    def autonomous_step(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform one autonomous decision-action cycle
        
        This is the core autonomous loop:
        1. Select optimal action
        2. Execute action
        3. Return results
        
        Args:
            current_state: Current system state
            
        Returns:
            Action results
        """
        # Select action autonomously
        action = self.select_action(current_state)
        
        # Execute action
        result = self.execute_action(action)
        
        return result
    
    def set_goals(self, goals: Dict[str, float]):
        """
        Set goals for pragmatic value computation
        
        Args:
            goals: Dictionary of goal_name -> importance (0-1)
        """
        self.goals = goals
    
    def adjust_exploration_drive(self, drive: float):
        """
        Adjust intrinsic exploration motivation
        
        Args:
            drive: Exploration drive [0, 1]
        """
        self.exploration_bonus = np.clip(drive, 0.0, 1.0)
    
    def get_action_summary(self) -> str:
        """
        Get summary of recent actions
        
        Returns:
            Human-readable action summary
        """
        if not self.action_history:
            return "No actions taken yet"
        
        recent_actions = self.action_history[-5:]
        summary_parts = []
        
        for i, action in enumerate(recent_actions, 1):
            summary_parts.append(
                f"{i}. {action.action_type.value}: "
                f"EFE={action.expected_free_energy:.3f}, "
                f"epistemic={action.epistemic_value:.2f}, "
                f"pragmatic={action.pragmatic_value:.2f}"
            )
        
        return "\n".join(summary_parts)
    
    def _simulate_next_state(self, state: Dict[str, Any], action: Action) -> Dict[str, Any]:
        """
        Simple state transition simulation
        
        Args:
            state: Current state
            action: Action taken
            
        Returns:
            Predicted next state
        """
        next_state = state.copy()
        
        # Simple heuristics for state changes
        if action.action_type == ActionType.EXPLORE:
            # Exploration increases information but also uncertainty initially
            next_state['uncertainty'] = state.get('uncertainty', 0.5) * 1.1
            next_state['learning_opportunity'] = state.get('learning_opportunity', 0.0) + 0.2
        
        elif action.action_type == ActionType.REFINE:
            # Refining reduces uncertainty
            next_state['uncertainty'] = state.get('uncertainty', 0.5) * 0.8
            next_state['confidence'] = min(state.get('confidence', 0.5) + 0.1, 1.0)
        
        elif action.action_type == ActionType.EXPLOIT:
            # Exploitation uses current knowledge
            next_state['confidence'] = state.get('confidence', 0.5)
            next_state['goal_progress'] = state.get('goal_progress', 0.0) + 0.1
        
        return next_state

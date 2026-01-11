"""
Active Attention Mechanism

Implements dynamic attention allocation based on:
- Salience detection
- Expected free energy (epistemic + pragmatic value)
- Relevance to current goals
- Surprise and novelty

Based on theories of attention, salience, and active inference
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import heapq


@dataclass
class AttentionTarget:
    """Represents something that can receive attention"""
    id: str
    features: np.ndarray
    salience: float = 0.0
    relevance: float = 0.0
    epistemic_value: float = 0.0  # Information gain
    pragmatic_value: float = 0.0  # Goal achievement
    total_value: float = 0.0
    
    def compute_total_value(self, epistemic_weight: float = 0.5, pragmatic_weight: float = 0.5):
        """Compute total attention value"""
        self.total_value = (
            self.salience * 0.3 +
            self.relevance * 0.2 +
            self.epistemic_value * epistemic_weight +
            self.pragmatic_value * pragmatic_weight
        )
        return self.total_value
    
    def __lt__(self, other):
        """For heap operations"""
        return self.total_value > other.total_value  # Max heap


class ActiveAttentionMechanism:
    """
    Active Attention Mechanism for dynamic focus allocation
    
    Implements:
    - Bottom-up salience detection
    - Top-down goal-driven attention
    - Expected free energy minimization
    - Attentional switching and inhibition of return
    """
    
    def __init__(self, capacity: int = 5, switching_threshold: float = 0.3):
        """
        Initialize Active Attention Mechanism
        
        Args:
            capacity: Maximum number of concurrent attention targets
            switching_threshold: Threshold for switching attention
        """
        self.capacity = capacity
        self.switching_threshold = switching_threshold
        
        # Current attention allocation
        self.current_focus: List[AttentionTarget] = []
        
        # Attention history (inhibition of return)
        self.attention_history: List[str] = []
        self.inhibition_decay = 0.9
        
        # Goals/priors for top-down attention
        self.goals: Dict[str, np.ndarray] = {}
        
        # Weights for attention computation
        self.epistemic_weight = 0.5  # Exploration
        self.pragmatic_weight = 0.5  # Exploitation
        
    def detect_salience(self, features: np.ndarray) -> float:
        """
        Detect bottom-up salience of features
        
        Salience = distinctiveness, contrast, surprise
        
        Args:
            features: Feature vector
            
        Returns:
            Salience score [0, 1]
        """
        # Compute variance as proxy for distinctiveness
        feature_variance = np.var(features)
        
        # Compute magnitude (intensity)
        feature_magnitude = np.linalg.norm(features)
        
        # Normalize salience
        salience = np.tanh(feature_variance + 0.1 * feature_magnitude)
        
        return float(salience)
    
    def compute_relevance(self, features: np.ndarray) -> float:
        """
        Compute top-down relevance to current goals
        
        Args:
            features: Feature vector
            
        Returns:
            Relevance score [0, 1]
        """
        if not self.goals:
            return 0.5  # Neutral if no goals
        
        # Compute similarity to each goal
        relevances = []
        for goal_features in self.goals.values():
            # Cosine similarity
            if len(goal_features) == len(features):
                similarity = np.dot(features, goal_features) / (
                    np.linalg.norm(features) * np.linalg.norm(goal_features) + 1e-8
                )
                relevances.append((similarity + 1.0) / 2.0)  # Normalize to [0, 1]
        
        return float(np.max(relevances)) if relevances else 0.5
    
    def compute_epistemic_value(self, features: np.ndarray, 
                               current_uncertainty: float) -> float:
        """
        Compute epistemic value (information gain potential)
        
        Args:
            features: Feature vector
            current_uncertainty: Current uncertainty in beliefs
            
        Returns:
            Epistemic value [0, 1]
        """
        # Novel/uncertain features have high epistemic value
        # Check how different this is from current focus
        
        if not self.current_focus:
            return current_uncertainty  # Everything is valuable when starting
        
        # Compute distance from current focus
        distances = []
        for target in self.current_focus:
            if len(target.features) == len(features):
                dist = np.linalg.norm(features - target.features)
                distances.append(dist)
        
        # Higher distance = more potential for information gain
        avg_distance = np.mean(distances) if distances else 1.0
        epistemic_value = np.tanh(avg_distance) * current_uncertainty
        
        return float(epistemic_value)
    
    def compute_pragmatic_value(self, features: np.ndarray,
                               expected_reward: float = 0.5) -> float:
        """
        Compute pragmatic value (utility for achieving goals)
        
        Args:
            features: Feature vector
            expected_reward: Expected reward/utility
            
        Returns:
            Pragmatic value [0, 1]
        """
        # Relevance to goals weighted by expected reward
        relevance = self.compute_relevance(features)
        pragmatic_value = relevance * expected_reward
        
        return float(pragmatic_value)
    
    def apply_inhibition_of_return(self, target_id: str) -> float:
        """
        Apply inhibition to recently attended targets
        
        Args:
            target_id: Target identifier
            
        Returns:
            Inhibition factor [0, 1] where 1 = no inhibition
        """
        if target_id not in self.attention_history:
            return 1.0
        
        # More recent = stronger inhibition
        history_position = len(self.attention_history) - self.attention_history.index(target_id) - 1
        inhibition = self.inhibition_decay ** history_position
        
        return inhibition
    
    def allocate_attention(self, 
                          candidates: List[Tuple[str, np.ndarray]], 
                          current_uncertainty: float = 0.5,
                          expected_rewards: Optional[Dict[str, float]] = None) -> List[AttentionTarget]:
        """
        Allocate attention across candidate targets
        
        Args:
            candidates: List of (id, features) tuples
            current_uncertainty: Current belief uncertainty
            expected_rewards: Optional dict of expected rewards per candidate
            
        Returns:
            List of attention targets, ranked by total value
        """
        if expected_rewards is None:
            expected_rewards = {}
        
        # Evaluate each candidate
        attention_targets = []
        
        for target_id, features in candidates:
            target = AttentionTarget(id=target_id, features=features)
            
            # Compute salience (bottom-up)
            target.salience = self.detect_salience(features)
            
            # Compute relevance (top-down)
            target.relevance = self.compute_relevance(features)
            
            # Compute epistemic value (exploration)
            target.epistemic_value = self.compute_epistemic_value(features, current_uncertainty)
            
            # Compute pragmatic value (exploitation)
            expected_reward = expected_rewards.get(target_id, 0.5)
            target.pragmatic_value = self.compute_pragmatic_value(features, expected_reward)
            
            # Compute total value
            target.compute_total_value(self.epistemic_weight, self.pragmatic_weight)
            
            # Apply inhibition of return
            inhibition = self.apply_inhibition_of_return(target_id)
            target.total_value *= inhibition
            
            attention_targets.append(target)
        
        # Sort by total value and select top K
        attention_targets.sort(key=lambda x: x.total_value, reverse=True)
        selected = attention_targets[:self.capacity]
        
        # Update current focus
        self.current_focus = selected
        
        # Update attention history
        for target in selected:
            self.attention_history.append(target.id)
            if len(self.attention_history) > 50:  # Keep limited history
                self.attention_history.pop(0)
        
        return selected
    
    def should_switch_attention(self, new_salience: float) -> bool:
        """
        Decide whether to switch attention to new stimulus
        
        Args:
            new_salience: Salience of new potential target
            
        Returns:
            True if should switch attention
        """
        if not self.current_focus:
            return True
        
        # Compare to current minimum attention value
        min_current_value = min(t.total_value for t in self.current_focus)
        
        return new_salience > (min_current_value + self.switching_threshold)
    
    def set_goals(self, goals: Dict[str, np.ndarray]):
        """
        Set top-down goals for attention guidance
        
        Args:
            goals: Dictionary of goal_name -> goal_feature_vector
        """
        self.goals = goals
    
    def adjust_exploration_exploitation(self, epistemic_weight: float):
        """
        Adjust exploration vs exploitation balance
        
        Args:
            epistemic_weight: Weight for epistemic value [0, 1]
                             0 = pure exploitation, 1 = pure exploration
        """
        self.epistemic_weight = np.clip(epistemic_weight, 0.0, 1.0)
        self.pragmatic_weight = 1.0 - self.epistemic_weight
    
    def get_attention_distribution(self) -> Dict[str, float]:
        """
        Get current attention distribution across targets
        
        Returns:
            Dictionary mapping target_id to attention weight
        """
        if not self.current_focus:
            return {}
        
        # Softmax over values
        values = np.array([t.total_value for t in self.current_focus])
        weights = np.exp(values) / np.sum(np.exp(values))
        
        distribution = {
            target.id: float(weight)
            for target, weight in zip(self.current_focus, weights)
        }
        
        return distribution
    
    def get_focus_summary(self) -> str:
        """
        Get human-readable summary of current attention focus
        
        Returns:
            Summary string
        """
        if not self.current_focus:
            return "No current focus"
        
        summary_parts = []
        for i, target in enumerate(self.current_focus[:3], 1):
            summary_parts.append(
                f"{i}. {target.id}: salience={target.salience:.2f}, "
                f"relevance={target.relevance:.2f}, "
                f"epistemic={target.epistemic_value:.2f}, "
                f"value={target.total_value:.2f}"
            )
        
        return "\n".join(summary_parts)

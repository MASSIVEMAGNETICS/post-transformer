"""
Active Inference Engine

Based on the Free Energy Principle (Karl Friston et al.)
Implements predictive processing and belief updating through
variational Bayesian inference.

Key concepts:
- Free Energy Minimization
- Prediction Error Minimization
- Precision-Weighted Prediction
- Generative Model
- Belief Updating
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field


@dataclass
class BeliefState:
    """Represents the agent's beliefs about hidden states"""
    mean: np.ndarray
    precision: np.ndarray  # Inverse covariance (confidence)
    
    def update(self, prediction_error: np.ndarray, learning_rate: float = 0.1):
        """Update beliefs based on prediction error"""
        self.mean += learning_rate * self.precision * prediction_error


@dataclass
class GenerativeModel:
    """
    Generative model P(observations, states)
    Maps from hidden states to predicted observations
    """
    state_dim: int
    obs_dim: int
    transition_matrix: np.ndarray = None
    observation_matrix: np.ndarray = None
    
    def __post_init__(self):
        if self.transition_matrix is None:
            # Initialize with identity + small noise for stability
            self.transition_matrix = np.eye(self.state_dim) * 0.95 + np.random.randn(self.state_dim, self.state_dim) * 0.01
        if self.observation_matrix is None:
            # Initialize observation mapping
            self.observation_matrix = np.random.randn(self.obs_dim, self.state_dim) * 0.1
    
    def predict_observation(self, state: np.ndarray) -> np.ndarray:
        """Predict observations from current state"""
        return self.observation_matrix @ state
    
    def predict_next_state(self, state: np.ndarray) -> np.ndarray:
        """Predict next state from current state"""
        return self.transition_matrix @ state


class ActiveInferenceEngine:
    """
    Core Active Inference Engine implementing the Free Energy Principle
    
    The system continuously:
    1. Predicts sensory inputs
    2. Computes prediction errors
    3. Updates beliefs to minimize free energy
    4. Selects actions to minimize expected free energy
    """
    
    def __init__(self, state_dim: int = 64, obs_dim: int = 128):
        """
        Initialize the Active Inference Engine
        
        Args:
            state_dim: Dimensionality of hidden state space
            obs_dim: Dimensionality of observation space
        """
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        
        # Generative model
        self.generative_model = GenerativeModel(state_dim, obs_dim)
        
        # Current belief state
        self.belief = BeliefState(
            mean=np.zeros(state_dim),
            precision=np.eye(state_dim)
        )
        
        # Prediction error tracking
        self.prediction_errors: List[float] = []
        self.free_energy_history: List[float] = []
        
        # Hyperparameters
        self.learning_rate = 0.1
        self.precision_learning_rate = 0.01
        
    def compute_prediction_error(self, observation: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute precision-weighted prediction error
        
        Args:
            observation: Current sensory observation
            
        Returns:
            Tuple of (prediction_error_vector, magnitude)
        """
        # Predict observation from current beliefs
        predicted_obs = self.generative_model.predict_observation(self.belief.mean)
        
        # Compute raw prediction error
        error = observation - predicted_obs
        
        # Precision-weighted error (confidence-weighted)
        # Higher precision = more confidence = larger weight
        obs_precision = np.eye(self.obs_dim) * np.mean(np.diag(self.belief.precision))
        weighted_error = obs_precision @ error
        
        # Magnitude for tracking
        magnitude = np.linalg.norm(weighted_error)
        
        return weighted_error, magnitude
    
    def compute_free_energy(self, observation: np.ndarray) -> float:
        """
        Compute variational free energy
        
        Free Energy = Complexity - Accuracy
        Where:
        - Complexity: KL divergence between beliefs and prior
        - Accuracy: Log likelihood of observations given beliefs
        
        Minimizing free energy = maximizing evidence lower bound (ELBO)
        """
        # Predict observation
        predicted_obs = self.generative_model.predict_observation(self.belief.mean)
        
        # Accuracy term: squared prediction error
        prediction_error = observation - predicted_obs
        accuracy = -0.5 * np.sum(prediction_error ** 2)
        
        # Complexity term: divergence from prior (assume zero-mean prior)
        complexity = 0.5 * np.sum(self.belief.mean ** 2)
        
        # Free energy
        free_energy = complexity - accuracy
        
        return free_energy
    
    def update_beliefs(self, observation: np.ndarray):
        """
        Update beliefs through variational message passing
        
        This implements one step of belief updating to minimize
        free energy (prediction error)
        """
        # Compute prediction error
        weighted_error, error_magnitude = self.compute_prediction_error(observation)
        
        # Map observation error back to state space
        state_error = self.generative_model.observation_matrix.T @ weighted_error[:self.obs_dim]
        
        # Update beliefs (gradient descent on free energy)
        self.belief.update(state_error, self.learning_rate)
        
        # Update precision based on prediction error (meta-learning)
        # Smaller errors -> increase precision (confidence)
        precision_update = self.precision_learning_rate * (1.0 / (1.0 + error_magnitude))
        self.belief.precision *= (1.0 + precision_update)
        
        # Track metrics
        self.prediction_errors.append(error_magnitude)
        free_energy = self.compute_free_energy(observation)
        self.free_energy_history.append(free_energy)
        
    def predict_next_state(self) -> np.ndarray:
        """
        Predict next hidden state using generative model
        
        Returns:
            Predicted next state mean
        """
        return self.generative_model.predict_next_state(self.belief.mean)
    
    def infer(self, observation: np.ndarray, n_iterations: int = 5) -> Dict[str, Any]:
        """
        Perform active inference given an observation
        
        Args:
            observation: Current sensory input
            n_iterations: Number of belief update iterations
            
        Returns:
            Dictionary with inference results
        """
        # Iterative belief updating (multiple passes)
        for _ in range(n_iterations):
            self.update_beliefs(observation)
        
        # Predict future
        predicted_next_state = self.predict_next_state()
        predicted_next_obs = self.generative_model.predict_observation(predicted_next_state)
        
        return {
            'current_belief': self.belief.mean.copy(),
            'belief_precision': np.mean(np.diag(self.belief.precision)),
            'predicted_next_state': predicted_next_state,
            'predicted_next_observation': predicted_next_obs,
            'prediction_error': self.prediction_errors[-1] if self.prediction_errors else 0.0,
            'free_energy': self.free_energy_history[-1] if self.free_energy_history else 0.0,
        }
    
    def get_epistemic_value(self, potential_observation: np.ndarray) -> float:
        """
        Compute epistemic value (information gain) of a potential observation
        
        Used for active learning and exploration
        
        Args:
            potential_observation: Hypothetical future observation
            
        Returns:
            Epistemic value (expected information gain)
        """
        # Information gain = reduction in uncertainty
        # Approximated by reduction in prediction error variance
        
        current_uncertainty = 1.0 / np.mean(np.diag(self.belief.precision))
        
        # Simulate what would happen if we observed this
        predicted_obs = self.generative_model.predict_observation(self.belief.mean)
        error = np.linalg.norm(potential_observation - predicted_obs)
        
        # Expected uncertainty reduction
        epistemic_value = current_uncertainty * np.exp(-error)
        
        return epistemic_value
    
    def reset(self):
        """Reset the inference engine to initial state"""
        self.belief = BeliefState(
            mean=np.zeros(self.state_dim),
            precision=np.eye(self.state_dim)
        )
        self.prediction_errors.clear()
        self.free_energy_history.clear()

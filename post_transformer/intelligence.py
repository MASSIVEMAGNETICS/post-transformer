"""
Post-Transformer Intelligence

Revolutionary AI system integrating:
- Active Inference (Free Energy Principle)
- Active Awareness (Multi-level consciousness)
- Active Attention (Dynamic focus allocation)
- Autonomous Response (Self-directed action)

This system operates continuously and autonomously,
inferring, learning, and acting without requiring external input.
"""

import numpy as np
from typing import Dict, List, Optional, Any
import time

from .core.active_inference import ActiveInferenceEngine
from .core.awareness import ActiveAwarenessSystem
from .core.attention import ActiveAttentionMechanism, AttentionTarget
from .core.response import AutonomousResponseSystem, Action, ActionType


class PostTransformerIntelligence:
    """
    Revolutionary Post-Transformer Intelligence System
    
    Integrates all components into a unified autonomous intelligence:
    - Continuously infers and predicts (Active Inference)
    - Monitors itself and environment (Active Awareness)
    - Dynamically allocates attention (Active Attention)
    - Autonomously decides and acts (Autonomous Response)
    
    Key feature: Fully autonomous - operates without external prompting
    """
    
    def __init__(self, 
                 state_dim: int = 64,
                 obs_dim: int = 128,
                 attention_capacity: int = 5,
                 planning_horizon: int = 3):
        """
        Initialize the Post-Transformer Intelligence
        
        Args:
            state_dim: Dimensionality of hidden state space
            obs_dim: Dimensionality of observation space
            attention_capacity: Number of concurrent attention targets
            planning_horizon: Steps ahead for planning
        """
        # Core components
        self.inference_engine = ActiveInferenceEngine(state_dim, obs_dim)
        self.awareness_system = ActiveAwarenessSystem(history_size=100)
        self.attention_mechanism = ActiveAttentionMechanism(
            capacity=attention_capacity,
            switching_threshold=0.3
        )
        self.response_system = AutonomousResponseSystem(
            horizon=planning_horizon,
            n_policies=10
        )
        
        # System state
        self.running = False
        self.cycle_count = 0
        self.start_time = None
        
        # Performance metrics
        self.metrics = {
            'free_energy': [],
            'prediction_error': [],
            'confidence': [],
            'actions_taken': [],
        }
        
    def sense(self, observation: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Sense the environment (or generate internal observation if none provided)
        
        Args:
            observation: Optional external observation
            
        Returns:
            Observation vector
        """
        if observation is None:
            # Generate synthetic observation from internal state
            # This allows autonomous operation without external input
            observation = self._generate_internal_observation()
        
        return observation
    
    def _generate_internal_observation(self) -> np.ndarray:
        """
        Generate observation from internal state
        Allows system to operate autonomously
        """
        # Use current belief state plus some noise
        base = self.inference_engine.belief.mean
        
        # Pad or truncate to observation dimension
        if len(base) < self.inference_engine.obs_dim:
            obs = np.zeros(self.inference_engine.obs_dim)
            obs[:len(base)] = base
        else:
            obs = base[:self.inference_engine.obs_dim]
        
        # Add small noise for variability
        obs += np.random.randn(self.inference_engine.obs_dim) * 0.1
        
        return obs
    
    def infer(self, observation: np.ndarray) -> Dict[str, Any]:
        """
        Perform active inference on observation
        
        Args:
            observation: Sensory observation
            
        Returns:
            Inference results
        """
        inference_result = self.inference_engine.infer(observation, n_iterations=5)
        return inference_result
    
    def update_awareness(self, inference_result: Dict[str, Any]):
        """
        Update awareness system with inference results
        
        Args:
            inference_result: Results from active inference
        """
        # Extract metrics
        cognitive_metrics = {
            'cognitive_load': np.linalg.norm(inference_result['current_belief']) / 10.0,
            'uncertainty': 1.0 / max(inference_result['belief_precision'], 0.01),
            'exploration_drive': inference_result['prediction_error'] / 2.0,
            'stability': 1.0 - min(inference_result['prediction_error'], 1.0),
        }
        
        # Update awareness
        self.awareness_system.update(
            environmental_input={'cycle': self.cycle_count},
            cognitive_metrics=cognitive_metrics,
            belief_state=inference_result['current_belief'],
            prediction_error=inference_result['prediction_error']
        )
    
    def allocate_attention(self, inference_result: Dict[str, Any]) -> List[AttentionTarget]:
        """
        Allocate attention based on current state
        
        Args:
            inference_result: Results from active inference
            
        Returns:
            List of attention targets
        """
        # Create attention candidates from belief state
        # Split belief into chunks as different attention targets
        belief = inference_result['current_belief']
        chunk_size = max(len(belief) // 5, 1)
        
        candidates = []
        for i in range(0, len(belief), chunk_size):
            chunk = belief[i:i+chunk_size]
            if len(chunk) > 0:
                # Pad to consistent size
                padded = np.zeros(chunk_size)
                padded[:len(chunk)] = chunk
                candidates.append((f"state_chunk_{i}", padded))
        
        # Get current uncertainty
        uncertainty = 1.0 / max(inference_result['belief_precision'], 0.01)
        
        # Allocate attention
        attention_targets = self.attention_mechanism.allocate_attention(
            candidates,
            current_uncertainty=uncertainty
        )
        
        return attention_targets
    
    def select_and_act(self, inference_result: Dict[str, Any], 
                      awareness_state: Any) -> Dict[str, Any]:
        """
        Autonomously select and execute action
        
        Args:
            inference_result: Results from active inference
            awareness_state: Current awareness state
            
        Returns:
            Action results
        """
        # Build current state representation
        current_state = {
            'uncertainty': 1.0 / max(inference_result['belief_precision'], 0.01),
            'prediction_error': inference_result['prediction_error'],
            'free_energy': inference_result['free_energy'],
            'confidence': awareness_state.confidence,
            'attention_entropy': self._compute_attention_entropy(),
            'learning_opportunity': inference_result['prediction_error'] * 0.5,
        }
        
        # Autonomous action selection
        action_result = self.response_system.autonomous_step(current_state)
        
        return action_result
    
    def _compute_attention_entropy(self) -> float:
        """Compute entropy of current attention distribution"""
        dist = self.attention_mechanism.get_attention_distribution()
        if not dist:
            return 0.0
        
        weights = np.array(list(dist.values()))
        weights = weights / np.sum(weights)  # Normalize
        
        # Shannon entropy
        entropy = -np.sum(weights * np.log(weights + 1e-10))
        
        return float(entropy)
    
    def autonomous_cycle(self) -> Dict[str, Any]:
        """
        Execute one complete autonomous intelligence cycle
        
        This is the main loop:
        1. Sense (with or without external input)
        2. Infer (predict and update beliefs)
        3. Aware (monitor self and environment)
        4. Attend (allocate attention)
        5. Act (autonomous response)
        
        Returns:
            Cycle results
        """
        # 1. SENSE
        observation = self.sense()
        
        # 2. INFER
        inference_result = self.infer(observation)
        
        # 3. AWARE
        self.update_awareness(inference_result)
        awareness_state = self.awareness_system.get_awareness_state()
        
        # 4. ATTEND
        attention_targets = self.allocate_attention(inference_result)
        
        # 5. ACT
        action_result = self.select_and_act(inference_result, awareness_state)
        
        # Update metrics
        self.metrics['free_energy'].append(inference_result['free_energy'])
        self.metrics['prediction_error'].append(inference_result['prediction_error'])
        self.metrics['confidence'].append(awareness_state.confidence)
        self.metrics['actions_taken'].append(action_result['action_type'])
        
        # Increment cycle
        self.cycle_count += 1
        
        # Return comprehensive cycle results
        return {
            'cycle': self.cycle_count,
            'inference': inference_result,
            'awareness': {
                'confidence': awareness_state.confidence,
                'internal_state': awareness_state.internal_state,
                'metacognitive_state': awareness_state.metacognitive_state,
            },
            'attention': {
                'n_targets': len(attention_targets),
                'top_target': attention_targets[0].id if attention_targets else None,
                'distribution': self.attention_mechanism.get_attention_distribution(),
            },
            'action': action_result,
        }
    
    def run_autonomous(self, n_cycles: int = 10, verbose: bool = True) -> List[Dict[str, Any]]:
        """
        Run autonomous intelligence for multiple cycles
        
        Args:
            n_cycles: Number of cycles to run
            verbose: Whether to print progress
            
        Returns:
            List of cycle results
        """
        self.running = True
        self.start_time = time.time()
        results = []
        
        if verbose:
            print("=" * 70)
            print("POST-TRANSFORMER INTELLIGENCE: AUTONOMOUS OPERATION")
            print("=" * 70)
            print(f"Running {n_cycles} autonomous cycles...")
            print()
        
        for cycle in range(n_cycles):
            cycle_result = self.autonomous_cycle()
            results.append(cycle_result)
            
            if verbose and (cycle % max(n_cycles // 10, 1) == 0 or cycle == n_cycles - 1):
                self._print_cycle_status(cycle_result)
        
        self.running = False
        
        if verbose:
            print()
            print("=" * 70)
            self._print_summary()
            print("=" * 70)
        
        return results
    
    def _print_cycle_status(self, cycle_result: Dict[str, Any]):
        """Print status for a cycle"""
        print(f"\n--- Cycle {cycle_result['cycle']} ---")
        
        # Inference status
        inf = cycle_result['inference']
        print(f"Inference: Free Energy={inf['free_energy']:.3f}, "
              f"Prediction Error={inf['prediction_error']:.3f}, "
              f"Precision={inf['belief_precision']:.2f}")
        
        # Awareness status
        aware = cycle_result['awareness']
        print(f"Awareness: Confidence={aware['confidence']:.2f}, "
              f"Uncertainty={aware['internal_state'].get('uncertainty', 0):.2f}")
        
        # Attention status
        att = cycle_result['attention']
        print(f"Attention: {att['n_targets']} targets, "
              f"Focus on: {att['top_target']}")
        
        # Action status
        act = cycle_result['action']
        print(f"Action: {act['action_type']} "
              f"(EFE={act['expected_free_energy']:.3f})")
    
    def _print_summary(self):
        """Print summary of autonomous operation"""
        duration = time.time() - self.start_time if self.start_time else 0
        
        print("\nSUMMARY OF AUTONOMOUS OPERATION")
        print(f"Total cycles: {self.cycle_count}")
        print(f"Duration: {duration:.2f}s")
        print(f"Cycles per second: {self.cycle_count / duration if duration > 0 else 0:.2f}")
        
        if self.metrics['free_energy']:
            print(f"\nFree Energy: {np.mean(self.metrics['free_energy'][-10:]):.3f} "
                  f"(trend: {self._compute_trend(self.metrics['free_energy'])})")
        
        if self.metrics['prediction_error']:
            print(f"Prediction Error: {np.mean(self.metrics['prediction_error'][-10:]):.3f} "
                  f"(trend: {self._compute_trend(self.metrics['prediction_error'])})")
        
        if self.metrics['confidence']:
            print(f"Confidence: {np.mean(self.metrics['confidence'][-10:]):.2f} "
                  f"(trend: {self._compute_trend(self.metrics['confidence'])})")
        
        # Action distribution
        if self.metrics['actions_taken']:
            from collections import Counter
            action_counts = Counter(self.metrics['actions_taken'][-20:])
            print(f"\nRecent Actions:")
            for action, count in action_counts.most_common():
                print(f"  {action}: {count}")
    
    def _compute_trend(self, values: List[float]) -> str:
        """Compute trend direction"""
        if len(values) < 2:
            return "stable"
        
        recent = values[-10:]
        if len(recent) < 2:
            return "stable"
        
        trend = np.mean(np.diff(recent))
        if abs(trend) < 0.01:
            return "stable"
        elif trend > 0:
            return "increasing ↑"
        else:
            return "decreasing ↓"
    
    def get_state_report(self) -> str:
        """
        Get comprehensive state report
        
        Returns:
            Human-readable state report
        """
        report_parts = []
        
        report_parts.append("=" * 70)
        report_parts.append("POST-TRANSFORMER INTELLIGENCE: STATE REPORT")
        report_parts.append("=" * 70)
        
        # Inference state
        report_parts.append("\n[ACTIVE INFERENCE]")
        report_parts.append(f"Belief state dimension: {self.inference_engine.state_dim}")
        report_parts.append(f"Recent free energy: {self.inference_engine.free_energy_history[-1]:.3f}" if self.inference_engine.free_energy_history else "Recent free energy: N/A")
        report_parts.append(f"Recent prediction error: {self.inference_engine.prediction_errors[-1]:.3f}" if self.inference_engine.prediction_errors else "Recent prediction error: N/A")
        
        # Awareness state
        report_parts.append("\n[ACTIVE AWARENESS]")
        report_parts.append(self.awareness_system.get_situation_summary())
        
        # Attention state
        report_parts.append("\n[ACTIVE ATTENTION]")
        report_parts.append(self.attention_mechanism.get_focus_summary())
        
        # Response state
        report_parts.append("\n[AUTONOMOUS RESPONSE]")
        report_parts.append(self.response_system.get_action_summary())
        
        report_parts.append("\n" + "=" * 70)
        
        return "\n".join(report_parts)

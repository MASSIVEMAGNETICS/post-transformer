"""
Active Awareness System

Implements multi-level awareness including:
- Environmental awareness (external context)
- Interoceptive awareness (internal state monitoring)
- Metacognitive awareness (self-modeling)
- Temporal awareness (history and future projection)

Based on theories of consciousness and metacognition
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import deque
import time


@dataclass
class AwarenessState:
    """Represents the current awareness state"""
    environmental_context: Dict[str, Any] = field(default_factory=dict)
    internal_state: Dict[str, float] = field(default_factory=dict)
    metacognitive_state: Dict[str, float] = field(default_factory=dict)
    temporal_context: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5
    timestamp: float = 0.0


class ActiveAwarenessSystem:
    """
    Active Awareness System for continuous self and environmental monitoring
    
    Implements:
    - Multi-modal sensory integration
    - Self-monitoring (metacognition)
    - Environmental context tracking
    - Temporal awareness (past patterns, future expectations)
    """
    
    def __init__(self, history_size: int = 100):
        """
        Initialize the Active Awareness System
        
        Args:
            history_size: Number of historical states to maintain
        """
        self.history_size = history_size
        
        # Current awareness state
        self.current_state = AwarenessState(timestamp=time.time())
        
        # Historical states for pattern detection
        self.state_history: deque = deque(maxlen=history_size)
        
        # Environmental features tracking
        self.environmental_features: Dict[str, Any] = {}
        
        # Internal monitoring
        self.internal_metrics = {
            'cognitive_load': 0.0,
            'uncertainty': 0.5,
            'exploration_drive': 0.5,
            'stability': 1.0,
        }
        
        # Metacognitive monitoring
        self.metacognitive_metrics = {
            'confidence_in_beliefs': 0.5,
            'awareness_of_uncertainty': 0.5,
            'learning_progress': 0.0,
            'self_coherence': 1.0,
        }
        
    def sense_environment(self, environmental_input: Dict[str, Any]):
        """
        Process environmental sensory input
        
        Args:
            environmental_input: Dictionary of environmental features/observations
        """
        # Update environmental features
        self.environmental_features.update(environmental_input)
        
        # Extract and compute environmental metrics
        self.current_state.environmental_context = {
            'n_features': len(self.environmental_features),
            'feature_keys': list(self.environmental_features.keys()),
            'environment_complexity': self._compute_complexity(environmental_input),
            'environment_novelty': self._compute_novelty(environmental_input),
        }
        
    def monitor_internal_state(self, cognitive_metrics: Dict[str, float]):
        """
        Monitor internal cognitive state (interoception for AI)
        
        Args:
            cognitive_metrics: Dictionary of internal cognitive measurements
        """
        # Update internal metrics
        self.internal_metrics.update(cognitive_metrics)
        
        # Compute derived internal awareness
        self.current_state.internal_state = {
            'cognitive_load': self.internal_metrics.get('cognitive_load', 0.0),
            'uncertainty': self.internal_metrics.get('uncertainty', 0.5),
            'exploration_drive': self.internal_metrics.get('exploration_drive', 0.5),
            'stability': self.internal_metrics.get('stability', 1.0),
            'arousal': self._compute_arousal(),
        }
        
    def metacognitive_monitoring(self, belief_state: np.ndarray, 
                                 prediction_error: float,
                                 learning_rate: float):
        """
        Metacognitive awareness - monitoring own cognitive processes
        
        Args:
            belief_state: Current belief state from inference
            prediction_error: Current prediction error
            learning_rate: Current learning rate
        """
        # Update confidence based on prediction error
        confidence = 1.0 / (1.0 + prediction_error)
        self.metacognitive_metrics['confidence_in_beliefs'] = confidence
        
        # Awareness of uncertainty
        belief_variance = np.var(belief_state)
        self.metacognitive_metrics['awareness_of_uncertainty'] = float(belief_variance)
        
        # Track learning progress (reduction in error over time)
        if len(self.state_history) > 1:
            prev_confidence = self.state_history[-1].metacognitive_state.get('confidence_in_beliefs', 0.5)
            progress = confidence - prev_confidence
            self.metacognitive_metrics['learning_progress'] = progress
        
        # Self-coherence (consistency of beliefs)
        self.metacognitive_metrics['self_coherence'] = 1.0 - min(belief_variance, 1.0)
        
        self.current_state.metacognitive_state = self.metacognitive_metrics.copy()
        self.current_state.confidence = confidence
        
    def update_temporal_awareness(self):
        """
        Build awareness of temporal context (past patterns, trends)
        """
        if len(self.state_history) < 2:
            return
        
        # Analyze trends in recent history
        recent_confidences = [s.confidence for s in list(self.state_history)[-10:]]
        recent_uncertainty = [s.internal_state.get('uncertainty', 0.5) 
                            for s in list(self.state_history)[-10:]]
        
        self.current_state.temporal_context = {
            'confidence_trend': np.mean(np.diff(recent_confidences)) if len(recent_confidences) > 1 else 0.0,
            'uncertainty_trend': np.mean(np.diff(recent_uncertainty)) if len(recent_uncertainty) > 1 else 0.0,
            'time_since_start': time.time() - self.state_history[0].timestamp if self.state_history else 0.0,
            'n_historical_states': len(self.state_history),
        }
        
    def get_awareness_state(self) -> AwarenessState:
        """
        Get current comprehensive awareness state
        
        Returns:
            Current AwarenessState with all components
        """
        self.current_state.timestamp = time.time()
        return self.current_state
    
    def update(self, environmental_input: Optional[Dict[str, Any]] = None,
              cognitive_metrics: Optional[Dict[str, float]] = None,
              belief_state: Optional[np.ndarray] = None,
              prediction_error: Optional[float] = None) -> AwarenessState:
        """
        Comprehensive awareness update
        
        Args:
            environmental_input: Environmental observations
            cognitive_metrics: Internal cognitive state
            belief_state: Current belief state
            prediction_error: Current prediction error
            
        Returns:
            Updated awareness state
        """
        # Process environmental input
        if environmental_input is not None:
            self.sense_environment(environmental_input)
        
        # Monitor internal state
        if cognitive_metrics is not None:
            self.monitor_internal_state(cognitive_metrics)
        
        # Metacognitive monitoring
        if belief_state is not None and prediction_error is not None:
            self.metacognitive_monitoring(belief_state, prediction_error, 0.1)
        
        # Update temporal awareness
        self.update_temporal_awareness()
        
        # Store current state in history
        self.state_history.append(self.current_state)
        
        return self.get_awareness_state()
    
    def _compute_complexity(self, data: Dict[str, Any]) -> float:
        """Compute complexity of environmental input"""
        # Simple heuristic: number of features and their variance
        if not data:
            return 0.0
        
        numeric_values = [v for v in data.values() if isinstance(v, (int, float))]
        if numeric_values:
            return float(len(data) * (1.0 + np.std(numeric_values)))
        return float(len(data))
    
    def _compute_novelty(self, data: Dict[str, Any]) -> float:
        """Compute novelty of environmental input"""
        # Compare to recent history
        if len(self.state_history) < 5:
            return 1.0  # Everything is novel at the start
        
        # Check how different current input is from recent inputs
        recent_keys = set()
        for state in list(self.state_history)[-5:]:
            recent_keys.update(state.environmental_context.get('feature_keys', []))
        
        current_keys = set(data.keys())
        novelty = len(current_keys - recent_keys) / max(len(current_keys), 1)
        
        return float(novelty)
    
    def _compute_arousal(self) -> float:
        """
        Compute arousal level based on uncertainty and cognitive load
        
        High arousal = high uncertainty or high cognitive load
        """
        uncertainty = self.internal_metrics.get('uncertainty', 0.5)
        cognitive_load = self.internal_metrics.get('cognitive_load', 0.0)
        
        arousal = (uncertainty + cognitive_load) / 2.0
        return float(np.clip(arousal, 0.0, 1.0))
    
    def get_situation_summary(self) -> str:
        """
        Generate natural language summary of current awareness
        
        Returns:
            Human-readable situation summary
        """
        state = self.get_awareness_state()
        
        summary_parts = []
        
        # Environmental awareness
        env = state.environmental_context
        summary_parts.append(f"Environmental: {env.get('n_features', 0)} features, "
                           f"complexity={env.get('environment_complexity', 0.0):.2f}, "
                           f"novelty={env.get('environment_novelty', 0.0):.2f}")
        
        # Internal state
        internal = state.internal_state
        summary_parts.append(f"Internal: load={internal.get('cognitive_load', 0.0):.2f}, "
                           f"uncertainty={internal.get('uncertainty', 0.5):.2f}, "
                           f"arousal={internal.get('arousal', 0.0):.2f}")
        
        # Metacognitive state
        meta = state.metacognitive_state
        summary_parts.append(f"Metacognitive: confidence={meta.get('confidence_in_beliefs', 0.5):.2f}, "
                           f"learning_progress={meta.get('learning_progress', 0.0):.3f}")
        
        return " | ".join(summary_parts)

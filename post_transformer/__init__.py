"""
Post-Transformer Intelligence System

A revolutionary AI system implementing:
- Active Inference (Free Energy Principle)
- Active Awareness (Self-monitoring and environmental awareness)
- Active Attention (Dynamic salience-based attention)
- Autonomous Response (Self-directed action selection)
"""

__version__ = "0.1.0"
__author__ = "MASSIVEMAGNETICS"

from .core.active_inference import ActiveInferenceEngine
from .core.awareness import ActiveAwarenessSystem
from .core.attention import ActiveAttentionMechanism
from .core.response import AutonomousResponseSystem
from .intelligence import PostTransformerIntelligence

__all__ = [
    "ActiveInferenceEngine",
    "ActiveAwarenessSystem",
    "ActiveAttentionMechanism",
    "AutonomousResponseSystem",
    "PostTransformerIntelligence",
]

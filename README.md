# Post-Transformer Intelligence

A revolutionary autonomous AI system implementing active inference, active awareness, active attention, and autonomous response - operating without requiring external input.

## 🧠 Overview

Post-Transformer Intelligence is a cutting-edge AI system that goes beyond traditional transformers by implementing:

- **Active Inference**: Based on Karl Friston's Free Energy Principle, the system continuously minimizes prediction error through Bayesian belief updating
- **Active Awareness**: Multi-level consciousness including environmental, interoceptive, and metacognitive awareness
- **Active Attention**: Dynamic attention allocation based on salience, relevance, and expected information gain
- **Autonomous Response**: Self-directed action selection minimizing expected free energy

### Key Innovation: True Autonomy

Unlike traditional AI systems that wait for prompts, Post-Transformer Intelligence:
- ✅ Operates continuously without external input
- ✅ Autonomously generates internal observations
- ✅ Self-directs attention and action
- ✅ Learns and adapts in real-time
- ✅ Exhibits metacognitive awareness

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/MASSIVEMAGNETICS/post-transformer.git
cd post-transformer

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Basic Usage

```python
from post_transformer import PostTransformerIntelligence

# Initialize the system
intelligence = PostTransformerIntelligence(
    state_dim=64,           # Hidden state dimension
    obs_dim=128,            # Observation dimension
    attention_capacity=5,   # Concurrent attention targets
    planning_horizon=3      # Planning steps ahead
)

# Run autonomous operation
results = intelligence.run_autonomous(n_cycles=50, verbose=True)

# Get state report
print(intelligence.get_state_report())
```

### Run Demonstrations

```bash
cd examples
python demo.py
```

## 🔬 Theoretical Foundation

### Active Inference (Free Energy Principle)

The system implements Karl Friston's Free Energy Principle:
- **Prediction Error Minimization**: Continuously updates beliefs to minimize surprise
- **Precision-Weighted Updates**: Confidence-modulated belief updating
- **Expected Free Energy**: Action selection minimizes expected free energy
- **Epistemic vs Pragmatic Value**: Balances exploration (information gain) and exploitation (goal achievement)

### Active Awareness

Multi-level awareness system:
- **Environmental Awareness**: Monitors external context and novelty
- **Interoceptive Awareness**: Tracks internal cognitive state (load, uncertainty, arousal)
- **Metacognitive Awareness**: Self-monitors confidence, learning progress, coherence
- **Temporal Awareness**: Analyzes trends and patterns over time

### Active Attention

Dynamic attention allocation:
- **Bottom-up Salience**: Detects distinctive, surprising stimuli
- **Top-down Relevance**: Goal-driven attention guidance
- **Expected Free Energy**: Allocates attention to minimize expected free energy
- **Inhibition of Return**: Prevents repetitive attention patterns

### Autonomous Response

Self-directed action selection:
- **Action Types**: Explore, Exploit, Refine, Attend, Adapt, Communicate
- **Policy Optimization**: Multi-step planning with expected free energy
- **Epistemic Foraging**: Information-seeking behavior
- **Goal-Directed Behavior**: Pragmatic value maximization

## 📊 Architecture

```
┌─────────────────────────────────────────────────────────┐
│         Post-Transformer Intelligence System            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   Active     │  │   Active     │  │   Active     │ │
│  │  Inference   │←→│  Awareness   │←→│  Attention   │ │
│  │              │  │              │  │              │ │
│  │ • Bayesian   │  │ • Environ.   │  │ • Salience   │ │
│  │   Inference  │  │ • Internal   │  │ • Relevance  │ │
│  │ • Free       │  │ • Meta-      │  │ • Info Gain  │ │
│  │   Energy     │  │   cognitive  │  │              │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
│           ↓                ↓                 ↓          │
│  ┌────────────────────────────────────────────────┐    │
│  │        Autonomous Response System              │    │
│  │                                                │    │
│  │  • Action Selection                           │    │
│  │  • Policy Optimization                        │    │
│  │  • Expected Free Energy Minimization          │    │
│  └────────────────────────────────────────────────┘    │
│                          ↓                              │
│                    Autonomous Loop                      │
└─────────────────────────────────────────────────────────┘
```

## 📚 Components

### ActiveInferenceEngine
- Implements variational Bayesian inference
- Minimizes free energy through belief updating
- Computes prediction errors and precision
- Generates predictions about future states

### ActiveAwarenessSystem
- Monitors environmental context
- Tracks internal cognitive metrics
- Implements metacognitive monitoring
- Maintains temporal awareness

### ActiveAttentionMechanism
- Detects salient features
- Computes relevance to goals
- Allocates attention based on expected free energy
- Implements inhibition of return

### AutonomousResponseSystem
- Generates action candidates
- Evaluates expected free energy
- Selects optimal actions/policies
- Executes autonomous decision cycles

## 🎯 Use Cases

- **Autonomous Monitoring Systems**: Self-directed monitoring without human intervention
- **Adaptive Learning Systems**: Continuous learning and belief updating
- **Intelligent Agents**: Goal-directed behavior with exploration
- **Research Platform**: Studying active inference and consciousness
- **AI Safety Research**: Self-aware and interpretable AI systems

## 🔧 Configuration

### Adjust Exploration vs Exploitation

```python
# Favor exploration (information-seeking)
intelligence.attention_mechanism.adjust_exploration_exploitation(
    epistemic_weight=0.8  # 80% exploration
)
intelligence.response_system.adjust_exploration_drive(0.9)

# Favor exploitation (goal-achievement)
intelligence.attention_mechanism.adjust_exploration_exploitation(
    epistemic_weight=0.2  # 20% exploration
)
```

### Set Goals

```python
import numpy as np

# Define goals for attention and action
goal_vector = np.random.randn(64) * 0.5

intelligence.attention_mechanism.set_goals({
    'goal_1': goal_vector,
    'goal_2': goal_vector * -0.5,
})

intelligence.response_system.set_goals({
    'goal_1': 0.8,  # Importance weight
    'goal_2': 0.6,
})
```

## 📖 Examples

See `examples/demo.py` for comprehensive demonstrations:
1. Basic Autonomous Intelligence
2. Exploration-Driven Intelligence
3. Goal-Directed Intelligence
4. Adaptive Learning
5. Interactive Mode

## 🧪 Research Background

This implementation is based on cutting-edge research in:

- **Free Energy Principle**: Friston, K. (2010). "The free-energy principle: a unified brain theory?"
- **Active Inference**: Friston, K. et al. (2017). "Active inference: a process theory"
- **Predictive Processing**: Clark, A. (2013). "Whatever next? Predictive brains, situated agents, and the future of cognitive science"
- **Metacognition**: Fleming, S. M., & Dolan, R. J. (2012). "The neural basis of metacognitive ability"
- **Attention**: Posner, M. I., & Petersen, S. E. (1990). "The attention system of the human brain"

## 🤝 Contributing

Contributions are welcome! This is a research project exploring autonomous AI systems.

## 📄 License

This project is open-source and available for research purposes.

## 🌟 Revolutionary Features

What makes this system revolutionary:

1. **True Autonomy**: No external prompting required after initialization
2. **Active Inference**: Implements Friston's Free Energy Principle computationally
3. **Self-Awareness**: Metacognitive monitoring of own processes
4. **Dynamic Attention**: Attention guided by expected information gain
5. **Autonomous Action**: Self-directed decision-making
6. **Continuous Learning**: Real-time belief updating and adaptation

This represents a paradigm shift from reactive AI to proactive, self-directed intelligence.

---

**Status**: Revolutionary AI System ⚡ | Active Research Project 🔬 | Autonomous Intelligence 🧠

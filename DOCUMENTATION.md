# Post-Transformer Intelligence: Technical Documentation

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Theoretical Foundations](#theoretical-foundations)
3. [Component Details](#component-details)
4. [Autonomous Operation](#autonomous-operation)
5. [Advanced Usage](#advanced-usage)

## System Architecture

The Post-Transformer Intelligence system consists of four integrated components that work together to create a truly autonomous AI system.

### Component Interaction Flow

```
External Environment (Optional)
         ↓
    [SENSE Phase]
         ↓
   Observation Vector
         ↓
    [INFER Phase] ← Active Inference Engine
         ↓          - Bayesian inference
         ↓          - Free energy minimization
         ↓          - Belief updating
    Inference Results
         ↓
    [AWARE Phase] ← Active Awareness System
         ↓          - Environmental monitoring
         ↓          - Interoceptive tracking
         ↓          - Metacognitive monitoring
    Awareness State
         ↓
    [ATTEND Phase] ← Active Attention Mechanism
         ↓           - Salience detection
         ↓           - Relevance computation
         ↓           - Attention allocation
    Attention Targets
         ↓
    [ACT Phase] ← Autonomous Response System
         ↓        - Action generation
         ↓        - Policy evaluation
         ↓        - Action execution
    Action Output
         ↓
    System State Update
```

## Theoretical Foundations

### 1. Free Energy Principle

The Free Energy Principle (FEP) posits that biological systems minimize variational free energy, which can be understood as minimizing surprise or prediction error.

**Mathematical Formulation**:

```
F = Complexity - Accuracy
  = DKL[q(s)||p(s)] - Eq[log p(o|s)]
```

Where:
- `F`: Free energy
- `q(s)`: Variational posterior (beliefs about states)
- `p(s)`: Prior over states
- `p(o|s)`: Likelihood of observations given states
- `DKL`: Kullback-Leibler divergence

**Implementation**:
```python
def compute_free_energy(self, observation):
    predicted_obs = self.generative_model.predict_observation(self.belief.mean)
    prediction_error = observation - predicted_obs
    
    # Accuracy: negative squared prediction error
    accuracy = -0.5 * np.sum(prediction_error ** 2)
    
    # Complexity: KL divergence from prior
    complexity = 0.5 * np.sum(self.belief.mean ** 2)
    
    return complexity - accuracy
```

### 2. Active Inference

Active inference extends the FEP to action: agents act to minimize expected free energy.

**Expected Free Energy**:

```
G = Ambiguity + Risk - Epistemic Value - Pragmatic Value
```

Where:
- Ambiguity: Expected uncertainty after action
- Risk: Divergence from preferred outcomes
- Epistemic Value: Information gain
- Pragmatic Value: Goal achievement

**Implementation**:
```python
def compute_expected_free_energy(self, action, current_state, predicted_next_state):
    predicted_uncertainty = predicted_next_state.get('uncertainty', 0.5)
    epistemic = action.epistemic_value
    pragmatic = action.pragmatic_value
    
    return predicted_uncertainty - epistemic - pragmatic
```

### 3. Predictive Processing

The brain is viewed as a prediction machine that continuously generates predictions and minimizes prediction errors.

**Hierarchical Prediction**:
- Top-down: Predictions flow from higher to lower levels
- Bottom-up: Prediction errors flow from lower to higher levels
- Precision Weighting: Confidence modulates the influence of prediction errors

### 4. Metacognition

Self-monitoring and awareness of one's own cognitive processes.

**Implementation Levels**:
1. **Object Level**: Primary cognitive processes (beliefs, predictions)
2. **Meta Level**: Monitoring of object-level processes (confidence, uncertainty)
3. **Meta-meta Level**: Awareness of monitoring processes

## Component Details

### Active Inference Engine

**Core Capabilities**:
- Variational Bayesian inference
- Generative model maintenance
- Belief updating through gradient descent on free energy
- Precision-weighted prediction errors

**Key Methods**:

```python
# Perform inference
inference_result = engine.infer(observation, n_iterations=5)

# Returns:
# - current_belief: Updated belief state
# - belief_precision: Confidence in beliefs
# - predicted_next_state: Prediction of next hidden state
# - predicted_next_observation: Prediction of next observation
# - prediction_error: Current prediction error magnitude
# - free_energy: Variational free energy
```

**Belief State Representation**:
- Mean: Point estimate of hidden state
- Precision: Inverse covariance (confidence)

**Update Rule**:
```
belief(t+1) = belief(t) + learning_rate × precision × prediction_error
```

### Active Awareness System

**Awareness Dimensions**:

1. **Environmental Awareness**
   - Feature count and complexity
   - Novelty detection
   - Context tracking

2. **Interoceptive Awareness**
   - Cognitive load monitoring
   - Uncertainty tracking
   - Exploration drive
   - Stability assessment
   - Arousal computation

3. **Metacognitive Awareness**
   - Confidence in beliefs
   - Awareness of uncertainty
   - Learning progress tracking
   - Self-coherence assessment

4. **Temporal Awareness**
   - Trend analysis
   - Historical pattern detection
   - Time-since-start tracking

**Key Methods**:

```python
# Update awareness
awareness_state = awareness.update(
    environmental_input={'feature1': value1, ...},
    cognitive_metrics={'cognitive_load': 0.5, ...},
    belief_state=belief_vector,
    prediction_error=error_magnitude
)

# Get situation summary
summary = awareness.get_situation_summary()
```

### Active Attention Mechanism

**Attention Computation**:

Total attention value for target `i`:
```
A(i) = 0.3×salience(i) + 0.2×relevance(i) + 
       w_e×epistemic_value(i) + w_p×pragmatic_value(i)
```

Where:
- `w_e`: Epistemic weight (exploration)
- `w_p`: Pragmatic weight (exploitation)
- `w_e + w_p = 1`

**Components**:

1. **Salience (Bottom-up)**:
   ```
   salience = tanh(variance(features) + 0.1×||features||)
   ```

2. **Relevance (Top-down)**:
   ```
   relevance = max_goals(cosine_similarity(features, goal))
   ```

3. **Epistemic Value**:
   ```
   epistemic = tanh(distance_from_current_focus) × uncertainty
   ```

4. **Pragmatic Value**:
   ```
   pragmatic = relevance × expected_reward
   ```

**Inhibition of Return**:
```
inhibition(target, t) = decay^(t - t_last_attended)
```

### Autonomous Response System

**Action Types**:
- `EXPLORE`: Information-seeking, epistemic foraging
- `EXPLOIT`: Goal-directed, utility maximization
- `REFINE`: Improve belief precision
- `ATTEND`: Shift attention focus
- `ADAPT`: Update internal model
- `COMMUNICATE`: Express current state

**Action Selection Algorithm**:

1. Generate candidate actions based on current state
2. For each candidate, compute expected free energy
3. Select action with minimum expected free energy
4. Execute and record outcome

**Policy Generation**:

For planning horizon `h`, generate `n` policies:
1. For each time step 1..h:
   - Select action based on simulated state
   - Simulate next state
2. Compute total expected free energy for policy
3. Select policy with minimum expected free energy

## Autonomous Operation

### The Autonomous Cycle

```python
def autonomous_cycle(self):
    # 1. SENSE
    observation = self.sense()  # Can be self-generated
    
    # 2. INFER
    inference_result = self.infer(observation)
    
    # 3. AWARE
    self.update_awareness(inference_result)
    awareness_state = self.awareness_system.get_awareness_state()
    
    # 4. ATTEND
    attention_targets = self.allocate_attention(inference_result)
    
    # 5. ACT
    action_result = self.select_and_act(inference_result, awareness_state)
    
    return cycle_results
```

### Key Innovation: Self-Generated Observations

The system can generate internal observations when external input is unavailable:

```python
def _generate_internal_observation(self):
    # Use current beliefs to generate observation
    base = self.inference_engine.belief.mean
    obs = pad_or_truncate(base, self.obs_dim)
    obs += noise()
    return obs
```

This allows truly autonomous operation without external prompting.

### Continuous Learning

The system learns continuously through:
1. **Belief Updating**: Gradient descent on free energy
2. **Precision Learning**: Meta-learning confidence
3. **Model Learning**: Updating generative model parameters
4. **Action Learning**: Tracking action outcomes

## Advanced Usage

### Custom Generative Models

```python
from post_transformer.core.active_inference import GenerativeModel

# Create custom generative model
custom_model = GenerativeModel(
    state_dim=64,
    obs_dim=128,
    transition_matrix=custom_transition,
    observation_matrix=custom_observation
)

engine = ActiveInferenceEngine(64, 128)
engine.generative_model = custom_model
```

### Monitoring and Introspection

```python
# Access internal metrics
free_energy_history = intelligence.inference_engine.free_energy_history
prediction_errors = intelligence.inference_engine.prediction_errors
confidence_history = [s.confidence for s in intelligence.awareness_system.state_history]

# Get comprehensive state report
report = intelligence.get_state_report()
print(report)

# Access action history
actions = intelligence.response_system.action_history
```

### Exploration-Exploitation Trade-off

```python
# Pure exploration
intelligence.attention_mechanism.adjust_exploration_exploitation(epistemic_weight=1.0)
intelligence.response_system.adjust_exploration_drive(1.0)

# Balanced
intelligence.attention_mechanism.adjust_exploration_exploitation(epistemic_weight=0.5)
intelligence.response_system.adjust_exploration_drive(0.5)

# Pure exploitation
intelligence.attention_mechanism.adjust_exploration_exploitation(epistemic_weight=0.0)
intelligence.response_system.adjust_exploration_drive(0.0)
```

### Goal-Directed Behavior

```python
import numpy as np

# Define goals in state space
goal1 = np.random.randn(state_dim) * 0.5
goal2 = np.random.randn(state_dim) * 0.3

# Set goals for attention
intelligence.attention_mechanism.set_goals({
    'reach_goal1': goal1,
    'reach_goal2': goal2,
})

# Set goal importance for action selection
intelligence.response_system.set_goals({
    'reach_goal1': 0.9,  # High priority
    'reach_goal2': 0.5,  # Medium priority
})
```

### External Observation Integration

```python
# Provide external observations
external_obs = np.random.randn(obs_dim)

# Manual cycle with external input
observation = intelligence.sense(observation=external_obs)
inference_result = intelligence.infer(observation)
# ... continue cycle
```

## Performance Optimization

### Dimensionality Selection

- **State Dim**: Higher = more expressive but slower
  - Recommended: 32-128 for most applications
- **Obs Dim**: Should match input dimensionality
  - Recommended: 1-2× state dimension

### Iteration Count

- **Inference Iterations**: More = better convergence but slower
  - Recommended: 3-10 iterations
- **Planning Horizon**: Longer = better planning but exponentially slower
  - Recommended: 2-5 steps

### Attention Capacity

- **Capacity**: Number of concurrent attention targets
  - Recommended: 3-7 targets
  - Higher = more parallel processing but more complexity

## Troubleshooting

### High Free Energy

If free energy doesn't decrease:
- Increase inference iterations
- Adjust learning rate
- Check observation dimensionality

### Low Confidence

If system confidence remains low:
- Provide more informative observations
- Increase precision learning rate
- Reduce environmental novelty

### Repetitive Actions

If system repeats actions:
- Adjust inhibition of return
- Increase exploration drive
- Set explicit goals

## References

1. Friston, K. (2010). The free-energy principle: a unified brain theory?
2. Friston, K. et al. (2017). Active inference: a process theory
3. Clark, A. (2013). Whatever next? Predictive brains
4. Fleming, S. M., & Dolan, R. J. (2012). Neural basis of metacognitive ability
5. Posner, M. I., & Petersen, S. E. (1990). The attention system of the human brain

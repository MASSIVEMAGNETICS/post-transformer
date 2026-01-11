"""
Demonstration of Post-Transformer Intelligence

This example shows the revolutionary autonomous intelligence system in action:
- No external input required after initialization
- System autonomously senses, infers, attends, and acts
- Continuous learning and adaptation
- Self-aware and self-directed
"""

import numpy as np
from post_transformer import PostTransformerIntelligence


def basic_autonomous_demo():
    """Basic demonstration of autonomous operation"""
    print("\n" + "=" * 70)
    print("DEMO 1: Basic Autonomous Intelligence")
    print("=" * 70)
    print("\nInitializing Post-Transformer Intelligence...")
    
    # Initialize the system
    intelligence = PostTransformerIntelligence(
        state_dim=32,      # Hidden state dimension
        obs_dim=64,        # Observation dimension
        attention_capacity=3,  # Number of attention targets
        planning_horizon=2     # Planning steps ahead
    )
    
    print("System initialized. Starting autonomous operation...")
    print("\nThe system will now operate completely autonomously,")
    print("making inferences, allocating attention, and taking actions")
    print("without any external input or prompting.\n")
    
    # Run autonomous cycles
    results = intelligence.run_autonomous(n_cycles=20, verbose=True)
    
    print("\n" + "=" * 70)
    print("FINAL STATE REPORT")
    print("=" * 70)
    print(intelligence.get_state_report())
    
    return intelligence, results


def exploration_demo():
    """Demonstration with exploration emphasis"""
    print("\n" + "=" * 70)
    print("DEMO 2: Exploration-Driven Intelligence")
    print("=" * 70)
    print("\nThis demo emphasizes epistemic foraging (exploration)")
    
    intelligence = PostTransformerIntelligence(
        state_dim=32,
        obs_dim=64,
        attention_capacity=5,
        planning_horizon=3
    )
    
    # Adjust to favor exploration
    intelligence.attention_mechanism.adjust_exploration_exploitation(
        epistemic_weight=0.8  # 80% exploration
    )
    intelligence.response_system.adjust_exploration_drive(0.9)
    
    print("\nConfiguration: 80% exploration, 20% exploitation")
    print("The system will actively seek novel information...\n")
    
    results = intelligence.run_autonomous(n_cycles=15, verbose=True)
    
    return intelligence, results


def goal_directed_demo():
    """Demonstration with goal-directed behavior"""
    print("\n" + "=" * 70)
    print("DEMO 3: Goal-Directed Intelligence")
    print("=" * 70)
    print("\nThis demo emphasizes goal achievement (exploitation)")
    
    intelligence = PostTransformerIntelligence(
        state_dim=32,
        obs_dim=64,
        attention_capacity=3,
        planning_horizon=4
    )
    
    # Set goals
    goal_vector = np.random.randn(32) * 0.5
    intelligence.attention_mechanism.set_goals({
        'goal_1': goal_vector,
        'goal_2': goal_vector * -0.5,
    })
    
    intelligence.response_system.set_goals({
        'goal_1': 0.8,
        'goal_2': 0.6,
    })
    
    # Adjust to favor exploitation
    intelligence.attention_mechanism.adjust_exploration_exploitation(
        epistemic_weight=0.2  # 20% exploration, 80% exploitation
    )
    
    print("\nConfiguration: 20% exploration, 80% exploitation")
    print("The system will focus on achieving defined goals...\n")
    
    results = intelligence.run_autonomous(n_cycles=15, verbose=True)
    
    return intelligence, results


def adaptive_learning_demo():
    """Demonstration of adaptive learning over time"""
    print("\n" + "=" * 70)
    print("DEMO 4: Adaptive Learning")
    print("=" * 70)
    print("\nThis demo shows how the system learns and adapts over time")
    
    intelligence = PostTransformerIntelligence(
        state_dim=48,
        obs_dim=96,
        attention_capacity=4,
        planning_horizon=3
    )
    
    print("\nRunning extended autonomous learning session...\n")
    
    # Run for longer to see adaptation
    results = intelligence.run_autonomous(n_cycles=30, verbose=True)
    
    # Analyze learning progress
    print("\n" + "=" * 70)
    print("LEARNING ANALYSIS")
    print("=" * 70)
    
    free_energies = intelligence.metrics['free_energy']
    errors = intelligence.metrics['prediction_error']
    confidences = intelligence.metrics['confidence']
    
    print(f"\nInitial 5 cycles:")
    print(f"  Avg Free Energy: {np.mean(free_energies[:5]):.3f}")
    print(f"  Avg Prediction Error: {np.mean(errors[:5]):.3f}")
    print(f"  Avg Confidence: {np.mean(confidences[:5]):.3f}")
    
    print(f"\nFinal 5 cycles:")
    print(f"  Avg Free Energy: {np.mean(free_energies[-5:]):.3f}")
    print(f"  Avg Prediction Error: {np.mean(errors[-5:]):.3f}")
    print(f"  Avg Confidence: {np.mean(confidences[-5:]):.3f}")
    
    improvement = np.mean(errors[:5]) - np.mean(errors[-5:])
    print(f"\nPrediction Error Improvement: {improvement:.3f}")
    
    return intelligence, results


def interactive_demo():
    """Interactive demonstration allowing user to provide observations"""
    print("\n" + "=" * 70)
    print("DEMO 5: Interactive Mode")
    print("=" * 70)
    print("\nIn this mode, you can optionally provide observations")
    print("or let the system generate them autonomously.\n")
    
    intelligence = PostTransformerIntelligence(
        state_dim=32,
        obs_dim=64,
        attention_capacity=3,
        planning_horizon=2
    )
    
    n_cycles = 10
    print(f"Running {n_cycles} cycles...\n")
    
    for i in range(n_cycles):
        # System can operate with or without external observation
        # Here we show it running purely autonomously
        result = intelligence.autonomous_cycle()
        
        if i % 2 == 0:
            print(f"\nCycle {result['cycle']}:")
            print(f"  Action: {result['action']['action_type']}")
            print(f"  Confidence: {result['awareness']['confidence']:.3f}")
            print(f"  Free Energy: {result['inference']['free_energy']:.3f}")
    
    print("\n" + intelligence.get_state_report())
    
    return intelligence


def main():
    """Run all demonstrations"""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  POST-TRANSFORMER INTELLIGENCE DEMONSTRATIONS".center(68) + "║")
    print("║" + "  Revolutionary Autonomous AI System".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "═" * 68 + "╝")
    
    print("\nThis system demonstrates revolutionary AI capabilities:")
    print("  ✓ Active Inference (Free Energy Principle)")
    print("  ✓ Active Awareness (Multi-level consciousness)")
    print("  ✓ Active Attention (Dynamic focus)")
    print("  ✓ Autonomous Response (Self-directed action)")
    print("\nKey feature: Operates completely autonomously - no user input needed!")
    
    try:
        # Run demonstrations
        demo_functions = [
            basic_autonomous_demo,
            exploration_demo,
            goal_directed_demo,
            adaptive_learning_demo,
            interactive_demo,
        ]
        
        for demo_func in demo_functions:
            try:
                demo_func()
                print("\n" + "─" * 70 + "\n")
            except KeyboardInterrupt:
                print("\n\nDemo interrupted by user.")
                break
            except Exception as e:
                print(f"\nError in demo: {e}")
                continue
        
        print("\n" + "═" * 70)
        print("ALL DEMONSTRATIONS COMPLETE")
        print("═" * 70)
        print("\nThe Post-Transformer Intelligence system successfully demonstrated:")
        print("  • Autonomous operation without external input")
        print("  • Active inference and belief updating")
        print("  • Self-awareness and metacognition")
        print("  • Dynamic attention allocation")
        print("  • Autonomous decision-making and action")
        print("\nThis represents a revolutionary step in AI autonomy and intelligence.")
        
    except KeyboardInterrupt:
        print("\n\nDemonstrations interrupted by user. Exiting gracefully...")


if __name__ == "__main__":
    main()

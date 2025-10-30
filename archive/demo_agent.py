"""Quick demo of agentic system capabilities."""
from workflow_agent import AgenticWorkflowSystem

print("🎬 AGENTIC SYSTEM DEMO\n")
print("=" * 70)

# Initialize agent
agent = AgenticWorkflowSystem()
print("✅ Agent initialized\n")

# Show current knowledge
insights = agent.learner.get_insights()
print("📊 Current Knowledge Base:")
print(f"  Total executions: {insights['total_executions']}")
print(f"  Workflow types learned: {', '.join(insights['workflow_types']) if insights['workflow_types'] else 'None yet'}")
print(f"  Recommendations: {len(insights['recommendations'])}")

if insights['recommendations']:
    print("\n  Active Recommendations:")
    for rec in insights['recommendations']:
        print(f"    - {rec['type']}: {rec['suggestion']}")

# Show performance trend
if insights['workflow_types']:
    for wf_type in insights['workflow_types']:
        trend = agent.optimizer.analyze_performance_trend(wf_type)
        print(f"\n📈 Performance Trend ({wf_type}):")
        print(f"  Status: {trend.get('trend', 'unknown')}")
        print(f"  Average FPS: {trend.get('avg_fps', 0):.1f}")
        print(f"  Range: {trend.get('min_fps', 0):.1f} - {trend.get('max_fps', 0):.1f} FPS")
        
        # Show suggested parameters
        params = agent.optimizer.suggest_parameters(wf_type, 20.0)
        print(f"\n💡 Suggested parameters for 20 FPS target:")
        for key, value in params.items():
            print(f"    {key}: {value}")

print("\n" + "=" * 70)
print("\n✨ System learns and improves with each execution!")
print("\n🚀 Next steps:")
print('  1. Run: uv run python agentic_integration.py "Detect objects in video"')
print("  2. Execute: uv run python agentic_integration.py workflows/granular_video_detection_mp4.json")
print("  3. Watch the system learn and optimize automatically!")

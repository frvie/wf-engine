"""Test fully atomic workflow generation."""
from workflow_agent import AgenticWorkflowSystem, WorkflowGoal
import json
from pathlib import Path

system = AgenticWorkflowSystem()

# Create goal with quality_over_speed=True to trigger fully atomic mode
goal = WorkflowGoal(
    task='object_detection',
    input_type='videos/sample.mp4',
    output_type='display',
    performance_target=20.0,
    hardware_preference='auto',
    quality_over_speed=True  # This triggers fully atomic mode
)

print("Creating fully atomic workflow...")
print(f"Goal: quality_over_speed={goal.quality_over_speed}")

wf = system.create_workflow_from_goal(goal)

print(f"\nStrategy: {wf['workflow']['strategy']}")
print(f"Nodes: {len(wf['nodes'])}")
print("\nNode IDs:")
for n in wf['nodes']:
    print(f"  - {n['id']}: {n['function'].split('.')[-1]}")

# Save
output_path = Path('workflows/demo_fully_atomic.json')
with open(output_path, 'w') as f:
    json.dump(wf, f, indent=2)

print(f"\n✅ Saved to: {output_path}")

# Compare with regular mode
print("\n" + "="*60)
print("Comparison: Regular vs Fully Atomic")
print("="*60)

goal_regular = WorkflowGoal(
    task='object_detection',
    input_type='videos/sample.mp4',
    output_type='display',
    performance_target=20.0,
    hardware_preference='auto',
    quality_over_speed=False  # Regular mode
)

wf_regular = system.create_workflow_from_goal(goal_regular)

print(f"\nRegular Mode (quality_over_speed=False):")
print(f"  Strategy: {wf_regular['workflow']['strategy']}")
print(f"  Nodes: {len(wf_regular['nodes'])}")
print("  Node IDs:")
for n in wf_regular['nodes']:
    print(f"    - {n['id']}")

print(f"\nFully Atomic Mode (quality_over_speed=True):")
print(f"  Strategy: {wf['workflow']['strategy']}")
print(f"  Nodes: {len(wf['nodes'])}")
print("  Node IDs:")
for n in wf['nodes']:
    print(f"    - {n['id']}")

"""
Test script for Workflow Builder

This script tests the workflow builder's ability to generate
workflows from natural language descriptions.
"""

import asyncio
import sys
from workflow_builder import build_workflow_interactive


async def test_workflows():
    """Test workflow generation with various descriptions."""
    
    test_cases = [
        "Create a simple CPU inference workflow for YOLOv8",
        "Build a DirectML GPU inference workflow",
        "Compare CPU and DirectML performance",
    ]
    
    print("🧪 Testing Workflow Builder")
    print("=" * 60)
    print(f"Running {len(test_cases)} test cases...\n")
    
    for i, description in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}/{len(test_cases)}")
        print(f"{'='*60}")
        
        try:
            await build_workflow_interactive(description)
            print(f"✅ Test {i} completed")
        except Exception as e:
            print(f"❌ Test {i} failed: {e}")
        
        # Small delay between tests
        if i < len(test_cases):
            print("\nWaiting 2 seconds before next test...")
            await asyncio.sleep(2)
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")


async def interactive_mode():
    """Interactive mode for custom workflow generation."""
    
    print("🏗️  Interactive Workflow Builder")
    print("=" * 60)
    print("Enter workflow descriptions (or 'quit' to exit)\n")
    
    while True:
        try:
            description = input("\n📝 Describe your workflow: ").strip()
            
            if description.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if not description:
                print("⚠️  Please provide a description")
                continue
            
            await build_workflow_interactive(description)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


async def main():
    """Main entry point."""
    
    if len(sys.argv) > 1:
        # Single workflow from command line
        description = " ".join(sys.argv[1:])
        await build_workflow_interactive(description)
    else:
        # Show menu
        print("🏗️  Workflow Builder Test Suite")
        print("=" * 60)
        print("\nOptions:")
        print("1. Run automated tests")
        print("2. Interactive mode")
        print("3. Exit")
        
        try:
            choice = input("\nSelect option (1-3): ").strip()
            
            if choice == "1":
                await test_workflows()
            elif choice == "2":
                await interactive_mode()
            elif choice == "3":
                print("👋 Goodbye!")
            else:
                print("⚠️  Invalid option")
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")

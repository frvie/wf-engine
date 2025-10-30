"""
Demo: Workflow Node Generator

Demonstrates automatic node generation using LLM.
"""

import asyncio
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from workflow_nodes.generator.node_generator import NodeGenerator, NodeSpec


async def demo_basic_generation():
    """Demo: Generate a simple image processing node."""
    print("\n" + "="*80)
    print("DEMO 1: Generate Simple Image Processing Node")
    print("="*80)
    
    generator = NodeGenerator(model_name="qwen2.5-coder:7b")
    
    spec = NodeSpec(
        goal="Apply Gaussian blur to an image",
        inputs=["image", "kernel_size"],
        outputs=["blurred_image"],
        category="custom",
        description="Applies Gaussian blur filter to smooth an image"
    )
    
    result = await generator.generate_node(spec)
    
    if result['success']:
        print(f"\n✅ Generated node: {result['node_name']}")
        print(f"   Location: {result['file_path']}")
        print(f"\n📄 Generated Code:")
        print("-" * 80)
        print(result['code'])
    else:
        print(f"\n❌ Failed: {result['error']}")


async def demo_edge_detection():
    """Demo: Generate edge detection node."""
    print("\n" + "="*80)
    print("DEMO 2: Generate Edge Detection Node")
    print("="*80)
    
    generator = NodeGenerator(model_name="qwen2.5-coder:7b")
    
    spec = NodeSpec(
        goal="Detect edges in image using Canny edge detector",
        inputs=["image"],
        outputs=["edges"],
        category="atomic",
        description="Atomic node for Canny edge detection",
        constraints=[
            "Convert to grayscale if needed",
            "Use cv2.Canny",
            "Return binary edge map"
        ]
    )
    
    result = await generator.generate_node(spec)
    
    if result['success']:
        print(f"\n✅ Generated node: {result['node_name']}")
        print(f"   Location: {result['file_path']}")
    else:
        print(f"\n❌ Failed: {result['error']}")


async def demo_custom_filter():
    """Demo: Generate custom filter node."""
    print("\n" + "="*80)
    print("DEMO 3: Generate Custom Image Filter")
    print("="*80)
    
    generator = NodeGenerator(model_name="qwen2.5-coder:7b")
    
    spec = NodeSpec(
        goal="Apply custom artistic filter combining blur and edge detection",
        inputs=["image"],
        outputs=["artistic_image"],
        category="custom",
        description="Creates artistic effect by blending blurred image with detected edges",
        constraints=[
            "First blur the image",
            "Then detect edges",
            "Combine using weighted sum",
            "Return RGB image"
        ],
        examples=[
            {
                "input": "RGB image (640x480)",
                "output": "Artistic filtered RGB image (640x480)"
            }
        ]
    )
    
    result = await generator.generate_node(spec)
    
    if result['success']:
        print(f"\n✅ Generated node: {result['node_name']}")
        print(f"   Location: {result['file_path']}")
        print(f"\n📄 Generated Code:")
        print("-" * 80)
        print(result['code'])
    else:
        print(f"\n❌ Failed: {result['error']}")


async def demo_list_generated():
    """Demo: List all generated custom nodes."""
    print("\n" + "="*80)
    print("DEMO 4: List Generated Custom Nodes")
    print("="*80)
    
    generator = NodeGenerator()
    nodes = generator.list_generated_nodes()
    
    if nodes:
        print(f"\nFound {len(nodes)} generated nodes:")
        for node_path in nodes:
            print(f"  • {node_path.name}")
    else:
        print("\nNo custom nodes generated yet.")


async def main():
    """Run all demos."""
    print("\n🎨 WORKFLOW NODE GENERATOR DEMO")
    print("=" * 80)
    print("\nThis demo shows how to automatically generate workflow nodes")
    print("using LLM-powered code generation with Ollama.")
    print("\nPrerequisites:")
    print("  • Ollama running locally (http://localhost:11434)")
    print("  • qwen2.5-coder:7b model installed")
    print("\n" + "=" * 80)
    
    try:
        # Run demos
        await demo_basic_generation()
        await asyncio.sleep(2)
        
        await demo_edge_detection()
        await asyncio.sleep(2)
        
        await demo_custom_filter()
        await asyncio.sleep(2)
        
        await demo_list_generated()
        
        print("\n" + "="*80)
        print("✅ DEMO COMPLETE")
        print("="*80)
        print("\nGenerated nodes are immediately available!")
        print("Run 'python wf.py nodes' to see all available nodes.")
        print("\nYou can now use these nodes in your workflows!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    asyncio.run(main())

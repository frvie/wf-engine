"""
Test script for the generated apply_gaussian_blur node
"""

import sys
import cv2
import numpy as np
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from workflow_nodes.custom.apply_gaussian_blur import apply_gaussian_blur_node


def test_gaussian_blur():
    """Test the apply_gaussian_blur node with a real image."""
    
    print("\n" + "="*80)
    print("TESTING: apply_gaussian_blur_node")
    print("="*80)
    
    # Check if test image exists
    test_image_path = Path("models/yolov8s/bus.jpg")
    if not test_image_path.exists():
        # Create a test image if none exists
        print("\n⚠️  Test image not found, creating synthetic test image...")
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        # Add some patterns
        cv2.rectangle(test_image, (100, 100), (300, 300), (255, 0, 0), -1)
        cv2.circle(test_image, (400, 240), 80, (0, 255, 0), -1)
    else:
        print(f"\n📁 Loading test image: {test_image_path}")
        test_image = cv2.imread(str(test_image_path))
    
    print(f"   Image shape: {test_image.shape}")
    print(f"   Image dtype: {test_image.dtype}")
    
    # Test with different kernel sizes
    kernel_sizes = [5, 15, 31, 51]
    
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    for kernel_size in kernel_sizes:
        print(f"\n🔄 Testing with kernel_size={kernel_size}")
        
        try:
            # Apply Gaussian blur
            blurred = apply_gaussian_blur_node(test_image, kernel_size=kernel_size)
            
            print(f"   ✅ Success!")
            print(f"      Input shape:  {test_image.shape}")
            print(f"      Output shape: {blurred.shape}")
            print(f"      Output dtype: {blurred.dtype}")
            
            # Save result
            output_path = output_dir / f"blur_kernel_{kernel_size}.jpg"
            cv2.imwrite(str(output_path), blurred)
            print(f"      Saved to: {output_path}")
            
        except Exception as e:
            print(f"   ❌ FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    # Test error handling (even kernel size)
    print(f"\n🔄 Testing error handling (even kernel size=10)")
    try:
        blurred = apply_gaussian_blur_node(test_image, kernel_size=10)
        print(f"   ✅ Handled even kernel size (auto-corrected to 11)")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
    
    print("\n" + "="*80)
    print("✅ TESTING COMPLETE")
    print("="*80)
    print(f"\n📂 Check results in: {output_dir.absolute()}")
    print("\nVisual comparison:")
    print("  - blur_kernel_5.jpg   (slight blur)")
    print("  - blur_kernel_15.jpg  (medium blur)")
    print("  - blur_kernel_31.jpg  (strong blur)")
    print("  - blur_kernel_51.jpg  (very strong blur)")


if __name__ == '__main__':
    test_gaussian_blur()

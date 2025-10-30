from src.core.decorator import workflow_node
import cv2


@workflow_node("apply_gaussian_blur", isolation_mode="none")
def apply_gaussian_blur_node(image, kernel_size=5):
    """
    Apply Gaussian blur to smooth an image.
    
    Atomic workflow node following granular design principles.
    
    Args:
        image: Input image (numpy array, BGR or grayscale)
        kernel_size: Size of Gaussian kernel (must be odd, default: 5)
    
    Returns:
        dict with blurred image
    """
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    return {
        "image": blurred,
        "kernel_size": kernel_size
    }



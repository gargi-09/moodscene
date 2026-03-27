# depth.py
# MoodScene - Monocular Depth Estimation Module
# Uses Depth Anything V2 (small variant) for CPU-friendly inference
# Output: normalized depth map as numpy array (H x W), float32, range [0, 1]

import cv2
import numpy as np
from PIL import Image
from transformers import pipeline

# ------------------------------------------------------------
# Load the depth estimation pipeline
# Using depth-anything-v2-small-hf for CPU compatibility
# ------------------------------------------------------------
def load_depth_model():
    """
    Loads the Depth Anything V2 small model via HuggingFace transformers.
    Returns a depth estimation pipeline object.
    Call once at startup and reuse — model loading is expensive.
    """
    print("Loading Depth Anything V2 model...")
    depth_pipeline = pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
    )
    print("Model loaded successfully.")
    return depth_pipeline


# ------------------------------------------------------------
# Run depth estimation on an input image
# ------------------------------------------------------------
def estimate_depth(image_path, depth_pipeline):
    """
    Estimates per-pixel depth from a single RGB image.

    Args:
        image_path (str): Path to input image file
        depth_pipeline: HuggingFace depth estimation pipeline

    Returns:
        depth_map (np.ndarray): Normalized depth map, shape (H, W),
                                float32, values in [0, 1]
                                0 = closest to camera, 1 = furthest
        original_image (np.ndarray): Original image in BGR (OpenCV format)
    """
    # Load image via PIL (required by HuggingFace pipeline)
    pil_image = Image.open(image_path).convert("RGB")

    # Run depth estimation
    result = depth_pipeline(pil_image)
    depth_raw = np.array(result["depth"], dtype=np.float32)

    # Normalize depth to [0, 1]
    depth_min = depth_raw.min()
    depth_max = depth_raw.max()
    if depth_max - depth_min > 0:
        depth_normalized = (depth_raw - depth_min) / (depth_max - depth_min)
    else:
        depth_normalized = np.zeros_like(depth_raw)

    # Load original image in OpenCV BGR format for downstream processing
    original_image = cv2.imread(image_path)

    return depth_normalized, original_image


# ------------------------------------------------------------
# Segment depth map into N layers (near / mid / far)
# ------------------------------------------------------------
def segment_depth_layers(depth_map, n_layers=3):
    """
    Divides the depth map into discrete layers using equal-range binning.
    These layers are used downstream to apply different mood treatments
    to foreground, midground, and background independently.

    Args:
        depth_map (np.ndarray): Normalized depth map, values in [0, 1]
        n_layers (int): Number of depth layers (default: 3)

    Returns:
        layer_masks (list of np.ndarray): Binary masks for each layer,
                                          ordered near → far
                                          Each mask shape: (H, W), bool
    """
    boundaries = np.linspace(0, 1, n_layers + 1)
    layer_masks = []

    for i in range(n_layers):
        low = boundaries[i]
        high = boundaries[i + 1]
        # Include upper boundary only on the last layer to avoid gaps
        if i < n_layers - 1:
            mask = (depth_map >= low) & (depth_map < high)
        else:
            mask = (depth_map >= low) & (depth_map <= high)
        layer_masks.append(mask)

    return layer_masks


# ------------------------------------------------------------
# Visualize depth map (useful for debugging and report figures)
# ------------------------------------------------------------
def visualize_depth(depth_map, save_path=None):
    """
    Converts a normalized depth map to a colorized heatmap for visualization.
    Uses COLORMAP_INFERNO: dark = near, bright = far.

    Args:
        depth_map (np.ndarray): Normalized depth map, values in [0, 1]
        save_path (str, optional): If provided, saves the visualization to disk

    Returns:
        depth_colored (np.ndarray): BGR colorized depth map, shape (H, W, 3)
    """
    depth_uint8 = (depth_map * 255).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)

    if save_path:
        cv2.imwrite(save_path, depth_colored)
        print(f"Depth visualization saved to {save_path}")

    return depth_colored


# ------------------------------------------------------------
# Quick test — run this file directly to verify setup
# ------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python depth.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]

    # Load model
    model = load_depth_model()

    # Estimate depth
    depth_map, original = estimate_depth(image_path, model)
    print(f"Depth map shape: {depth_map.shape}")
    print(f"Depth range: [{depth_map.min():.3f}, {depth_map.max():.3f}]")

    # Segment into layers
    masks = segment_depth_layers(depth_map, n_layers=3)
    for i, mask in enumerate(masks):
        layer_names = ["near", "mid", "far"]
        print(f"Layer '{layer_names[i]}': {mask.sum()} pixels "
              f"({100 * mask.mean():.1f}% of image)")

    # Save depth visualization
    visualize_depth(depth_map, save_path="depth_output.png")
    print("Done. Check depth_output.png.")
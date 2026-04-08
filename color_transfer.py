# color_transfer.py
# MoodScene - Reinhard LAB Color Transfer Module
# Based on: Reinhard et al. (2001) "Color Transfer between Images"
# IEEE Computer Graphics and Applications, 21(5), 34-41.
#
# Core idea: match the per-channel mean and std of a target palette
# to the source image in LAB color space. LAB is used because its
# channels are more perceptually decorrelated than RGB, making
# independent per-channel transfer more visually natural.

import cv2
import numpy as np


# ------------------------------------------------------------
# Core Reinhard color transfer
# ------------------------------------------------------------

def reinhard_transfer(source_bgr, target_bgr):
    """
    Transfers the color statistics (mean + std) of target onto source
    using Reinhard et al. (2001) method in LAB color space.

    Args:
        source_bgr (np.ndarray): Source image in BGR, uint8, shape (H, W, 3)
        target_bgr (np.ndarray): Target/reference image in BGR, uint8, shape (H, W, 3)
                                 This defines the color mood to transfer.

    Returns:
        result_bgr (np.ndarray): Color-transferred image in BGR, uint8, shape (H, W, 3)
    """
    # Convert both images to LAB float32
    source_lab = _bgr_to_lab(source_bgr)
    target_lab = _bgr_to_lab(target_bgr)

    # Split into channels
    src_L, src_A, src_B = cv2.split(source_lab)
    tgt_L, tgt_A, tgt_B = cv2.split(target_lab)

    # Transfer each channel independently
    result_L = _transfer_channel(src_L, tgt_L)
    result_A = _transfer_channel(src_A, tgt_A)
    result_B = _transfer_channel(src_B, tgt_B)

    # Merge and convert back to BGR
    result_lab = cv2.merge([result_L, result_A, result_B])
    result_bgr = _lab_to_bgr(result_lab)

    return result_bgr


def _transfer_channel(src_channel, tgt_channel):
    """
    Matches the mean and std of src_channel to tgt_channel.
    This is the core statistical transfer from Reinhard et al.

    Formula: result = (src - mean_src) * (std_tgt / std_src) + mean_tgt

    Args:
        src_channel (np.ndarray): Source channel, float32
        tgt_channel (np.ndarray): Target channel, float32

    Returns:
        transferred (np.ndarray): Adjusted channel, float32, same shape as src
    """
    src_mean, src_std = src_channel.mean(), src_channel.std()
    tgt_mean, tgt_std = tgt_channel.mean(), tgt_channel.std()

    # Avoid division by zero for flat channels
    if src_std < 1e-6:
        return src_channel.copy()

    transferred = (src_channel - src_mean) * (tgt_std / src_std) + tgt_mean
    return transferred


# ------------------------------------------------------------
# Depth-layer aware color transfer
# Applies transfer with varying intensity per depth layer
# ------------------------------------------------------------

def layered_color_transfer(source_bgr, target_bgr, layer_masks, layer_strengths):
    """
    Applies Reinhard color transfer to each depth layer independently,
    with a per-layer blend strength. Supports both binary (bool) and
    feathered (float32) masks — feathered masks blend smoothly across
    layer boundaries, eliminating hard edge artifacts.

    Args:
        source_bgr      (np.ndarray): Original image, BGR uint8, shape (H, W, 3)
        target_bgr      (np.ndarray): Mood reference palette, BGR uint8
        layer_masks     (list):       List of masks [near, mid, far], each (H, W)
                                      Either bool (binary) or float32 (feathered)
        layer_strengths (list):       Blend strengths per layer, values in [0, 1]

    Returns:
        result_bgr (np.ndarray): Layered color-transferred image, BGR uint8
    """
    # Run full transfer once — then blend per layer
    transferred_bgr = reinhard_transfer(source_bgr, target_bgr)

    result = source_bgr.copy().astype(np.float32)
    source_f = source_bgr.astype(np.float32)
    transferred_f = transferred_bgr.astype(np.float32)

    for mask, strength in zip(layer_masks, layer_strengths):
        # Normalize mask to float32 [0, 1] regardless of input type
        mask_f = mask.astype(np.float32)
        if mask_f.max() > 1.0:
            mask_f = mask_f / mask_f.max()

        # Expand to 3 channels for broadcasting
        mask_3d = mask_f[..., np.newaxis]           # (H, W, 1)

        # Per-pixel blend: strength controls how much transfer is applied
        # mask_3d controls where (soft boundary from feathering)
        blended = strength * transferred_f + (1 - strength) * source_f

        # Accumulate: soft mask weight determines contribution per pixel
        result = result * (1 - mask_3d) + blended * mask_3d

    return np.clip(result, 0, 255).astype(np.uint8)


# ------------------------------------------------------------
# Helper: generate a solid color reference image from RGB tuple
# Used to create mood palettes from simple color definitions
# rather than needing full reference images
# ------------------------------------------------------------

def palette_from_color(color_rgb, size=(100, 100)):
    """
    Creates a solid-color reference image for use as a transfer target.
    Useful for mood palettes defined as single representative colors.

    Args:
        color_rgb (tuple): (R, G, B) values in [0, 255]
        size (tuple): (height, width) of the output image

    Returns:
        palette_bgr (np.ndarray): Solid color image in BGR, uint8
    """
    r, g, b = color_rgb
    palette = np.full((size[0], size[1], 3), [b, g, r], dtype=np.uint8)
    return palette


def palette_from_colors(colors_rgb, size=(100, 100)):
    """
    Creates a multi-color gradient reference image from a list of RGB tuples.
    Colors are arranged as vertical stripes — gives a richer LAB distribution
    than a single solid color, leading to more nuanced transfers.

    Args:
        colors_rgb (list of tuples): List of (R, G, B) values
        size (tuple): (height, width) of output image

    Returns:
        palette_bgr (np.ndarray): Multi-color palette image in BGR, uint8
    """
    n = len(colors_rgb)
    palette = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    stripe_width = size[1] // n

    for i, (r, g, b) in enumerate(colors_rgb):
        x_start = i * stripe_width
        x_end = (i + 1) * stripe_width if i < n - 1 else size[1]
        palette[:, x_start:x_end] = [b, g, r]  # BGR

    return palette


# ------------------------------------------------------------
# Color space helpers
# ------------------------------------------------------------

def _bgr_to_lab(bgr_uint8):
    """Converts BGR uint8 image to LAB float32."""
    bgr_f = bgr_uint8.astype(np.float32) / 255.0
    lab = cv2.cvtColor(bgr_f, cv2.COLOR_BGR2Lab)
    return lab


def _lab_to_bgr(lab_float32):
    """Converts LAB float32 image back to BGR uint8, clipped to [0, 255]."""
    bgr_f = cv2.cvtColor(lab_float32, cv2.COLOR_Lab2BGR)
    bgr_f = np.clip(bgr_f, 0, 1)
    return (bgr_f * 255).astype(np.uint8)


# ------------------------------------------------------------
# Quick test
# ------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from depth import load_depth_model, estimate_depth, segment_depth_layers

    if len(sys.argv) < 2:
        print("Usage: python color_transfer.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]

    # Load depth model and estimate depth
    model = load_depth_model()
    depth_map, original_bgr = estimate_depth(image_path, model)
    masks = segment_depth_layers(depth_map, n_layers=3)

    # Test with a dreamy mood palette — soft lavender + rose + peach
    dreamy_palette = palette_from_colors([
        (180, 160, 210),  # lavender
        (210, 170, 180),  # rose
        (255, 210, 180),  # peach
    ])

    # Apply layered transfer: fg subtle, mid medium, bg full
    # layer_strengths = [0.5, 0.75, 1.0]
    layer_strengths = [0.3, 0.6, 0.9]   # even softer for foreground to preserve subject details
    result = layered_color_transfer(
        original_bgr, dreamy_palette, masks, layer_strengths
    )

    # Boost contrast slightly after color transfer to counteract washout
    result_lab = cv2.cvtColor(result, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(result_lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    result = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_Lab2BGR)
    
    cv2.imwrite("color_transfer_output_dreamy.png", result)
    print("Saved color_transfer_output_dreamy.png — dreamy mood test.")
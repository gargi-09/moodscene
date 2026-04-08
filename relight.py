# relight.py
# MoodScene - Depth-Aware Relighting Module
#
# This module handles the physically-grounded lighting simulation.
# Key ideas:
#   1. Feathered depth masks — smooth transitions between depth layers
#      to avoid hard boundaries seen in naive binary masking
#   2. Depth gradient as surface normal proxy — the gradient of the
#      depth map approximates surface orientation, letting us simulate
#      how light would fall across the scene geometry
#   3. Directional light simulation — each mood has a characteristic
#      light direction; we use the surface normal proxy to compute
#      a per-pixel lambertian shading term
#   4. Atmospheric haze — far layers get a depth-proportional fog
#      overlay, reinforcing spatial depth and mood
#
# References:
#   - Lambertian reflectance: standard computer graphics shading model
#   - Depth gradient as normal proxy: Horn & Brooks (1989),
#     "Shape from Shading", MIT Press
#   - Atmospheric perspective: Leonardo da Vinci (1490s), formalized
#     in computer graphics by Nishita et al. (1993)

import cv2
import numpy as np

def get_subject_background_masks(depth_map, threshold=0.35, blur_radius=51):
    """
    Separates subject (foreground) from background using depth threshold.
    Subject = pixels with depth below threshold (closest to camera).
    Masks are feathered for smooth blending at boundaries.

    Args:
        depth_map   (np.ndarray): Normalized depth map, float32 (H, W)
        threshold   (float): Depth cutoff — pixels below this = subject
        blur_radius (int): Feathering radius for smooth edges

    Returns:
        subject_mask (np.ndarray): Float32 (H, W), 1=subject, 0=background
        bg_mask      (np.ndarray): Float32 (H, W), 1=background, 0=subject
    """
    # Subject = near pixels
    subject = (depth_map < threshold).astype(np.float32)
    bg = (depth_map >= threshold).astype(np.float32)

    # Feather both masks
    subject = cv2.GaussianBlur(subject, (blur_radius, blur_radius), 0)
    bg = cv2.GaussianBlur(bg, (blur_radius, blur_radius), 0)

    # Renormalize
    subject = subject / (subject.max() + 1e-8)
    bg = bg / (bg.max() + 1e-8)

    return subject, bg

# ------------------------------------------------------------
# Feathered depth layer masks
# Replaces hard binary masks from depth.py with smooth transitions
# ------------------------------------------------------------

def feather_masks(layer_masks, feather_radius=31):
    """
    Applies Gaussian blur to binary depth layer masks to create
    smooth transitions at layer boundaries. This prevents the
    visible hard edges seen when blending layers with binary masks.

    Args:
        layer_masks (list of np.ndarray): Binary masks [near, mid, far],
                                          each shape (H, W), bool
        feather_radius (int): Gaussian kernel size (must be odd).
                              Larger = smoother transition zone.

    Returns:
        feathered (list of np.ndarray): Soft masks, float32, values in [0,1]
                                        same order as input [near, mid, far]
    """
    feathered = []
    for mask in layer_masks:
        # Convert bool mask to float
        mask_f = mask.astype(np.float32)
        # Gaussian blur creates soft falloff at boundaries
        soft = cv2.GaussianBlur(mask_f, (feather_radius, feather_radius), 0)
        # Renormalize to [0, 1]
        if soft.max() > 0:
            soft = soft / soft.max()
        feathered.append(soft)
    return feathered


# ------------------------------------------------------------
# Depth gradient → surface normal proxy
# ------------------------------------------------------------

def compute_surface_normals(depth_map):
    """
    Approximates per-pixel surface normals from the depth map gradient.
    Uses Sobel operators to compute dZ/dx and dZ/dy, then constructs
    a normal vector N = (-dZ/dx, -dZ/dy, 1) normalized to unit length.

    This is a classical technique from shape-from-shading literature
    (Horn & Brooks, 1989). While not geometrically precise without
    true camera intrinsics, it provides a plausible local surface
    orientation for shading simulation.

    Args:
        depth_map (np.ndarray): Normalized depth map, float32, shape (H, W)

    Returns:
        normals (np.ndarray): Per-pixel surface normals, float32,
                              shape (H, W, 3), each vector unit length
                              channels ordered as (Nx, Ny, Nz)
    """
    # Sobel gradients — rate of depth change in x and y directions
    dzdx = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=5)
    dzdy = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, ksize=5)

    # Construct normal vectors: flat surface → (0, 0, 1)
    # sloped surface tilts Nx and Ny away from zero
    Nx = -dzdx
    Ny = -dzdy
    Nz = np.ones_like(depth_map)

    # Normalize to unit length per pixel
    magnitude = np.sqrt(Nx**2 + Ny**2 + Nz**2) + 1e-8
    Nx /= magnitude
    Ny /= magnitude
    Nz /= magnitude

    normals = np.stack([Nx, Ny, Nz], axis=-1)
    return normals


# ------------------------------------------------------------
# Lambertian shading term
# ------------------------------------------------------------

def compute_shading(normals, light_direction):
    """
    Computes a per-pixel Lambertian shading term: dot(N, L).
    This gives a scalar intensity in [0, 1] for each pixel based
    on how directly the surface faces the light source.

    Lambertian model: I = max(0, dot(N, L))
    where N is the surface normal and L is the normalized light direction.

    Args:
        normals (np.ndarray): Surface normals, shape (H, W, 3), unit vectors
        light_direction (tuple): (x, y, z) light direction vector.
                                  Will be normalized internally.
                                  Convention: (0, 0, 1) = frontal light,
                                  (1, 0, 1) = light from right,
                                  (-1, -1, 1) = top-left dramatic light

    Returns:
        shading (np.ndarray): Per-pixel shading intensity, float32,
                              shape (H, W), values in [0, 1]
    """
    # Normalize light direction
    L = np.array(light_direction, dtype=np.float32)
    L = L / (np.linalg.norm(L) + 1e-8)

    # Dot product: sum over channel axis
    shading = np.sum(normals * L, axis=-1)

    # Clamp to [0, 1] — back-facing surfaces get zero light
    shading = np.clip(shading, 0, 1)

    return shading


# ------------------------------------------------------------
# Atmospheric haze overlay
# Adds depth-proportional fog to reinforce spatial recession
# ------------------------------------------------------------

def apply_atmospheric_haze(image_bgr, depth_map, haze_color_bgr,
                            haze_strength=0.4):
    """
    Blends a haze color into the image proportional to depth.
    Pixels further from the camera receive more haze, simulating
    atmospheric perspective — a depth cue used since the Renaissance
    and formalized in computer graphics by Nishita et al. (1993).

    haze_amount(pixel) = depth(pixel) * haze_strength

    Args:
        image_bgr (np.ndarray): Input image, BGR uint8, shape (H, W, 3)
        depth_map (np.ndarray): Normalized depth map, float32, shape (H, W)
        haze_color_bgr (tuple): (B, G, R) color of the haze/fog
        haze_strength (float): Maximum haze opacity at furthest depth [0, 1]

    Returns:
        hazed (np.ndarray): Image with atmospheric haze, BGR uint8
    """
    image_f = image_bgr.astype(np.float32)
    haze = np.array(haze_color_bgr, dtype=np.float32)

    # Depth-proportional blend factor per pixel
    alpha = (depth_map * haze_strength)[..., np.newaxis]  # (H, W, 1)

    hazed = (1 - alpha) * image_f + alpha * haze
    return np.clip(hazed, 0, 255).astype(np.uint8)


# ------------------------------------------------------------
# Brightness adjustment via LAB L-channel
# ------------------------------------------------------------

def adjust_brightness(image_bgr, delta, contrast_clip=2.0):
    """
    Adjusts image brightness by shifting the L channel in LAB space,
    then applies CLAHE for local contrast enhancement.
    Operating in LAB keeps color channels unaffected by brightness changes.

    Args:
        image_bgr (np.ndarray): Input image, BGR uint8
        delta (int): Brightness shift applied to L channel (-100 to +100)
                     Positive = brighter, negative = darker
        contrast_clip (float): CLAHE clipLimit. Higher = more contrast.
                               0 = no CLAHE applied.

    Returns:
        result (np.ndarray): Adjusted image, BGR uint8
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)

    # Shift L channel with clipping to valid range [0, 255]
    l = np.clip(l.astype(np.int16) + delta, 0, 255).astype(np.uint8)

    # Apply CLAHE for local contrast if requested
    if contrast_clip > 0:
        clahe = cv2.createCLAHE(clipLimit=contrast_clip,
                                 tileGridSize=(8, 8))
        l = clahe.apply(l)

    result = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_Lab2BGR)
    return result


# ------------------------------------------------------------
# Master relighting function
# Combines all of the above into a single call
# ------------------------------------------------------------

def apply_relighting(image_bgr, depth_map, mood_profile):
    """
    Applies depth-aware relighting with explicit subject protection.

    Strategy:
      - Background gets full color transfer + atmospheric haze + shading
      - Subject (foreground) gets only a very gentle brightness/contrast tweak
      - Boundary between them is feathered to avoid hard edges

    Args:
        image_bgr    (np.ndarray): Color-transferred image, BGR uint8
        depth_map    (np.ndarray): Normalized depth map, float32 (H, W)
        mood_profile (dict): From mood_profiles.get_mood()

    Returns:
        relit (np.ndarray): Final relit image, BGR uint8
    """
    # Separate subject from background
    subject_mask, bg_mask = get_subject_background_masks(depth_map)

    image_f = image_bgr.astype(np.float32)

    # --- Background treatment ---
    # Full atmospheric haze on background only
    bg_relit = apply_atmospheric_haze(
        image_bgr,
        depth_map * bg_mask,           # haze only where background is
        mood_profile["haze_color_bgr"],
        mood_profile.get("haze_strength", 0.35)
    ).astype(np.float32)

    # Surface normal shading on background only
    normals = compute_surface_normals(depth_map)
    shading = compute_shading(normals, mood_profile["light_direction"])
    shading_strength = mood_profile.get("shading_strength", 0.25)

    # Apply shading only to background regions
    bg_shading = shading * shading_strength * bg_mask
    bg_shading_3d = bg_shading[..., np.newaxis]
    bg_relit = bg_relit * (1.0 - bg_shading_3d)

    # --- Subject treatment ---
    # Only brightness/contrast — no color tinting on the face
    subject_relit = adjust_brightness(
        image_bgr,
        mood_profile["brightness_delta"] // 3,  # much gentler on subject
        contrast_clip=1.5
    ).astype(np.float32)

    # --- Composite: blend subject and background using masks ---
    subject_3d = subject_mask[..., np.newaxis]
    bg_3d = bg_mask[..., np.newaxis]

    # Normalize weights so they sum to 1 per pixel
    total = subject_3d + bg_3d + 1e-8
    result_f = (subject_relit * subject_3d + bg_relit * bg_3d) / total

    # Final brightness/contrast pass on the whole image
    result = np.clip(result_f, 0, 255).astype(np.uint8)
    result = adjust_brightness(
        result,
        mood_profile["brightness_delta"] // 2,
        mood_profile["contrast_clip"]
    )

    return result

# ------------------------------------------------------------
# Quick test
# ------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from depth import load_depth_model, estimate_depth, segment_depth_layers
    from color_transfer import layered_color_transfer
    from mood_profiles import get_palette, get_layer_strengths, get_mood

    if len(sys.argv) < 3:
        print("Usage: python relight.py <image_path> <mood>")
        print("Moods: dreamy, melancholic, tense, euphoric")
        sys.exit(1)

    image_path = sys.argv[1]
    mood_name   = sys.argv[2]

    # Load depth
    model = load_depth_model()
    depth_map, original_bgr = estimate_depth(image_path, model)
    masks = segment_depth_layers(depth_map, n_layers=3)

    # Color transfer
    palette = get_palette(mood_name)
    strengths = get_layer_strengths(mood_name)
    feathered = feather_masks(masks)
    color_result = layered_color_transfer(
        original_bgr, palette, feathered, strengths
    )

    # Relighting — add lighting params to mood profile
    profile = get_mood(mood_name)
    relit = apply_relighting(color_result, depth_map, profile)

    out_path = f"relit_{mood_name}.png"
    cv2.imwrite(out_path, relit)
    print(f"Saved {out_path}")
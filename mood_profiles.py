# mood_profiles.py
# MoodScene - Mood Profile Definitions
#
# Each mood is defined by:
#   - A color palette (list of RGB tuples) used for Reinhard color transfer
#   - Layer strengths [near, mid, far] controlling transfer intensity per depth layer
#   - A brightness adjustment applied to the L channel in LAB space
#   - A contrast multiplier applied post-transfer
#   - A description (used in report figures and future UI)
#
# Color choices are grounded in color psychology literature:
#   - Warm hues (red, orange, yellow) → arousal, tension, energy
#   - Cool hues (blue, cyan) → calm, sadness, distance
#   - Desaturated palettes → melancholy, detachment
#   - Soft pastels → dreaminess, softness, romance

import numpy as np
from color_transfer import palette_from_colors


# ------------------------------------------------------------
# Mood definitions
# ------------------------------------------------------------

MOOD_PROFILES = {

    "dreamy": {
        # Soft lavender, rose, and peach — warm pastels that feel
        # ethereal and romantic. Low foreground strength preserves
        # subject naturalness while background floats into fantasy.
        "description": "Soft, ethereal, romantic. Pastel tones with "
                       "gentle warmth in the background.",
        "colors_rgb": [
            (180, 160, 210),   # lavender
            (210, 170, 180),   # rose
            (255, 210, 180),   # peach
            (220, 200, 230),   # soft purple
        ],
        "layer_strengths": [0.3, 0.6, 0.9],   # fg subtle, bg full
        "brightness_delta": 8,                  # slight lift
        "contrast_clip": 2.0,                   # gentle CLAHE
        "light_direction":  (0.2, -0.5, 1.0),   # soft overhead diffuse
        "haze_color_bgr":   (220, 200, 230),     # lavender haze
        "haze_strength":    0.25,
        "shading_strength": 0.2,
    },

    "melancholic": {
        # Desaturated slate blues and cool greys — the palette of
        # overcast skies and quiet reflection. Higher foreground
        # strength here is intentional: melancholy envelops the subject.
        "description": "Cool, withdrawn, introspective. Desaturated blues "
                       "and greys that flatten warmth from the scene.",
        "colors_rgb": [
            (100, 120, 150),   # slate blue
            (120, 130, 145),   # cool grey
            (80,  100, 130),   # dark blue-grey
            (140, 150, 165),   # muted periwinkle
        ],
        "layer_strengths": [0.5, 0.7, 0.95],   # fg more affected than dreamy
        "brightness_delta": -10,                 # darken slightly
        "contrast_clip": 1.5,                    # softer contrast
        "light_direction":  (-0.3, 0.2, 1.0),   # backlight from above
        "haze_color_bgr":   (130, 120, 110),     # cool grey haze
        "haze_strength":    0.35,
        "shading_strength": 0.3,

    
    },

    "tense": {
        # Deep reds, burnt orange, and near-black — the palette of
        # danger, urgency, and unease. Strong across all layers
        # because tension should feel inescapable.
        "description": "High-stakes, urgent, unsettling. Deep reds and "
                       "shadows that compress the scene into unease.",
        "colors_rgb": [
            (180,  40,  30),   # deep red
            (140,  50,  20),   # burnt orange-red
            ( 60,  20,  20),   # near black-red
            (200,  80,  40),   # ember orange
        ],
        "layer_strengths": [0.55, 0.75, 1.0],  # strong throughout
        "brightness_delta": -15,                 # noticeably darker
        "contrast_clip": 3.5,                    # harsh contrast
        "light_direction":  (1.0, 0.5, 0.8),    # hard side light
        "haze_color_bgr":   (20,  10,  60),      # dark red-black haze
        "haze_strength":    0.4,
        "shading_strength": 0.45,
    },

    "euphoric": {
        "description": "Vibrant, joyful, electric. Warm golds and saturated "
                    "highlights that make the scene feel alive.",
        "colors_rgb": [
            (255, 200,  80),
            (255, 140,  80),
            ( 80, 220, 210),
            (255, 180, 100),
        ],
        "layer_strengths": [0.25, 0.6, 0.95],    # ← changed
        "brightness_delta": 15,
        "contrast_clip": 2.5,
        "light_direction":  (0.3, -0.5, 1.0),    # ← changed
        "haze_color_bgr":   (100, 180, 255),
        "haze_strength":    0.2,
        "shading_strength": 0.1,                  # ← changed
    },
}


# ------------------------------------------------------------
# Accessor functions
# ------------------------------------------------------------

def get_mood(mood_name):
    """
    Returns the full profile dict for a given mood name.

    Args:
        mood_name (str): One of 'dreamy', 'melancholic', 'tense', 'euphoric'

    Returns:
        profile (dict): Mood profile with keys: description, colors_rgb,
                        layer_strengths, brightness_delta, contrast_clip

    Raises:
        ValueError: If mood_name is not recognized
    """
    mood_name = mood_name.lower().strip()
    if mood_name not in MOOD_PROFILES:
        valid = list(MOOD_PROFILES.keys())
        raise ValueError(f"Unknown mood '{mood_name}'. Valid options: {valid}")
    return MOOD_PROFILES[mood_name]


def get_palette(mood_name):
    """
    Returns the BGR palette image for a given mood.
    Ready to pass directly into reinhard_transfer() or layered_color_transfer().

    Args:
        mood_name (str): One of 'dreamy', 'melancholic', 'tense', 'euphoric'

    Returns:
        palette_bgr (np.ndarray): Palette image in BGR uint8
    """
    profile = get_mood(mood_name)
    return palette_from_colors(profile["colors_rgb"])


def get_layer_strengths(mood_name):
    """
    Returns the [near, mid, far] blend strength list for a mood.

    Args:
        mood_name (str): Mood name

    Returns:
        strengths (list): Three floats in [0, 1]
    """
    return get_mood(mood_name)["layer_strengths"]


def list_moods():
    """Returns a list of all available mood names."""
    return list(MOOD_PROFILES.keys())


# ------------------------------------------------------------
# Quick test — preview all four mood palettes as saved images
# ------------------------------------------------------------

if __name__ == "__main__":
    import cv2

    print("Available moods:", list_moods())
    print()

    for mood_name in list_moods():
        profile = get_mood(mood_name)
        palette = get_palette(mood_name)

        # Save palette swatch
        swatch_path = f"palette_{mood_name}.png"
        # Scale up for visibility
        swatch_large = cv2.resize(palette, (400, 100),
                                  interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(swatch_path, swatch_large)

        print(f"Mood: {mood_name.upper()}")
        print(f"  Description : {profile['description']}")
        print(f"  Layer strengths: {profile['layer_strengths']}")
        print(f"  Brightness delta: {profile['brightness_delta']}")
        print(f"  Contrast clip: {profile['contrast_clip']}")
        print(f"  Palette saved: {swatch_path}")
        print()
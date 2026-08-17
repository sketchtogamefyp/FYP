"""
Master Generation Prompt Kit configuration and prompt builder utilities.
Exact semantic definitions for all 9 core genres with unified, high-polish art styles.
"""

import os

BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
LORA_NAME = "pixel_art_xl"
LORA_TRIGGER = "pixel art"
SAMPLER = "DPM++ 2M Karras"
STEPS = 30
CFG_SCALE = 7.0
SPRITE_RESOLUTION = (512, 512)
BACKGROUND_RESOLUTION = (1024, 512)

# Unified Style blocks matching exact user descriptions
STYLE_BLOCKS = {
    "racing": (
        "arcade racing game asset, sleek aerodynamic supercar, dynamic highlights, "
        "crisp vehicle silhouette, polished 2D arcade sprite, isolated on pure solid white background"
    ),
    "fighting": (
        "2D arcade fighting game sprite, Capcom and NeoGeo fighting game style, bold dark outlines, "
        "muscular martial artist combat pose, shaded folds, isolated on pure solid white background"
    ),
    "mario": (
        "colorful retro platformer sprite, iconic plumber hero, bold clean shapes, "
        "charming character details, vibrant palette, isolated on pure solid white background"
    ),
    "adventure": (
        "exploration adventure game sprite, detailed fantasy adventurer hero, compass, map, "
        "rich organic colors, clean dark outlines, isolated on pure solid white background"
    ),
    "dungeon": (
        "dark dungeon crawler combat asset, armored knight with glowing runic sword and shield, "
        "gothic dark fantasy aesthetic, stone and torchlight details, isolated on pure solid white background"
    ),
    "strategy": (
        "tactical real-time strategy game unit asset, high-tech mech commander, distinct faction colors, "
        "clean mechanical shading, isolated on pure solid white background"
    ),
    "tower_defense": (
        "tower defense turret asset, multi-barrel plasma defense cannon on sturdy pedestal, "
        "bold outlines, high visibility, isolated on pure solid white background"
    ),
    "running": (
        "fast-paced endless runner game asset, cyber parkour athlete in dynamic mid-sprint, "
        "high visibility neon sports colors, isolated on pure solid white background"
    ),
    "adventure_fighting": (
        "action-adventure RPG combat sprite, armored warrior with magical greatsword, "
        "detailed fantasy art, ready for real-time combos, isolated on pure solid white background"
    ),
    "default": (
        "polished 2D game asset, clean silhouette, sharp outlines, vibrant color palette, "
        "isolated on pure solid white background"
    ),
}

def get_style_block(genre=""):
    genre_key = (genre or "default").lower().strip()
    if genre_key in STYLE_BLOCKS:
        return STYLE_BLOCKS[genre_key]
    
    if "adventure_fight" in genre_key or "action_adv" in genre_key:
        return STYLE_BLOCKS["adventure_fighting"]
    elif "rac" in genre_key or "car" in genre_key:
        return STYLE_BLOCKS["racing"]
    elif "fight" in genre_key:
        return STYLE_BLOCKS["fighting"]
    elif "dungeon" in genre_key:
        return STYLE_BLOCKS["dungeon"]
    elif "strat" in genre_key:
        return STYLE_BLOCKS["strategy"]
    elif "tower" in genre_key:
        return STYLE_BLOCKS["tower_defense"]
    elif "run" in genre_key:
        return STYLE_BLOCKS["running"]
    elif "mario" in genre_key or "platform" in genre_key:
        return STYLE_BLOCKS["mario"]
    elif "adv" in genre_key:
        return STYLE_BLOCKS["adventure"]
    return STYLE_BLOCKS["default"]

NEGATIVE_PROMPT_BLOCK = (
    "blurry, noisy, artifacts, watermark, signature, text, ui, hud, border, frame, "
    "cropped, cut off, bad anatomy, distorted, ugly, low resolution, multiple subjects, "
    "cluttered background, floating fragments, photorealistic photo"
)

def build_structured_asset_prompt(role, subject, camera, composition, material="", lighting="studio lighting", colors="", genre="default"):
    style = get_style_block(genre)
    parts = [
        style,
        f"role: {role}",
        f"subject: {subject}",
        f"camera perspective: {camera}",
        f"composition: {composition}",
    ]
    if material:
        parts.append(f"material and texture: {material}")
    if lighting:
        parts.append(f"lighting: {lighting}")
    if colors:
        parts.append(f"colors: {colors}")
    parts.append("single complete subject centered, isolated on pure white background, full visibility, no clipping")
    return ", ".join(parts)

def build_full_prompt(prompt, genre="default"):
    style_block = get_style_block(genre)
    return f"{style_block}, {prompt}, centered on solid white background, high quality game asset"

def validate_payload(payload):
    return {
        "has_prompt": bool(payload.get("prompt")),
        "has_negative": bool(payload.get("negative_prompt")),
        "steps_valid": 10 <= payload.get("num_inference_steps", 0) <= 60,
        "cfg_valid": 3.0 <= payload.get("guidance_scale", 0.0) <= 15.0,
    }

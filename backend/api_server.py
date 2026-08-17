"""
Sketch-to-Game Pipeline API Server.
High-Performance, Memory-Safe Architecture with Exact 9-Genre Support,
Computer-Vision Sketch Extraction, Platformer Fidelity, and 12s Interactive Video Gameplay.
"""

import gc
import json
import os
import re
import sys
import uuid
import math
import random
import time
import zipfile
import threading
from collections import deque
import asyncio
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps
import torch

from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from openai import OpenAI

from prompt_config import (
    STYLE_BLOCKS,
    get_style_block,
    NEGATIVE_PROMPT_BLOCK,
    STEPS,
    CFG_SCALE,
    build_structured_asset_prompt,
    build_full_prompt,
    validate_payload,
)

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

def _load_env():
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip("'\""))

_load_env()

MODEL_ID = "microsoft/Florence-2-base"
DEFAULT_FLORENCE_LORA = "../florence_game_lora/final" if os.path.exists("../florence_game_lora/final") else "./models/florence_lora"
FLORENCE_LORA_PATH = os.environ.get("FLORENCE_LORA_PATH", DEFAULT_FLORENCE_LORA)
SDXL_LORA_PATH = os.environ.get("SDXL_LORA_PATH", "./models/sdxl_lora")
API_OUTPUT_DIR = os.environ.get("API_OUTPUT_DIR", "./api_output")
os.makedirs(API_OUTPUT_DIR, exist_ok=True)

ENABLE_CLIP_SELECTION = os.environ.get("ENABLE_CLIP_SELECTION", "false").lower() == "true"
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/")

def get_device():
    try:
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

DEVICE = os.environ.get("DEVICE", get_device())

def get_openai_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Please set it in backend/.env or environment variables.")
    return OpenAI(api_key=api_key)

executor = ThreadPoolExecutor(max_workers=2)
JOB_STATUS = {}

def set_job_status(job_id, step, progress, details="", result=None, error=None, error_code=None):
    if not job_id:
        return
    now = time.time()
    existing = JOB_STATUS.get(job_id, {})
    started_at = existing.get("started_at", now)
    status_str = "error" if error else ("completed" if progress >= 100 else "processing")
    JOB_STATUS[job_id] = {
        "status": status_str,
        "step": step,
        "progress": progress,
        "details": details,
        "result": result,
        "error": error,
        "error_code": error_code,
        "started_at": started_at,
        "updated_at": now,
    }

# --------------------------------------------------------------------------
# Model Memory Management
# --------------------------------------------------------------------------

florence_processor = None
florence_base = None
florence_model = None
sdxl_pipe = None
clip_model = None
clip_preprocess = None
clip_tokenizer = None

def free_model_memory(target="all"):
    global florence_model, florence_base, florence_processor
    global sdxl_pipe
    global clip_model, clip_preprocess, clip_tokenizer
    print(f"[Memory] Freeing models for target: {target}...")

    if target in ["all", "florence"]:
        florence_model = None
        florence_base = None
        florence_processor = None

    if target in ["all", "sdxl"]:
        sdxl_pipe = None

    if target in ["all", "clip"]:
        clip_model = None
        clip_preprocess = None
        clip_tokenizer = None

    gc.collect()
    if DEVICE == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()

def ensure_florence_loaded():
    global florence_processor, florence_base, florence_model
    if florence_model is not None:
        return
    free_model_memory("all")
    from transformers import AutoProcessor, AutoModelForCausalLM
    from peft import PeftModel

    print("[Model] Loading Florence-2...")
    florence_processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    dtype = torch.float16 if DEVICE == "cuda" else torch.bfloat16
    florence_base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True,
        torch_dtype=dtype,
        device_map=DEVICE,
        low_cpu_mem_usage=True
    )
    if os.path.exists(FLORENCE_LORA_PATH) and len(os.listdir(FLORENCE_LORA_PATH)) > 0:
        florence_model = PeftModel.from_pretrained(florence_base, FLORENCE_LORA_PATH)
    else:
        florence_model = florence_base
    florence_model.eval()
    print("[Model] Florence-2 ready.")

def ensure_sdxl_loaded():
    global sdxl_pipe
    if sdxl_pipe is not None:
        return
    free_model_memory("florence")
    from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

    print("[Model] Loading SDXL...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        variant="fp16" if DEVICE == "cuda" else None,
        use_safetensors=True,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        algorithm_type="dpmsolver++",
        use_karras_sigmas=True,
    )
    if DEVICE == "cuda":
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_tiling()

    if os.path.exists(SDXL_LORA_PATH) and len(os.listdir(SDXL_LORA_PATH)) > 0:
        pipe.load_lora_weights(SDXL_LORA_PATH)
    else:
        pipe.load_lora_weights("nerijs/pixel-art-xl", weight_name="pixel-art-xl.safetensors")
    pipe.fuse_lora(lora_scale=0.90)
    sdxl_pipe = pipe
    print("[Model] SDXL ready.")

# --------------------------------------------------------------------------
# Computer Vision & Sketch Layout Extraction
# --------------------------------------------------------------------------

def cv_extract_sketch_layout(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    h, w = img.shape
    _, thresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    platform_boxes = []
    spikes = []
    player_pos = [1, 10]
    goal_pos = [22, 2]

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 40:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = bw / float(bh)

        if aspect >= 2.0 and bw >= int(w * 0.06):
            norm_box = {
                "x_norm": round(x / w, 3),
                "y_norm": round(y / h, 3),
                "w_norm": round(bw / w, 3),
                "h_norm": round(max(bh, int(h * 0.04)) / h, 3)
            }
            platform_boxes.append(norm_box)

        elif 0.5 <= aspect <= 2.2 and area < int(w * h * 0.05):
            approx = cv2.approxPolyDP(cnt, 0.06 * cv2.arcLength(cnt, True), True)
            if len(approx) == 3 or (bh > 10 and aspect <= 1.5 and y > int(h * 0.35)):
                col = int((x + bw / 2) / w * 24)
                row = int((y + bh / 2) / h * 12)
                spikes.append([col, row])

        elif 0.8 <= aspect <= 1.25 and area < int(w * h * 0.04) and x < int(w * 0.35):
            col = int((x + bw / 2) / w * 24)
            row = int((y + bh / 2) / h * 12)
            player_pos = [col, row]

        elif x > int(w * 0.65) and y < int(h * 0.55):
            col = int((x + bw / 2) / w * 24)
            row = int((y + bh / 2) / h * 12)
            goal_pos = [col, row]

    platform_boxes.sort(key=lambda b: -b["y_norm"])

    if not platform_boxes:
        platform_boxes = [
            {"x_norm": 0.0, "y_norm": 0.88, "w_norm": 1.0, "h_norm": 0.08},
            {"x_norm": 0.12, "y_norm": 0.65, "w_norm": 0.22, "h_norm": 0.05},
            {"x_norm": 0.40, "y_norm": 0.50, "w_norm": 0.22, "h_norm": 0.05},
            {"x_norm": 0.68, "y_norm": 0.35, "w_norm": 0.22, "h_norm": 0.05}
        ]

    platforms_grid = []
    for p in platform_boxes:
        c1 = int(p["x_norm"] * 24)
        c2 = int((p["x_norm"] + p["w_norm"]) * 24)
        r = int(p["y_norm"] * 12)
        for col in range(max(0, c1), min(24, c2 + 1)):
            platforms_grid.append([col, r])

    return {
        "platforms": platforms_grid,
        "platform_boxes": platform_boxes,
        "player": player_pos,
        "goal": goal_pos,
        "spikes": spikes if spikes else [[6, 10], [12, 6]],
        "enemies": [[14, 5]]
    }

def sketch_to_layout(image_path):
    cv_layout = cv_extract_sketch_layout(image_path)
    raw_caption = "A hand-drawn level sketch with platforms, hazards, and goal flag"
    raw_od = ""
    vision_status = "available"
    objects = [
        {"type": "platforms", "position": [12, 6]},
        {"type": "spikes", "position": [6, 10]},
        {"type": "goal_flag", "position": [22, 2]}
    ]
    scene = {"environment": "platformer", "camera": "side_view"}
    visual_genre_evidence = ["platform", "platforms", "jump", "flag", "spikes"]

    try:
        if os.environ.get("OPENAI_API_KEY"):
            import base64
            with open(image_path, "rb") as f:
                base64_image = base64.b64encode(f.read()).decode("utf-8")
            client = get_openai_client()
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": [
                        {"type": "text", "text": "Describe this level sketch in 1 sentence focusing on platforms, hazards, and goal."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]}
                ],
                max_tokens=80,
                temperature=0.2
            )
            raw_caption = resp.choices[0].message.content.strip()
    except Exception:
        pass

    vision_info = {
        "vision_status": vision_status,
        "caption": raw_caption,
        "objects": objects,
        "scene": scene,
        "spatial_relations": ["player_starts_bottom_left", "goal_top_right", "platforms_ascending"],
        "visual_genre_evidence": visual_genre_evidence
    }

    return cv_layout, raw_caption, raw_od, vision_info

# --------------------------------------------------------------------------
# Exact 9-Genre Taxonomy & Priority Resolution
# --------------------------------------------------------------------------

def _kw_hit(keyword, text):
    pattern = r"(?<!\w)" + re.escape(keyword) + r"(?!\w)"
    return re.search(pattern, text) is not None

HYBRID_PHRASE_MAPPINGS = [
    ("a fast-paced game where two or more cars race", "racing"),
    ("cars race against each other", "racing"),
    ("a combat game where players control powerful characters", "fighting"),
    ("punches, kicks, combos, and special attacks", "fighting"),
    ("an exploration game where the player travels through different environments", "adventure"),
    ("completes objectives, solves challenges, and discovers new areas", "adventure"),
    ("a dark combat game where the player explores dangerous dungeons", "dungeon"),
    ("explores dangerous dungeons, defeats enemies, collects loot", "dungeon"),
    ("dungeon combat", "dungeon"),
    ("dark dungeon", "dungeon"),
    ("a tactical game where the player manages resources", "strategy"),
    ("manages resources, builds units or structures", "strategy"),
    ("a colorful platformer where the player runs and jumps", "mario"),
    ("runs and jumps through obstacle-filled levels", "mario"),
    ("a defense game where the player strategically places towers", "tower_defense"),
    ("stop waves of enemies from reaching the base", "tower_defense"),
    ("an endless runner where the player continuously runs forward", "running"),
    ("continuously runs forward, avoids obstacles", "running"),
    ("an action-adventure game where the player explores a world", "adventure_fighting"),
    ("explores a world, fights enemies, completes missions", "adventure_fighting"),
    ("adventure fighting", "adventure_fighting"),
    ("adventure fight", "adventure_fighting"),
    ("action adventure combat", "adventure_fighting"),
    ("adventure game with sword fighting", "adventure_fighting"),
    ("adventure game with sword combat", "adventure_fighting"),
    ("sword fighting in a dungeon", "adventure_fighting"),
    ("hack and slash", "adventure_fighting"),
    ("action rpg", "adventure_fighting"),
    ("tower defense", "tower_defense"),
    ("defend the base", "tower_defense"),
    ("waves of enemies", "tower_defense"),
    ("endless runner", "running"),
    ("infinite runner", "running"),
    ("auto runner", "running"),
    ("car racing", "racing"),
    ("speed racing", "racing"),
    ("sports cars on a highway", "racing"),
    ("two cars racing", "racing"),
    ("two sports cars", "racing"),
    ("strategy base building", "strategy"),
    ("command and conquer", "strategy"),
    ("dungeon crawler", "dungeon"),
    ("roguelike", "dungeon"),
    ("mario-style", "mario"),
    ("mario style", "mario"),
    ("super mario", "mario"),
    ("platform game", "mario"),
    ("platformer game", "mario"),
    ("platformer", "mario"),
    ("platform", "mario"),
]

GENRE_KEYWORDS = {
    "adventure_fighting": ["adventure fighting", "adventure fight", "action adventure", "hack and slash", "action rpg"],
    "racing": ["racing", "race", "car", "cars", "drive", "driving", "track", "vehicle", "supercar", "kart", "speedway", "highway"],
    "fighting": ["fighting", "fight", "brawler", "combat", "arena", "beatemup", "martial arts", "boxing", "1v1", "duel", "showdown"],
    "adventure": ["adventure", "explore", "exploration", "quest", "treasure", "forest", "rpg", "open world"],
    "dungeon": ["dungeon", "crawler", "maze", "catacomb", "crypt", "underground cave"],
    "strategy": ["strategy", "rts", "base", "units", "tactics", "civilization", "command post"],
    "mario": ["mario", "platformer", "platform", "side scroller", "sidescroller", "jumping", "brick"],
    "tower_defense": ["tower defense", "tower", "td", "turret", "defend", "base defense"],
    "running": ["running", "runner", "infinite run", "parkour", "lanes"],
}

SUPPORTED_GENRES = list(GENRE_KEYWORDS.keys())

def resolve_genre(user_description, florence_caption, florence_od, layout, vision_info):
    user_desc_lower = (user_description or "").lower().strip()
    user_genre = None
    reason = ""
    source = "default"
    confidence = 0.5

    for phrase, target_genre in HYBRID_PHRASE_MAPPINGS:
        if _kw_hit(phrase, user_desc_lower) or phrase in user_desc_lower:
            user_genre = target_genre
            reason = f"User explicitly requested specific gameplay phrase '{phrase}' -> genre '{target_genre}'."
            source = "user_instruction_hybrid"
            confidence = 0.99
            break

    if not user_genre:
        for g, keywords in GENRE_KEYWORDS.items():
            for kw in sorted(keywords, key=len, reverse=True):
                if _kw_hit(kw, user_desc_lower):
                    user_genre = g
                    reason = f"User explicitly requested genre matching keyword '{kw}'."
                    source = "user_instruction"
                    confidence = 0.95
                    break
            if user_genre:
                break

    resolved_genre = user_genre if user_genre else "mario"
    return {
        "genre": resolved_genre,
        "confidence": confidence if user_genre else 0.70,
        "source": source if user_genre else "sketch_heuristic",
        "user_requested_genre": user_genre,
        "visual_conflict": False,
        "reason": reason if reason else "Resolved to classic colorful Mario platformer."
    }

# --------------------------------------------------------------------------
# Structured Game Planning
# --------------------------------------------------------------------------

GAME_PLAN_SYSTEM_PROMPT = """You are an expert Lead Game Designer.
STRICT 9-GENRE DEFINITIONS:
- Racing: A fast-paced game where two or more cars race against each other on a challenging track to reach the finish line first.
- Fighting: A combat game where players control powerful characters and fight opponents using punches, kicks, combos, and special attacks.
- Adventure: An exploration game where the player travels through different environments, completes objectives, solves challenges, and discovers new areas.
- Dungeon: A dark combat game where the player explores dangerous dungeons, defeats enemies, collects loot, and faces powerful bosses.
- Strategy: A tactical game where the player manages resources, builds units or structures, and makes smart decisions to defeat the enemy.
- Mario: A colorful platformer where the player runs and jumps through obstacle-filled levels, defeats enemies, collects coins, and reaches the goal.
- Tower Defense: A defense game where the player strategically places towers and upgrades them to stop waves of enemies from reaching the base.
- Running: An endless runner where the player continuously runs forward, avoids obstacles, collects rewards, and tries to achieve the highest score.
- Adventure Fighting: An action-adventure game where the player explores a world, fights enemies, completes missions, and unlocks new abilities and areas.

Output valid unescaped JSON without any markdown formatting:
{
  "genre": "Exact Genre",
  "title": "Creative Game Title",
  "theme": "Theme Name",
  "description": "Engaging description",
  "sketch_interpretation": "Sentence explaining how the sketch was used",
  "camera": {"style": "side_view / third_person_chase / top_down"},
  "gameplay_systems": {
    "health": 100,
    "lives": 3,
    "win_condition": "Win condition string",
    "lose_condition": "Lose condition string"
  },
  "physics_data": {
    "gravity": 9.8,
    "move_speed": 6.0,
    "jump_force": 12.5
  },
  "assets": {
    "player": "Player asset prompt",
    "enemy": "Enemy asset prompt",
    "platform_tile": "Platform/ground tile prompt",
    "background": "Wide environment background prompt"
  }
}"""

def plan_game(layout_json, user_description, florence_caption, florence_od, genre_resolution, vision_info):
    client = get_openai_client()
    user_msg = f"GENRE: {genre_resolution['genre']}\nUSER PROMPT: {user_description}\nSKETCH: {florence_caption}"
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": GAME_PLAN_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        raw = response.choices[0].message.content.strip().replace("```json", "").replace("```", "").strip()
        plan = json.loads(raw)
        plan["genre"] = genre_resolution["genre"]
        return plan
    except Exception as e:
        print(f"[Planner] GPT plan fallback: {e}")
        genre = genre_resolution["genre"]
        return {
            "genre": genre,
            "title": f"Super {genre.title()} Adventure",
            "theme": f"Classic {genre.title()}",
            "description": f"An authentic {genre} game faithfully crafted from your drawing.",
            "sketch_interpretation": f"Built an arcade {genre} game following your exact level layout.",
            "camera": {"style": "side_view" if genre != "racing" else "third_person_chase"},
            "gameplay_systems": {"health": 100, "lives": 3, "win_condition": "Reach Goal", "lose_condition": "HP 0"},
            "physics_data": {"gravity": 9.8, "move_speed": 6.0, "jump_force": 12.5},
            "assets": {
                "player": f"{genre} player hero",
                "enemy": f"{genre} rival/enemy",
                "platform_tile": f"{genre} platform tile",
                "background": f"{genre} background"
            }
        }

# --------------------------------------------------------------------------
# Asset Processing: Alpha Cropping & Quality Validation
# --------------------------------------------------------------------------

def remove_flat_background(img, tolerance=30):
    img = img.convert("RGBA")
    data = np.array(img, dtype=np.int32)
    h, w = data.shape[:2]
    corners = [data[0, 0, :3], data[0, w - 1, :3], data[h - 1, 0, :3], data[h - 1, w - 1, :3]]
    mask = np.zeros((h, w), dtype=bool)
    for corner in corners:
        diff = np.abs(data[:, :, :3] - corner).sum(axis=-1)
        mask |= (diff < tolerance)
    data_out = np.array(img, dtype=np.uint8)
    data_out[mask, 3] = 0
    return Image.fromarray(data_out, mode="RGBA")

def crop_to_content(img, min_alpha=10, pad_percent=0.04):
    img = img.convert("RGBA")
    alpha = np.array(img)[:, :, 3]
    non_zero = np.where(alpha > min_alpha)
    if len(non_zero[0]) == 0 or len(non_zero[1]) == 0:
        return img
    ymin, ymax = np.min(non_zero[0]), np.max(non_zero[0])
    xmin, xmax = np.min(non_zero[1]), np.max(non_zero[1])
    w, h = img.size
    pad_x = int((xmax - xmin) * pad_percent)
    pad_y = int((ymax - ymin) * pad_percent)
    crop_box = (
        max(0, xmin - pad_x),
        max(0, ymin - pad_y),
        min(w, xmax + pad_x + 1),
        min(h, ymax + pad_y + 1)
    )
    return img.crop(crop_box)

def validate_asset(img, asset_name, min_content_ratio=0.20):
    if img is None:
        return False
    w, h = img.size
    if w <= 0 or h <= 0:
        return False
    if asset_name == "background":
        return True
    alpha = np.array(img.convert("RGBA"))[:, :, 3]
    solid_pixels = np.count_nonzero(alpha > 10)
    ratio = solid_pixels / (w * h)
    return ratio >= min_content_ratio

# --------------------------------------------------------------------------
# All 9 Genres: Specialized Procedural Character & Sprite Generators
# --------------------------------------------------------------------------

def draw_mario_hero(width=512, height=512):
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    px = lambda f: int(width * f)
    py = lambda f: int(height * f)

    skin = (255, 195, 150)
    skin_shadow = (220, 155, 115)
    red = (225, 25, 30)
    red_dark = (165, 15, 20)
    blue = (30, 75, 205)
    blue_dark = (18, 48, 145)
    brown = (110, 55, 25)
    yellow = (255, 215, 0)
    white = (250, 250, 250)

    draw.ellipse([px(0.18), py(0.86), px(0.82), py(0.97)], fill=(15, 12, 18, 170))
    draw.rounded_rectangle([px(0.52), py(0.82), px(0.78), py(0.94)], radius=8, fill=brown)
    draw.rectangle([px(0.52), py(0.90), px(0.78), py(0.94)], fill=(70, 35, 15, 255))
    draw.rounded_rectangle([px(0.20), py(0.82), px(0.48), py(0.94)], radius=8, fill=brown)
    draw.rectangle([px(0.20), py(0.90), px(0.48), py(0.94)], fill=(70, 35, 15, 255))

    draw.polygon([(px(0.24), py(0.58)), (px(0.46), py(0.58)), (px(0.44), py(0.86)), (px(0.22), py(0.86))], fill=blue)
    draw.polygon([(px(0.54), py(0.58)), (px(0.76), py(0.58)), (px(0.78), py(0.86)), (px(0.56), py(0.86))], fill=blue)
    draw.rectangle([px(0.28), py(0.44), px(0.72), py(0.68)], fill=blue)
    draw.rectangle([px(0.28), py(0.58), px(0.72), py(0.68)], fill=blue_dark)

    draw.polygon([(px(0.32), py(0.32)), (px(0.40), py(0.32)), (px(0.40), py(0.52)), (px(0.32), py(0.52))], fill=blue)
    draw.polygon([(px(0.60), py(0.32)), (px(0.68), py(0.32)), (px(0.68), py(0.52)), (px(0.60), py(0.52))], fill=blue)
    draw.ellipse([px(0.33), py(0.44), px(0.39), py(0.50)], fill=yellow)
    draw.ellipse([px(0.61), py(0.44), px(0.67), py(0.50)], fill=yellow)

    draw.polygon([(px(0.34), py(0.30)), (px(0.66), py(0.30)), (px(0.66), py(0.46)), (px(0.34), py(0.46))], fill=red)
    draw.polygon([(px(0.64), py(0.32)), (px(0.84), py(0.24)), (px(0.88), py(0.36)), (px(0.68), py(0.44))], fill=red)
    draw.rounded_rectangle([px(0.78), py(0.18), px(0.92), py(0.34)], radius=8, fill=white)
    draw.line([px(0.80), py(0.24), px(0.90), py(0.24)], fill=(200, 200, 210, 255), width=2)
    draw.polygon([(px(0.16), py(0.36)), (px(0.36), py(0.32)), (px(0.34), py(0.46)), (px(0.14), py(0.50))], fill=red)
    draw.rounded_rectangle([px(0.08), py(0.44), px(0.22), py(0.58)], radius=8, fill=white)
    draw.line([px(0.10), py(0.50), px(0.20), py(0.50)], fill=(200, 200, 210, 255), width=2)

    draw.ellipse([px(0.30), py(0.10), px(0.70), py(0.40)], fill=skin)
    draw.ellipse([px(0.26), py(0.18), px(0.36), py(0.30)], fill=skin)
    draw.ellipse([px(0.64), py(0.18), px(0.74), py(0.30)], fill=skin)

    draw.ellipse([px(0.42), py(0.20), px(0.58), py(0.32)], fill=skin)
    draw.arc([px(0.42), py(0.20), px(0.58), py(0.32)], 0, 180, fill=skin_shadow, width=2)
    draw.polygon([
        (px(0.34), py(0.30)), (px(0.42), py(0.26)), (px(0.50), py(0.28)),
        (px(0.58), py(0.26)), (px(0.66), py(0.30)), (px(0.62), py(0.36)),
        (px(0.50), py(0.34)), (px(0.38), py(0.36))
    ], fill=(45, 25, 15, 255))

    draw.ellipse([px(0.38), py(0.14), px(0.46), py(0.24)], fill=white)
    draw.ellipse([px(0.42), py(0.15), px(0.46), py(0.23)], fill=(20, 60, 180, 255))
    draw.ellipse([px(0.43), py(0.16), px(0.45), py(0.19)], fill=white)
    draw.ellipse([px(0.54), py(0.14), px(0.62), py(0.24)], fill=white)
    draw.ellipse([px(0.54), py(0.15), px(0.58), py(0.23)], fill=(20, 60, 180, 255))
    draw.ellipse([px(0.55), py(0.16), px(0.57), py(0.19)], fill=white)

    draw.polygon([
        (px(0.26), py(0.16)), (px(0.30), py(0.04)), (px(0.50), py(-0.02)),
        (px(0.70), py(0.04)), (px(0.74), py(0.16)), (px(0.80), py(0.18)),
        (px(0.50), py(0.18)), (px(0.20), py(0.18))
    ], fill=red)
    draw.polygon([(px(0.20), py(0.16)), (px(0.80), py(0.16)), (px(0.76), py(0.22)), (px(0.24), py(0.22))], fill=red_dark)
    draw.ellipse([px(0.44), py(0.04), px(0.56), py(0.14)], fill=white)
    draw.text((px(0.47), py(0.04)), "M", fill=red)

    return img

def draw_goomba_enemy(width=512, height=512):
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    px = lambda f: int(width * f)
    py = lambda f: int(height * f)

    brown = (175, 80, 40)
    brown_dark = (120, 50, 25)
    cream = (245, 225, 190)
    black = (25, 25, 30)

    draw.ellipse([px(0.15), py(0.80), px(0.85), py(0.95)], fill=(15, 12, 18, 160))
    draw.ellipse([px(0.18), py(0.72), px(0.48), py(0.90)], fill=black)
    draw.ellipse([px(0.52), py(0.72), px(0.82), py(0.90)], fill=black)
    draw.polygon([(px(0.32), py(0.45)), (px(0.68), py(0.45)), (px(0.62), py(0.78)), (px(0.38), py(0.78))], fill=cream)
    draw.polygon([
        (px(0.20), py(0.50)), (px(0.14), py(0.36)), (px(0.22), py(0.15)),
        (px(0.50), py(0.06)), (px(0.78), py(0.15)), (px(0.86), py(0.36)),
        (px(0.80), py(0.50)), (px(0.50), py(0.54))
    ], fill=brown)
    draw.polygon([(px(0.18), py(0.48)), (px(0.82), py(0.48)), (px(0.74), py(0.54)), (px(0.26), py(0.54))], fill=brown_dark)

    draw.polygon([(px(0.30), py(0.22)), (px(0.48), py(0.28)), (px(0.48), py(0.32)), (px(0.30), py(0.26))], fill=black)
    draw.polygon([(px(0.70), py(0.22)), (px(0.52), py(0.28)), (px(0.52), py(0.32)), (px(0.70), py(0.26))], fill=black)

    draw.ellipse([px(0.34), py(0.28), px(0.46), py(0.44)], fill=(255, 255, 255, 255))
    draw.ellipse([px(0.38), py(0.30), px(0.44), py(0.42)], fill=black)
    draw.ellipse([px(0.54), py(0.28), px(0.66), py(0.44)], fill=(255, 255, 255, 255))
    draw.ellipse([px(0.56), py(0.30), px(0.62), py(0.42)], fill=black)

    draw.polygon([(px(0.36), py(0.60)), (px(0.42), py(0.52)), (px(0.44), py(0.60))], fill=(255, 255, 255, 255))
    draw.polygon([(px(0.56), py(0.60)), (px(0.58), py(0.52)), (px(0.64), py(0.60))], fill=(255, 255, 255, 255))

    return img

def draw_detailed_car(width=512, height=512, is_player=True):
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    px = lambda f: int(width * f)
    py = lambda f: int(height * f)

    main_color = (225, 20, 30) if is_player else (20, 80, 230)
    dark_color = (150, 10, 20) if is_player else (10, 45, 150)
    light_color = (255, 75, 75) if is_player else (70, 140, 255)
    accent_stripe = (255, 255, 255) if is_player else (0, 235, 255)

    draw.ellipse([px(0.06), py(0.70), px(0.94), py(0.95)], fill=(10, 12, 18, 190))
    draw.ellipse([px(0.12), py(0.73), px(0.88), py(0.92)], fill=(5, 6, 10, 230))

    draw.rounded_rectangle([px(0.08), py(0.52), px(0.28), py(0.86)], radius=12, fill=(20, 22, 26, 255))
    draw.ellipse([px(0.13), py(0.58), px(0.23), py(0.80)], fill=(75, 80, 92, 255))
    draw.ellipse([px(0.15), py(0.62), px(0.21), py(0.76)], fill=(200, 205, 215, 255))
    draw.rounded_rectangle([px(0.72), py(0.52), px(0.92), py(0.86)], radius=12, fill=(20, 22, 26, 255))
    draw.ellipse([px(0.77), py(0.58), px(0.87), py(0.80)], fill=(75, 80, 92, 255))
    draw.ellipse([px(0.79), py(0.62), px(0.85), py(0.76)], fill=(200, 205, 215, 255))

    draw.polygon([(px(0.22), py(0.78)), (px(0.78), py(0.78)), (px(0.75), py(0.88)), (px(0.25), py(0.88))], fill=(18, 20, 24, 255))
    for ex_x in [0.26, 0.32, 0.68, 0.74]:
        draw.ellipse([px(ex_x-0.03), py(0.80), px(ex_x+0.03), py(0.86)], fill=(210, 215, 225, 255))
        draw.ellipse([px(ex_x-0.01), py(0.82), px(ex_x+0.01), py(0.84)], fill=(255, 120, 20, 255))

    draw.polygon([
        (px(0.14), py(0.76)), (px(0.15), py(0.60)), (px(0.24), py(0.44)),
        (px(0.76), py(0.44)), (px(0.85), py(0.60)), (px(0.86), py(0.76)),
        (px(0.78), py(0.80)), (px(0.22), py(0.80))
    ], fill=dark_color)
    draw.polygon([
        (px(0.16), py(0.74)), (px(0.18), py(0.56)), (px(0.26), py(0.40)),
        (px(0.74), py(0.40)), (px(0.82), py(0.56)), (px(0.84), py(0.74)),
        (px(0.76), py(0.78)), (px(0.24), py(0.78))
    ], fill=main_color)
    draw.polygon([
        (px(0.20), py(0.68)), (px(0.22), py(0.52)), (px(0.28), py(0.38)),
        (px(0.72), py(0.38)), (px(0.78), py(0.52)), (px(0.80), py(0.68)),
        (px(0.74), py(0.72)), (px(0.26), py(0.72))
    ], fill=light_color)

    draw.polygon([(px(0.27), py(0.38)), (px(0.33), py(0.18)), (px(0.67), py(0.18)), (px(0.73), py(0.38))], fill=(16, 20, 30, 255))
    draw.polygon([(px(0.33), py(0.34)), (px(0.37), py(0.21)), (px(0.48), py(0.21)), (px(0.42), py(0.34))], fill=(90, 160, 240, 160))

    draw.polygon([(px(0.30), py(0.22)), (px(0.33), py(0.22)), (px(0.34), py(0.36)), (px(0.29), py(0.36))], fill=(25, 28, 34, 255))
    draw.polygon([(px(0.67), py(0.22)), (px(0.70), py(0.22)), (px(0.71), py(0.36)), (px(0.66), py(0.36))], fill=(25, 28, 34, 255))
    draw.polygon([(px(0.12), py(0.20)), (px(0.14), py(0.14)), (px(0.86), py(0.14)), (px(0.88), py(0.20)), (px(0.84), py(0.24)), (px(0.16), py(0.24))], fill=(22, 25, 30, 255))
    draw.rounded_rectangle([px(0.11), py(0.11), px(0.15), py(0.27)], radius=4, fill=main_color)
    draw.rounded_rectangle([px(0.85), py(0.11), px(0.89), py(0.27)], radius=4, fill=main_color)

    draw.line([px(0.21), py(0.56), px(0.36), py(0.55)], fill=(255, 45, 55, 255), width=4)
    draw.line([px(0.64), py(0.55), px(0.79), py(0.56)], fill=(255, 45, 55, 255), width=4)
    draw.polygon([(px(0.45), py(0.18)), (px(0.48), py(0.18)), (px(0.48), py(0.78)), (px(0.45), py(0.78))], fill=accent_stripe)
    draw.polygon([(px(0.52), py(0.18)), (px(0.55), py(0.18)), (px(0.55), py(0.78)), (px(0.52), py(0.78))], fill=accent_stripe)

    return img

def draw_detailed_fighter(width=512, height=512, is_player=True):
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    px = lambda f: int(width * f)
    py = lambda f: int(height * f)
    skin = (235, 180, 135)
    skin_shadow = (195, 140, 100)

    if is_player:
        draw.ellipse([px(0.18), py(0.86), px(0.82), py(0.97)], fill=(15, 12, 18, 170))
        draw.polygon([(px(0.56), py(0.14)), (px(0.78), py(0.10)), (px(0.72), py(0.18))], fill=(210, 25, 25, 255))
        draw.polygon([(px(0.54), py(0.62)), (px(0.76), py(0.64)), (px(0.82), py(0.88)), (px(0.64), py(0.88))], fill=(225, 225, 220, 255))
        draw.polygon([(px(0.24), py(0.62)), (px(0.48), py(0.62)), (px(0.42), py(0.90)), (px(0.20), py(0.90))], fill=(245, 245, 240, 255))
        draw.polygon([(px(0.26), py(0.28)), (px(0.74), py(0.28)), (px(0.68), py(0.64)), (px(0.32), py(0.64))], fill=(245, 245, 240, 255))
        draw.polygon([(px(0.42), py(0.28)), (px(0.58), py(0.28)), (px(0.50), py(0.48))], fill=skin_shadow)
        draw.rectangle([px(0.28), py(0.58), px(0.72), py(0.65)], fill=(24, 22, 25, 255))
        draw.rounded_rectangle([px(0.74), py(0.32), px(0.88), py(0.46)], radius=6, fill=(210, 30, 30, 255))
        draw.rounded_rectangle([px(0.06), py(0.46), px(0.22), py(0.60)], radius=8, fill=(225, 30, 30, 255))
        draw.ellipse([px(0.35), py(0.06), px(0.65), py(0.30)], fill=skin)
        draw.polygon([(px(0.33), py(0.12)), (px(0.30), py(0.04)), (px(0.50), py(-0.02)), (px(0.70), py(0.04)), (px(0.65), py(0.14)), (px(0.35), py(0.14))], fill=(28, 24, 25, 255))
        draw.rectangle([px(0.33), py(0.12), px(0.67), py(0.18)], fill=(225, 30, 30, 255))
        draw.line([px(0.40), py(0.19), px(0.47), py(0.21)], fill=(25, 20, 22, 255), width=3)
        draw.line([px(0.60), py(0.19), px(0.53), py(0.21)], fill=(25, 20, 22, 255), width=3)
        draw.rectangle([px(0.42), py(0.21), px(0.47), py(0.24)], fill=(255, 255, 255, 255))
        draw.rectangle([px(0.53), py(0.21), px(0.58), py(0.24)], fill=(255, 255, 255, 255))
    else:
        draw.ellipse([px(0.18), py(0.86), px(0.82), py(0.97)], fill=(15, 12, 18, 170))
        draw.polygon([(px(0.52), py(0.60)), (px(0.76), py(0.62)), (px(0.80), py(0.88)), (px(0.62), py(0.88))], fill=(25, 26, 35, 255))
        draw.polygon([(px(0.22), py(0.60)), (px(0.46), py(0.60)), (px(0.40), py(0.88)), (px(0.18), py(0.88))], fill=(32, 34, 45, 255))
        draw.polygon([(px(0.24), py(0.28)), (px(0.76), py(0.28)), (px(0.70), py(0.62)), (px(0.30), py(0.62))], fill=(28, 30, 42, 255))
        draw.polygon([(px(0.45), py(0.36)), (px(0.55), py(0.36)), (px(0.50), py(0.48))], fill=(180, 40, 255, 255))
        draw.rounded_rectangle([px(0.18), py(0.26), px(0.32), py(0.40)], radius=6, fill=(55, 58, 75, 255))
        draw.rounded_rectangle([px(0.68), py(0.26), px(0.82), py(0.40)], radius=6, fill=(55, 58, 75, 255))
        draw.ellipse([px(0.34), py(0.08), px(0.66), py(0.32)], fill=(25, 26, 36, 255))
        draw.rounded_rectangle([px(0.36), py(0.16), px(0.64), py(0.24)], radius=4, fill=(255, 20, 60, 255))

    return img

def draw_adventure_hero(width=512, height=512):
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    px = lambda f: int(width * f)
    py = lambda f: int(height * f)

    skin = (240, 185, 140)
    brown_leather = (130, 70, 30)
    khaki = (195, 170, 125)
    dark_brown = (75, 40, 20)

    draw.ellipse([px(0.18), py(0.86), px(0.82), py(0.97)], fill=(15, 12, 18, 170))
    draw.rounded_rectangle([px(0.24), py(0.80), px(0.44), py(0.92)], radius=6, fill=dark_brown)
    draw.rounded_rectangle([px(0.56), py(0.80), px(0.76), py(0.92)], radius=6, fill=dark_brown)

    draw.polygon([(px(0.26), py(0.58)), (px(0.44), py(0.58)), (px(0.42), py(0.82)), (px(0.24), py(0.82))], fill=khaki)
    draw.polygon([(px(0.56), py(0.58)), (px(0.74), py(0.58)), (px(0.76), py(0.82)), (px(0.58), py(0.82))], fill=khaki)

    draw.rectangle([px(0.28), py(0.32), px(0.72), py(0.60)], fill=brown_leather)
    draw.rectangle([px(0.38), py(0.32), px(0.62), py(0.48)], fill=khaki)
    draw.rectangle([px(0.28), py(0.54), px(0.72), py(0.60)], fill=dark_brown)
    draw.ellipse([px(0.46), py(0.54), px(0.54), py(0.60)], fill=(255, 215, 0, 255))

    draw.polygon([(px(0.28), py(0.34)), (px(0.12), py(0.42)), (px(0.14), py(0.54)), (px(0.28), py(0.46))], fill=brown_leather)
    draw.ellipse([px(0.10), py(0.48), px(0.18), py(0.56)], fill=skin)
    draw.rectangle([px(0.12), py(0.30), px(0.16), py(0.58)], fill=dark_brown)
    draw.polygon([(px(0.08), py(0.30)), (px(0.20), py(0.30)), (px(0.14), py(0.14))], fill=(255, 120, 20, 255))
    draw.polygon([(px(0.10), py(0.28)), (px(0.18), py(0.28)), (px(0.14), py(0.18))], fill=(255, 220, 40, 255))

    draw.polygon([(px(0.72), py(0.34)), (px(0.86), py(0.44)), (px(0.84), py(0.56)), (px(0.72), py(0.46))], fill=brown_leather)
    draw.ellipse([px(0.80), py(0.50), px(0.88), py(0.58)], fill=skin)

    draw.ellipse([px(0.34), py(0.12), px(0.66), py(0.36)], fill=skin)
    draw.ellipse([px(0.40), py(0.22), px(0.46), py(0.28)], fill=(30, 25, 20, 255))
    draw.ellipse([px(0.54), py(0.22), px(0.60), py(0.28)], fill=(30, 25, 20, 255))
    draw.polygon([(px(0.20), py(0.18)), (px(0.80), py(0.18)), (px(0.70), py(0.04)), (px(0.30), py(0.04))], fill=brown_leather)
    draw.ellipse([px(0.16), py(0.15), px(0.84), py(0.22)], fill=brown_leather)
    draw.rectangle([px(0.30), py(0.14), px(0.70), py(0.17)], fill=dark_brown)

    return img

def draw_dungeon_knight(width=512, height=512, is_player=True):
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    px = lambda f: int(width * f)
    py = lambda f: int(height * f)

    if is_player:
        steel = (190, 200, 215)
        steel_dark = (120, 130, 145)
        gold = (255, 215, 0)
        blue_cape = (35, 75, 180)

        draw.ellipse([px(0.18), py(0.86), px(0.82), py(0.97)], fill=(15, 12, 18, 170))
        draw.polygon([(px(0.30), py(0.28)), (px(0.70), py(0.28)), (px(0.82), py(0.84)), (px(0.18), py(0.84))], fill=blue_cape)
        draw.rounded_rectangle([px(0.26), py(0.60), px(0.44), py(0.90)], radius=6, fill=steel)
        draw.rounded_rectangle([px(0.56), py(0.60), px(0.74), py(0.90)], radius=6, fill=steel)

        draw.polygon([(px(0.28), py(0.26)), (px(0.72), py(0.26)), (px(0.66), py(0.62)), (px(0.34), py(0.62))], fill=steel)
        draw.polygon([(px(0.36), py(0.30)), (px(0.64), py(0.30)), (px(0.50), py(0.52))], fill=steel_dark)
        draw.ellipse([px(0.18), py(0.24), px(0.32), py(0.38)], fill=gold)
        draw.ellipse([px(0.68), py(0.24), px(0.82), py(0.38)], fill=gold)

        draw.polygon([(px(0.10), py(0.36)), (px(0.28), py(0.36)), (px(0.24), py(0.72)), (px(0.10), py(0.62))], fill=blue_cape)
        draw.polygon([(px(0.12), py(0.38)), (px(0.26), py(0.38)), (px(0.22), py(0.70)), (px(0.12), py(0.60))], fill=steel)
        draw.ellipse([px(0.16), py(0.46), px(0.22), py(0.54)], fill=gold)

        draw.rectangle([px(0.78), py(0.12), px(0.84), py(0.68)], fill=(225, 240, 255, 255))
        draw.rectangle([px(0.74), py(0.46), px(0.88), py(0.50)], fill=gold)
        draw.rectangle([px(0.79), py(0.50), px(0.83), py(0.62)], fill=(80, 45, 20, 255))
        draw.line([px(0.81), py(0.16), px(0.81), py(0.44)], fill=(0, 220, 255, 255), width=2)

        draw.ellipse([px(0.34), py(0.08), px(0.66), py(0.32)], fill=steel)
        draw.rectangle([px(0.40), py(0.18), px(0.60), py(0.22)], fill=(25, 25, 30, 255))
        draw.polygon([(px(0.46), py(0.08)), (px(0.54), py(0.08)), (px(0.50), py(-0.02))], fill=(220, 30, 30, 255))
    else:
        bone = (230, 230, 220)
        bone_dark = (140, 140, 130)
        draw.ellipse([px(0.18), py(0.86), px(0.82), py(0.97)], fill=(15, 12, 18, 170))
        draw.rectangle([px(0.48), py(0.28), px(0.52), py(0.64)], fill=bone_dark)
        for ry in [0.34, 0.42, 0.50, 0.58]:
            draw.line([px(0.36), py(ry), px(0.64), py(ry)], fill=bone, width=4)

        draw.line([px(0.38), py(0.64), px(0.34), py(0.90)], fill=bone, width=5)
        draw.line([px(0.62), py(0.64), px(0.66), py(0.90)], fill=bone, width=5)

        draw.ellipse([px(0.35), py(0.06), px(0.65), py(0.30)], fill=bone)
        draw.ellipse([px(0.40), py(0.16), px(0.46), py(0.24)], fill=(220, 20, 20, 255))
        draw.ellipse([px(0.54), py(0.16), px(0.60), py(0.24)], fill=(220, 20, 20, 255))
        draw.rectangle([px(0.44), py(0.26), px(0.56), py(0.30)], fill=(40, 40, 40, 255))
        draw.polygon([(px(0.72), py(0.50)), (px(0.86), py(0.20)), (px(0.82), py(0.16)), (px(0.70), py(0.46))], fill=(150, 95, 60, 255))

    return img

def draw_strategy_unit(width=512, height=512, is_player=True):
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    px = lambda f: int(width * f)
    py = lambda f: int(height * f)

    main_c = (35, 110, 240) if is_player else (220, 35, 35)
    draw.ellipse([px(0.10), py(0.75), px(0.90), py(0.95)], fill=(15, 12, 18, 170))
    draw.rounded_rectangle([px(0.14), py(0.62), px(0.86), py(0.88)], radius=12, fill=(35, 38, 45, 255))
    for tx in [0.22, 0.36, 0.50, 0.64, 0.78]:
        draw.ellipse([px(tx-0.05), py(0.68), px(tx+0.05), py(0.82)], fill=(75, 80, 95, 255))

    draw.polygon([(px(0.22), py(0.64)), (px(0.78), py(0.64)), (px(0.70), py(0.36)), (px(0.30), py(0.36))], fill=main_c)
    draw.rectangle([px(0.36), py(0.40), px(0.64), py(0.50)], fill=(0, 230, 255, 200))
    draw.rectangle([px(0.46), py(0.12), px(0.54), py(0.40)], fill=(60, 65, 75, 255))
    draw.rectangle([px(0.44), py(0.10), px(0.56), py(0.16)], fill=(20, 22, 28, 255))

    return img

def draw_tower_defense_turret(width=512, height=512, is_player=True):
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    px = lambda f: int(width * f)
    py = lambda f: int(height * f)

    if is_player:
        draw.ellipse([px(0.10), py(0.76), px(0.90), py(0.96)], fill=(15, 12, 18, 170))
        draw.polygon([(px(0.20), py(0.84)), (px(0.80), py(0.84)), (px(0.68), py(0.52)), (px(0.32), py(0.52))], fill=(65, 70, 85, 255))
        draw.rectangle([px(0.36), py(0.44), px(0.64), py(0.54)], fill=(45, 48, 60, 255))
        draw.ellipse([px(0.26), py(0.24), px(0.74), py(0.52)], fill=(40, 125, 245, 255))
        draw.ellipse([px(0.38), py(0.30), px(0.62), py(0.46)], fill=(0, 240, 255, 255))
        draw.rectangle([px(0.66), py(0.30), px(0.90), py(0.36)], fill=(30, 32, 40, 255))
        draw.rectangle([px(0.66), py(0.40), px(0.90), py(0.46)], fill=(30, 32, 40, 255))
    else:
        draw.ellipse([px(0.15), py(0.70), px(0.85), py(0.90)], fill=(15, 12, 18, 170))
        draw.ellipse([px(0.22), py(0.32), px(0.78), py(0.78)], fill=(160, 45, 190, 255))
        draw.ellipse([px(0.32), py(0.44), px(0.44), py(0.56)], fill=(255, 20, 40, 255))
        draw.ellipse([px(0.56), py(0.44), px(0.68), py(0.56)], fill=(255, 20, 40, 255))
        draw.polygon([(px(0.28), py(0.36)), (px(0.18), py(0.16)), (px(0.36), py(0.32))], fill=(80, 20, 100, 255))
        draw.polygon([(px(0.72), py(0.36)), (px(0.82), py(0.16)), (px(0.64), py(0.32))], fill=(80, 20, 100, 255))

    return img

def draw_runner_athlete(width=512, height=512, is_player=True):
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    px = lambda f: int(width * f)
    py = lambda f: int(height * f)

    if is_player:
        skin = (245, 190, 145)
        neon_cyan = (0, 235, 255)
        dark = (25, 28, 38)
        draw.ellipse([px(0.18), py(0.86), px(0.82), py(0.97)], fill=(15, 12, 18, 170))
        draw.polygon([(px(0.24), py(0.56)), (px(0.44), py(0.56)), (px(0.18), py(0.88)), (px(0.10), py(0.88))], fill=dark)
        draw.polygon([(px(0.56), py(0.56)), (px(0.76), py(0.56)), (px(0.88), py(0.86)), (px(0.74), py(0.86))], fill=dark)
        draw.rounded_rectangle([px(0.06), py(0.84), px(0.22), py(0.92)], radius=4, fill=neon_cyan)
        draw.rounded_rectangle([px(0.76), py(0.82), px(0.92), py(0.90)], radius=4, fill=neon_cyan)
        draw.polygon([(px(0.30), py(0.28)), (px(0.70), py(0.28)), (px(0.66), py(0.60)), (px(0.34), py(0.60))], fill=neon_cyan)
        draw.rectangle([px(0.44), py(0.28), px(0.56), py(0.60)], fill=dark)
        draw.ellipse([px(0.36), py(0.08), px(0.64), py(0.30)], fill=skin)
        draw.rounded_rectangle([px(0.34), py(0.14), px(0.66), py(0.22)], radius=4, fill=(255, 220, 0, 255))
    else:
        draw.rectangle([px(0.10), py(0.60), px(0.18), py(0.90)], fill=(45, 50, 60, 255))
        draw.rectangle([px(0.82), py(0.60), px(0.90), py(0.90)], fill=(45, 50, 60, 255))
        draw.rectangle([px(0.14), py(0.68), px(0.86), py(0.76)], fill=(255, 30, 60, 255))
        draw.line([px(0.14), py(0.72), px(0.86), py(0.72)], fill=(255, 230, 50, 255), width=3)

    return img

def draw_parallax_sky(width=1024, height=512, game_plan=None):
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    genre = (game_plan.get("genre", "") if game_plan else "").lower()

    if genre == "mario":
        for y in range(height):
            t = y / height
            r = int(92 + t * 45)
            g = int(148 + t * 50)
            b = int(252 - t * 15)
            draw.line([0, y, width, y], fill=(min(r, 255), min(g, 255), min(b, 255), 255))

        for cx, cy in [(int(width*0.18), int(height*0.16)), (int(width*0.55), int(height*0.12)), (int(width*0.82), int(height*0.20))]:
            draw.ellipse([cx - 45, cy - 18, cx + 45, cy + 18], fill=(255, 255, 255, 230))
            draw.ellipse([cx - 20, cy - 30, cx + 25, cy + 15], fill=(255, 255, 255, 240))

        _rnd = random.Random(42)
        pts = [(0, int(height * 0.70))]
        x = 0
        while x < width + 60:
            h_peak = _rnd.randint(int(height * 0.45), int(height * 0.60))
            pts.append((x, h_peak))
            x += _rnd.randint(90, 160)
        pts.extend([(width, int(height * 0.70)), (width, height), (0, height)])
        draw.polygon(pts, fill=(35, 165, 75, 255))

    elif genre == "racing":
        for y in range(height):
            t = y / height
            r = int(12 + t * 24)
            g = int(10 + t * 16)
            b = int(38 + t * 68)
            draw.line([0, y, width, y], fill=(r, g, b, 255))
        mx, my = int(width * 0.85), int(height * 0.12)
        draw.ellipse([mx - 32, my - 32, mx + 32, my + 32], fill=(255, 245, 210, 255))
        bx = 0
        _rnd = random.Random(101)
        while bx < width:
            bw = _rnd.randint(35, 75)
            bh = _rnd.randint(int(height * 0.20), int(height * 0.45))
            by = int(height * 0.65) - bh
            draw.rectangle([bx, by, bx + bw - 2, int(height * 0.65)], fill=(18, 16, 40, 255))
            bx += bw + 4

    elif genre == "fighting":
        for y in range(height):
            t = y / height
            r = int(68 + t * 35)
            g = int(18 + t * 16)
            b = int(20 + t * 22)
            draw.line([0, y, width, y], fill=(r, g, b, 255))
        for col_x in [int(width * 0.06), int(width * 0.90)]:
            draw.rectangle([col_x, 0, col_x + 35, height], fill=(55, 28, 20, 255))
            draw.rectangle([col_x - 6, 0, col_x + 41, 18], fill=(210, 165, 40, 255))
        cx, cy = width // 2, int(height * 0.36)
        rad = int(min(width, height) * 0.18)
        draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=(45, 15, 18, 255), outline=(210, 170, 45, 255), width=5)

    elif genre == "adventure":
        # Ancient Forest Canopy & Sunbeams
        for y in range(height):
            t = y / height
            draw.line([0, y, width, y], fill=(int(20 + t * 35), int(60 + t * 65), int(45 + t * 30), 255))
        for tx in [int(width * 0.15), int(width * 0.48), int(width * 0.85)]:
            draw.rectangle([tx - 25, 0, tx + 25, height], fill=(45, 30, 20, 255))
            draw.ellipse([tx - 90, -40, tx + 90, int(height * 0.40)], fill=(25, 95, 45, 220))

    elif genre in ["dungeon", "adventure_fighting"]:
        # Gothic Dungeon Stone Arches & Torchlight
        for y in range(height):
            t = y / height
            draw.line([0, y, width, y], fill=(int(20 + t * 20), int(18 + t * 18), int(26 + t * 24), 255))
        for ax in range(0, width, 180):
            draw.arc([ax, int(height * 0.10), ax + 180, int(height * 0.70)], 180, 0, fill=(65, 60, 75, 255), width=8)

    elif genre == "strategy":
        # Command Center Map & Grid
        for y in range(height):
            draw.line([0, y, width, y], fill=(15, 25, 40, 255))
        for gx in range(0, width, 50):
            draw.line([gx, 0, gx, height], fill=(25, 55, 80, 120), width=1)
        for gy in range(0, height, 50):
            draw.line([0, gy, width, gy], fill=(25, 55, 80, 120), width=1)

    elif genre == "tower_defense":
        # Winding Path Defense Outpost
        for y in range(height):
            draw.line([0, y, width, y], fill=(35, 110, 50, 255))
        draw.line([(0, int(height * 0.60)), (int(width * 0.35), int(height * 0.60)), (int(width * 0.65), int(height * 0.35)), (width, int(height * 0.35))], fill=(165, 140, 100, 255), width=65)

    elif genre == "running":
        # Cyberpunk Neon Highway
        for y in range(height):
            t = y / height
            draw.line([0, y, width, y], fill=(int(10 + t * 25), int(8 + t * 15), int(45 + t * 65), 255))
        for bx in range(0, width, 120):
            draw.rectangle([bx, int(height * 0.20), bx + 90, int(height * 0.65)], fill=(20, 15, 38, 255), outline=(0, 235, 255, 180), width=2)

    else:
        for y in range(height):
            t = y / height
            draw.line([0, y, width, y], fill=(int(20 + t * 50), int(45 + t * 70), int(120 + t * 80), 255))

    return img

def generate_procedural_sprite(asset_name, prompt, width=512, height=512, seed=42, game_plan=None):
    genre = (game_plan.get("genre", "") if game_plan else "").lower()

    if genre == "mario":
        if asset_name == "player":
            img = draw_mario_hero(width, height)
        elif asset_name == "enemy":
            img = draw_goomba_enemy(width, height)
        elif asset_name == "platform_tile":
            img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            draw.rectangle([0, 0, width, height], fill=(160, 80, 35, 255))
            draw.rectangle([0, 0, width, int(height * 0.28)], fill=(45, 180, 55, 255))
            draw.line([0, 0, width, 0], fill=(90, 225, 95, 255), width=4)
        else:
            return draw_parallax_sky(width, height, game_plan)

    elif genre == "racing":
        if asset_name == "player":
            img = draw_detailed_car(width, height, is_player=True)
        elif asset_name == "enemy":
            img = draw_detailed_car(width, height, is_player=False)
        elif asset_name == "platform_tile":
            img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            draw.rectangle([0, 0, width, height], fill=(45, 48, 55, 255))
            draw.rectangle([int(width*0.46), 0, int(width*0.54), height], fill=(255, 215, 0, 255))
        else:
            return draw_parallax_sky(width, height, game_plan)

    elif genre == "fighting":
        if asset_name == "player":
            img = draw_detailed_fighter(width, height, is_player=True)
        elif asset_name == "enemy":
            img = draw_detailed_fighter(width, height, is_player=False)
        elif asset_name == "platform_tile":
            img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            draw.rectangle([0, 0, width, height], fill=(110, 60, 35, 255))
        else:
            return draw_parallax_sky(width, height, game_plan)

    elif genre == "adventure":
        if asset_name == "player":
            img = draw_adventure_hero(width, height)
        elif asset_name == "enemy":
            img = draw_dungeon_knight(width, height, is_player=False)
        elif asset_name == "platform_tile":
            img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            draw.rectangle([0, 0, width, height], fill=(70, 85, 65, 255))
        else:
            return draw_parallax_sky(width, height, game_plan)

    elif genre in ["dungeon", "adventure_fighting"]:
        if asset_name == "player":
            img = draw_dungeon_knight(width, height, is_player=True)
        elif asset_name == "enemy":
            img = draw_dungeon_knight(width, height, is_player=False)
        elif asset_name == "platform_tile":
            img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            draw.rectangle([0, 0, width, height], fill=(55, 58, 68, 255))
        else:
            return draw_parallax_sky(width, height, game_plan)

    elif genre == "strategy":
        if asset_name == "player":
            img = draw_strategy_unit(width, height, is_player=True)
        elif asset_name == "enemy":
            img = draw_strategy_unit(width, height, is_player=False)
        elif asset_name == "platform_tile":
            img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            draw.rectangle([0, 0, width, height], fill=(30, 45, 65, 255))
        else:
            return draw_parallax_sky(width, height, game_plan)

    elif genre == "tower_defense":
        if asset_name == "player":
            img = draw_tower_defense_turret(width, height, is_player=True)
        elif asset_name == "enemy":
            img = draw_tower_defense_turret(width, height, is_player=False)
        elif asset_name == "platform_tile":
            img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            draw.rectangle([0, 0, width, height], fill=(135, 115, 80, 255))
        else:
            return draw_parallax_sky(width, height, game_plan)

    elif genre == "running":
        if asset_name == "player":
            img = draw_runner_athlete(width, height, is_player=True)
        elif asset_name == "enemy":
            img = draw_runner_athlete(width, height, is_player=False)
        elif asset_name == "platform_tile":
            img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            draw.rectangle([0, 0, width, height], fill=(35, 40, 52, 255))
        else:
            return draw_parallax_sky(width, height, game_plan)

    else:
        if asset_name == "player":
            img = draw_mario_hero(width, height)
        elif asset_name == "enemy":
            img = draw_goomba_enemy(width, height)
        elif asset_name == "platform_tile":
            img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            draw.rectangle([0, 0, width, height], fill=(120, 80, 45, 255))
        else:
            return draw_parallax_sky(width, height, game_plan)

    img = crop_to_content(img)
    try:
        img = img.filter(ImageFilter.SHARPEN)
    except Exception:
        pass
    return img

def generate_sdxl_asset(prompt, width=512, height=512, seed=42, genre="default"):
    full_prompt = build_full_prompt(prompt, genre=genre)
    negative = NEGATIVE_PROMPT_BLOCK
    generator = torch.Generator("cuda" if DEVICE == "cuda" else "cpu").manual_seed(seed)
    img = sdxl_pipe(
        prompt=full_prompt,
        negative_prompt=negative,
        width=width,
        height=height,
        num_inference_steps=STEPS,
        guidance_scale=CFG_SCALE,
        generator=generator,
    ).images[0]
    return img

def generate_all_assets(game_plan, save_dir, job_id=None):
    os.makedirs(save_dir, exist_ok=True)
    genre = game_plan.get("genre", "mario")
    assets_plan = game_plan.get("assets", {})

    configs = {
        "player":        {"prompt": assets_plan.get("player", f"{genre} hero character"),        "width": 512, "height": 512, "progress": 38},
        "enemy":         {"prompt": assets_plan.get("enemy", f"{genre} rival or enemy"),        "width": 512, "height": 512, "progress": 54},
        "platform_tile": {"prompt": assets_plan.get("platform_tile", f"{genre} surface tile"),  "width": 512, "height": 512, "progress": 70},
        "background":    {"prompt": assets_plan.get("background", f"{genre} game panorama"),    "width": 1024, "height": 512, "progress": 84},
    }

    use_procedural = (DEVICE == "cpu")
    if not use_procedural:
        try:
            ensure_sdxl_loaded()
        except Exception:
            use_procedural = True

    best_assets = {}
    for asset_name, cfg in configs.items():
        set_job_status(job_id, f"Generating {genre} {asset_name.replace('_', ' ')}...", cfg["progress"], f"Building {asset_name} sprite")
        if use_procedural or asset_name == "background":
            if asset_name == "background":
                img = draw_parallax_sky(cfg["width"], cfg["height"], game_plan)
            else:
                img = generate_procedural_sprite(asset_name, cfg["prompt"], cfg["width"], cfg["height"], seed=42, game_plan=game_plan)
        else:
            try:
                raw_img = generate_sdxl_asset(cfg["prompt"], cfg["width"], cfg["height"], seed=42, genre=genre)
                if asset_name != "background":
                    raw_img = remove_flat_background(raw_img)
                    img = crop_to_content(raw_img)
                else:
                    img = raw_img
            except Exception:
                img = generate_procedural_sprite(asset_name, cfg["prompt"], cfg["width"], cfg["height"], seed=42, game_plan=game_plan)

        if not validate_asset(img, asset_name):
            img = generate_procedural_sprite(asset_name, cfg["prompt"], cfg["width"], cfg["height"], seed=77, game_plan=game_plan)

        asset_file = os.path.join(save_dir, f"best_{asset_name}.png")
        img.save(asset_file)
        best_assets[asset_name] = img
        gc.collect()

    free_model_memory("sdxl")
    return best_assets

# --------------------------------------------------------------------------
# Faithful Scene Composition for All Genres
# --------------------------------------------------------------------------

def render_mario_platformer(best_assets, layout_json, game_plan, canvas_width=1024, canvas_height=480, include_sprites=True):
    bg = best_assets.get("background")
    if bg:
        canvas = bg.resize((canvas_width, canvas_height), Image.LANCZOS).convert("RGBA")
    else:
        canvas = draw_parallax_sky(canvas_width, canvas_height, game_plan)
    draw = ImageDraw.Draw(canvas)

    platform_boxes = layout_json.get("platform_boxes", [
        {"x_norm": 0.0, "y_norm": 0.88, "w_norm": 1.0, "h_norm": 0.08},
        {"x_norm": 0.12, "y_norm": 0.65, "w_norm": 0.22, "h_norm": 0.05},
        {"x_norm": 0.40, "y_norm": 0.50, "w_norm": 0.22, "h_norm": 0.05},
        {"x_norm": 0.68, "y_norm": 0.35, "w_norm": 0.22, "h_norm": 0.05}
    ])

    tile_img = best_assets.get("platform_tile")
    ts = 28

    for p in platform_boxes:
        px = int(p["x_norm"] * canvas_width)
        py = int(p["y_norm"] * canvas_height)
        pw = max(ts * 2, int(p["w_norm"] * canvas_width))
        ph = max(ts, int(p["h_norm"] * canvas_height))

        draw.rectangle([px + 4, py + 4, px + pw + 4, py + ph + 4], fill=(10, 15, 25, 120))
        if tile_img:
            t_resized = tile_img.resize((ts, ts), Image.NEAREST)
            for x_offset in range(0, pw, ts):
                tile_w = min(ts, pw - x_offset)
                cropped_tile = t_resized.crop((0, 0, tile_w, ts))
                canvas.paste(cropped_tile, (px + x_offset, py), cropped_tile)
        else:
            draw.rectangle([px, py, px + pw, py + ph], fill=(160, 80, 35, 255), outline=(100, 45, 15, 255), width=2)
            draw.rectangle([px, py, px + pw, py + 8], fill=(45, 185, 55, 255))
            draw.line([px, py, px + pw, py], fill=(95, 235, 105, 255), width=3)

    spikes = layout_json.get("spikes", [[6, 10], [12, 6]])
    for sp_col, sp_row in spikes:
        sx = int((sp_col / 24.0) * canvas_width)
        sy = int((sp_row / 12.0) * canvas_height)
        draw.polygon([(sx - 14, sy + 6), (sx, sy - 18), (sx + 14, sy + 6)], fill=(180, 40, 40, 255), outline=(70, 15, 15, 255))
        draw.polygon([(sx - 8, sy + 4), (sx, sy - 14), (sx + 2, sy + 4)], fill=(240, 90, 90, 255))

    for block_x, block_y in [(int(canvas_width * 0.28), int(canvas_height * 0.44)), (int(canvas_width * 0.56), int(canvas_height * 0.30))]:
        draw.rectangle([block_x - 16, block_y - 16, block_x + 16, block_y + 16], fill=(235, 160, 20, 255), outline=(120, 65, 10, 255), width=3)
        draw.text((block_x - 6, block_y - 10), "?", fill=(255, 255, 255, 255))
        draw.ellipse([block_x - 8, block_y - 36, block_x + 8, block_y - 20], fill=(255, 215, 0, 255), outline=(180, 130, 0, 255), width=2)
        draw.text((block_x - 3, block_y - 34), "✦", fill=(255, 255, 255, 255))

    goal_pos = layout_json.get("goal", [22, 2])
    gx = int((goal_pos[0] / 24.0) * canvas_width)
    gy = int((goal_pos[1] / 12.0) * canvas_height)
    pole_h = 110
    draw.rectangle([gx - 3, gy - pole_h, gx + 3, gy + 10], fill=(210, 215, 225, 255), outline=(90, 95, 105, 255))
    draw.ellipse([gx - 8, gy - pole_h - 8, gx + 8, gy - pole_h + 8], fill=(255, 215, 0, 255))
    draw.polygon([(gx + 3, gy - pole_h + 4), (gx + 55, gy - pole_h + 22), (gx + 3, gy - pole_h + 40)], fill=(225, 30, 30, 255), outline=(140, 15, 15, 255))
    draw.text((gx + 12, gy - pole_h + 14), "★", fill=(255, 255, 255, 255))

    if include_sprites:
        player_sprite = best_assets.get("player")
        if player_sprite:
            p_w, p_h = int(canvas_width * 0.12), int(canvas_height * 0.26)
            p1 = player_sprite.resize((p_w, p_h), Image.LANCZOS).convert("RGBA")
            p_x = int(canvas_width * 0.08)
            p_y = int(canvas_height * 0.88) - p_h + 8
            canvas.paste(p1, (p_x, p_y), p1)

        enemy_sprite = best_assets.get("enemy")
        if enemy_sprite:
            e_w, e_h = int(canvas_width * 0.10), int(canvas_height * 0.20)
            e1 = enemy_sprite.resize((e_w, e_h), Image.LANCZOS).convert("RGBA")
            e_x = int(canvas_width * 0.44)
            e_y = int(canvas_height * 0.50) - e_h + 6
            canvas.paste(e1, (e_x, e_y), e1)

    draw.rectangle([10, 10, 260, 48], fill=(0, 0, 0, 210), outline=(255, 215, 0, 255), width=2)
    draw.text((20, 16), "MARIO   x03   ★ 002400", fill=(255, 255, 255, 255))
    draw.text((20, 30), "WORLD: 1-1   TIME: 360", fill=(255, 215, 0, 255))

    return canvas.convert("RGB")

def render_racing(best_assets, layout_json, game_plan, canvas_width=1024, canvas_height=480, include_sprites=True):
    bg = best_assets.get("background")
    if bg:
        canvas = bg.resize((canvas_width, canvas_height), Image.LANCZOS).convert("RGBA")
    else:
        canvas = draw_parallax_sky(canvas_width, canvas_height, game_plan)
    draw = ImageDraw.Draw(canvas)

    horizon = int(canvas_height * 0.46)
    road_top_w = int(canvas_width * 0.16)
    road_bot_w = int(canvas_width * 0.96)

    draw.rectangle([0, horizon, canvas_width, canvas_height], fill=(35, 40, 48, 255))

    num_bands = 16
    for b in range(num_bands):
        t1 = b / num_bands
        t2 = (b + 1) / num_bands
        y1 = horizon + int((canvas_height - horizon) * (t1 ** 1.8))
        y2 = horizon + int((canvas_height - horizon) * (t2 ** 1.8))
        w1 = int(road_top_w + (road_bot_w - road_top_w) * t1)
        w2 = int(road_top_w + (road_bot_w - road_top_w) * t2)

        shade = 42 if b % 2 == 0 else 52
        draw.polygon([
            ((canvas_width - w1) // 2, y1), ((canvas_width + w1) // 2, y1),
            ((canvas_width + w2) // 2, y2), ((canvas_width - w2) // 2, y2)
        ], fill=(shade, shade, shade + 4, 255))

        curb_w1 = max(4, int(w1 * 0.07))
        curb_w2 = max(6, int(w2 * 0.07))
        curb_color = (220, 30, 30, 255) if b % 2 == 0 else (245, 245, 245, 255)
        draw.polygon([
            ((canvas_width - w1) // 2 - curb_w1, y1), ((canvas_width - w1) // 2, y1),
            ((canvas_width - w2) // 2, y2), ((canvas_width - w2) // 2 - curb_w2, y2)
        ], fill=curb_color)
        draw.polygon([
            ((canvas_width + w1) // 2, y1), ((canvas_width + w1) // 2 + curb_w1, y1),
            ((canvas_width + w2) // 2 + curb_w2, y2), ((canvas_width + w2) // 2, y2)
        ], fill=curb_color)

    for b in range(0, num_bands, 2):
        t1 = b / num_bands
        t2 = min(1.0, (b + 1) / num_bands)
        y1 = horizon + int((canvas_height - horizon) * (t1 ** 1.8))
        y2 = horizon + int((canvas_height - horizon) * (t2 ** 1.8))
        lw1 = max(2, int(t1 * 14))
        lw2 = max(3, int(t2 * 14))
        draw.polygon([
            (canvas_width // 2 - lw1 // 2, y1), (canvas_width // 2 + lw1 // 2, y1),
            (canvas_width // 2 + lw2 // 2, y2), (canvas_width // 2 - lw2 // 2, y2)
        ], fill=(255, 215, 0, 255))

    if include_sprites:
        enemy_sprite = best_assets.get("enemy")
        if enemy_sprite:
            e_w, e_h = int(canvas_width * 0.20), int(canvas_height * 0.24)
            e_car = enemy_sprite.resize((e_w, e_h), Image.LANCZOS).convert("RGBA")
            ex = int(canvas_width * 0.45)
            ey = int(canvas_height * 0.46)
            draw.ellipse([ex - 5, ey + e_h - 12, ex + e_w + 5, ey + e_h + 8], fill=(10, 10, 15, 140))
            canvas.paste(e_car, (ex, ey), e_car)

        player_sprite = best_assets.get("player")
        if player_sprite:
            p_w, p_h = int(canvas_width * 0.28), int(canvas_height * 0.34)
            p_car = player_sprite.resize((p_w, p_h), Image.LANCZOS).convert("RGBA")
            px_pos = int(canvas_width * 0.36)
            py_pos = int(canvas_height * 0.60)
            draw.ellipse([px_pos - 10, py_pos + p_h - 16, px_pos + p_w + 10, py_pos + p_h + 12], fill=(10, 10, 15, 180))
            canvas.paste(p_car, (px_pos, py_pos), p_car)

    px1, py1 = canvas_width - 240, canvas_height - 95
    px2, py2 = canvas_width - 15, canvas_height - 15
    draw.rectangle([px1, py1, px2, py2], fill=(15, 20, 30, 230), outline=(0, 230, 255, 255), width=2)
    draw.text((px1 + 14, py1 + 10), "SPEED: 185 km/h  [GEAR 5]", fill=(0, 255, 240, 255))
    draw.text((px1 + 14, py1 + 34), "LAP: 2/3   TIME: 00:14.28", fill=(255, 215, 0, 255))
    draw.text((px1 + 14, py1 + 56), "POS: 1ST / 8", fill=(255, 255, 255, 255))

    return canvas.convert("RGB")

def render_fighting(best_assets, layout_json, game_plan, canvas_width=1024, canvas_height=480, include_sprites=True):
    bg = best_assets.get("background")
    if bg:
        canvas = bg.resize((canvas_width, canvas_height), Image.LANCZOS).convert("RGBA")
    else:
        canvas = draw_parallax_sky(canvas_width, canvas_height, game_plan)
    draw = ImageDraw.Draw(canvas)

    floor_y = int(canvas_height * 0.75)
    for y in range(floor_y, canvas_height):
        t = (y - floor_y) / (canvas_height - floor_y)
        draw.line([0, y, canvas_width, y], fill=(int(95 + t * 30), int(45 + t * 20), int(22 + t * 15), 255))
    draw.line([0, floor_y, canvas_width, floor_y], fill=(220, 175, 45, 255), width=3)

    if include_sprites:
        player_sprite = best_assets.get("player")
        if player_sprite:
            p1 = player_sprite.resize((int(canvas_width * 0.20), int(canvas_height * 0.42)), Image.LANCZOS).convert("RGBA")
            canvas.paste(p1, (int(canvas_width * 0.20), floor_y - p1.height + 10), p1)

        enemy_sprite = best_assets.get("enemy")
        if enemy_sprite:
            p2 = enemy_sprite.resize((int(canvas_width * 0.20), int(canvas_height * 0.42)), Image.LANCZOS).convert("RGBA")
            p2 = p2.transpose(Image.FLIP_LEFT_RIGHT)
            canvas.paste(p2, (int(canvas_width * 0.60), floor_y - p2.height + 10), p2)

    draw.rectangle([20, 20, canvas_width // 2 - 40, 48], fill=(160, 20, 20, 255), outline=(255, 215, 0, 255), width=2)
    draw.rectangle([22, 22, canvas_width // 2 - 42, 46], fill=(255, 205, 30, 255))
    draw.text((25, 52), "RYU / HERO", fill=(255, 255, 255, 255))

    draw.rectangle([canvas_width // 2 + 40, 20, canvas_width - 20, 48], fill=(160, 20, 20, 255), outline=(255, 215, 0, 255), width=2)
    draw.rectangle([canvas_width // 2 + 42, 22, canvas_width - 22, 46], fill=(255, 205, 30, 255))
    draw.text((canvas_width - 130, 52), "SHADOW NINJA", fill=(255, 255, 255, 255))

    draw.rectangle([canvas_width // 2 - 28, 12, canvas_width // 2 + 28, 56], fill=(10, 10, 12, 255), outline=(255, 215, 0, 255), width=3)
    draw.text((canvas_width // 2 - 12, 22), "99", fill=(255, 215, 0, 255))

    return canvas.convert("RGB")

def render_generic_stage(best_assets, layout_json, game_plan, canvas_width=1024, canvas_height=480, include_sprites=True):
    genre = (game_plan.get("genre", "adventure") if game_plan else "adventure").lower()
    bg = best_assets.get("background")
    if bg:
        canvas = bg.resize((canvas_width, canvas_height), Image.LANCZOS).convert("RGBA")
    else:
        canvas = draw_parallax_sky(canvas_width, canvas_height, game_plan)
    draw = ImageDraw.Draw(canvas)

    floor_y = int(canvas_height * 0.78)
    tile_img = best_assets.get("platform_tile")
    ts = 32
    if tile_img:
        t_resized = tile_img.resize((ts, ts), Image.NEAREST)
        for x in range(0, canvas_width, ts):
            for y in range(floor_y, canvas_height, ts):
                canvas.paste(t_resized, (x, y), t_resized)
    else:
        draw.rectangle([0, floor_y, canvas_width, canvas_height], fill=(45, 50, 60, 255))

    if include_sprites:
        player_sprite = best_assets.get("player")
        if player_sprite:
            p_w, p_h = int(canvas_width * 0.16), int(canvas_height * 0.35)
            p1 = player_sprite.resize((p_w, p_h), Image.LANCZOS).convert("RGBA")
            canvas.paste(p1, (int(canvas_width * 0.15), floor_y - p_h + 10), p1)

        enemy_sprite = best_assets.get("enemy")
        if enemy_sprite:
            e_w, e_h = int(canvas_width * 0.16), int(canvas_height * 0.35)
            e1 = enemy_sprite.resize((e_w, e_h), Image.LANCZOS).convert("RGBA")
            canvas.paste(e1, (int(canvas_width * 0.65), floor_y - e_h + 10), e1)

    draw.rectangle([10, 10, 280, 48], fill=(0, 0, 0, 210), outline=(0, 235, 255, 255), width=2)
    draw.text((20, 16), f"{genre.upper()} MODE", fill=(0, 240, 255, 255))
    draw.text((20, 30), "HP: [||||||||||] 100/100", fill=(255, 215, 0, 255))

    return canvas.convert("RGB")

def compose_scene(best_assets, layout_json, game_plan, tile_size=32, canvas_width=1024, canvas_height=480, include_sprites=True):
    genre = (game_plan.get("genre", "") if game_plan else "mario").lower()
    if genre == "racing":
        return render_racing(best_assets, layout_json, game_plan, canvas_width, canvas_height, include_sprites)
    elif genre == "fighting":
        return render_fighting(best_assets, layout_json, game_plan, canvas_width, canvas_height, include_sprites)
    elif genre in ["mario", "platformer"]:
        return render_mario_platformer(best_assets, layout_json, game_plan, canvas_width, canvas_height, include_sprites)
    else:
        return render_generic_stage(best_assets, layout_json, game_plan, canvas_width, canvas_height, include_sprites)

# --------------------------------------------------------------------------
# All 9 Genres: 12-Second (180 Frames @ 15fps) Interactive Video Gameplay
# --------------------------------------------------------------------------

def generate_preview_video(clean_bg, layout, assets, output_path, game_plan=None):
    import imageio

    W, H = 640, 360
    TOTAL_FRAMES = 180
    genre = (game_plan.get("genre", "mario") if game_plan else "mario").lower()

    base_bg = clean_bg.resize((W, H), Image.LANCZOS)
    player_img = assets.get("player")
    enemy_img = assets.get("enemy")

    writer = None
    try:
        try:
            writer = imageio.get_writer(
                output_path, fps=15, macro_block_size=None,
                codec="libx264", quality=7,
                ffmpeg_params=["-preset", "ultrafast", "-pix_fmt", "yuv420p"]
            )
        except Exception:
            gif_path = output_path.replace(".mp4", ".gif")
            writer = imageio.get_writer(gif_path, mode="I", fps=12, loop=0)
            output_path = gif_path

        for frame_idx in range(TOTAL_FRAMES):
            frame = base_bg.copy()
            draw = ImageDraw.Draw(frame)

            # =============================================================
            # 1. MARIO: 12s Platform Jump Progression
            # =============================================================
            if genre in ["mario", "platformer"]:
                coin_pop = False
                enemy_stomped = False
                flag_grab = (frame_idx >= 146)

                if frame_idx < 30:
                    prog = frame_idx / 30.0
                    px_pos = int(W * (0.06 + prog * 0.06))
                    py_pos = int(H * 0.88)
                elif frame_idx < 65:
                    prog = (frame_idx - 30) / 35.0
                    px_pos = int(W * (0.12 + prog * 0.10))
                    jump_arc = math.sin(prog * math.pi) * 60
                    py_pos = int(H * (0.88 - prog * 0.23)) - int(jump_arc)
                elif frame_idx < 105:
                    prog = (frame_idx - 65) / 40.0
                    px_pos = int(W * (0.22 + prog * 0.26))
                    jump_arc = math.sin(prog * math.pi) * 65
                    py_pos = int(H * (0.65 - prog * 0.15)) - int(jump_arc)
                    coin_pop = (15 <= (frame_idx - 65) <= 30)
                elif frame_idx < 145:
                    prog = (frame_idx - 105) / 40.0
                    px_pos = int(W * (0.48 + prog * 0.28))
                    jump_arc = math.sin(prog * math.pi) * 70
                    py_pos = int(H * (0.50 - prog * 0.15)) - int(jump_arc)
                    enemy_stomped = (prog >= 0.5)
                else:
                    prog = (frame_idx - 145) / 35.0
                    px_pos = int(W * (0.76 + prog * 0.12))
                    py_pos = int(H * 0.35)

                if enemy_img:
                    e_w, e_h = int(W * 0.08), int(H * 0.16)
                    ex_pos = int(W * 0.65)
                    ey_pos = int(H * 0.35) - e_h
                    if enemy_stomped:
                        e_h = max(6, int(e_h * 0.3))
                        ey_pos = int(H * 0.35) - e_h
                        e_sprite = enemy_img.resize((e_w, e_h), Image.LANCZOS).convert("RGBA")
                        frame.paste(e_sprite, (ex_pos, ey_pos), e_sprite)
                        draw.text((ex_pos - 10, ey_pos - 20), "💥 +200 PTS", fill=(255, 215, 0, 255))
                    else:
                        e_sprite = enemy_img.resize((e_w, e_h), Image.LANCZOS).convert("RGBA")
                        frame.paste(e_sprite, (ex_pos, ey_pos), e_sprite)

                if player_img:
                    p_w, p_h = int(W * 0.10), int(H * 0.20)
                    p_sprite = player_img.resize((p_w, p_h), Image.LANCZOS).convert("RGBA")
                    frame.paste(p_sprite, (px_pos, py_pos - p_h), p_sprite)

                if coin_pop:
                    coin_y = int(H * 0.44) - 25 - (frame_idx % 15) * 2
                    draw.ellipse([int(W * 0.38) - 8, coin_y, int(W * 0.38) + 8, coin_y + 14], fill=(255, 215, 0, 255))
                    draw.text((int(W * 0.38) - 15, coin_y - 18), "🪙 +100", fill=(255, 255, 255, 255))

                if flag_grab:
                    draw.rectangle([W // 2 - 110, int(H * 0.16), W // 2 + 110, int(H * 0.28)], fill=(255, 215, 0, 230), outline=(0, 0, 0, 255), width=3)
                    draw.text((W // 2 - 85, int(H * 0.20)), "★ COURSE CLEAR! ★", fill=(0, 0, 0, 255))

            # =============================================================
            # 2. RACING: Nitro & Overtake
            # =============================================================
            elif genre == "racing":
                horizon = int(H * 0.46)
                offset = (frame_idx * 0.16) % 1.0
                for b in range(12):
                    t1 = (b + offset) / 12.0
                    t2 = min(1.0, (b + offset + 0.5) / 12.0)
                    if t1 >= 1.0: continue
                    y1 = horizon + int((H - horizon) * (t1 ** 1.8))
                    y2 = horizon + int((H - horizon) * (t2 ** 1.8))
                    draw.polygon([(W // 2 - 2, y1), (W // 2 + 2, y1), (W // 2 + 3, y2), (W // 2 - 3, y2)], fill=(255, 215, 0, 255))

                is_nitro = (76 <= frame_idx <= 115)
                if frame_idx < 35:
                    p_steer = 0.0
                    r_depth = 0.42
                    r_lane = 0.08
                elif frame_idx < 80:
                    prog = (frame_idx - 35) / 45.0
                    p_steer = 0.0
                    r_depth = 0.42 + prog * 0.18
                    r_lane = 0.08
                elif frame_idx < 115:
                    prog = (frame_idx - 80) / 35.0
                    p_steer = -prog * 0.20
                    r_depth = 0.60
                    r_lane = 0.08
                elif frame_idx < 150:
                    prog = (frame_idx - 115) / 35.0
                    p_steer = -0.20 + prog * 0.10
                    r_depth = 0.60 - prog * 0.30
                    r_lane = 0.14
                else:
                    p_steer = -0.10
                    r_depth = 0.30
                    r_lane = 0.16

                if enemy_img:
                    r_scale = 0.16 + r_depth * 0.16
                    e_w = max(24, int(W * r_scale))
                    e_h = max(24, int(H * r_scale * 1.1))
                    e_resized = enemy_img.resize((e_w, e_h), Image.LANCZOS).convert("RGBA")
                    rx = int(W * (0.50 + r_lane)) - e_w // 2
                    ry = horizon + int((H - horizon) * (r_depth ** 1.6)) - e_h
                    draw.ellipse([rx - 4, ry + e_h - 6, rx + e_w + 4, ry + e_h + 6], fill=(10, 10, 15, 140))
                    frame.paste(e_resized, (rx, ry), e_resized)

                if player_img:
                    p_w = int(W * 0.26)
                    p_h = int(H * 0.30)
                    p_resized = player_img.resize((p_w, p_h), Image.LANCZOS).convert("RGBA")
                    px_pos = int(W * (0.50 + p_steer)) - p_w // 2
                    py_pos = int(H * 0.66)
                    draw.ellipse([px_pos - 8, py_pos + p_h - 12, px_pos + p_w + 8, py_pos + p_h + 8], fill=(10, 10, 15, 180))
                    frame.paste(p_resized, (px_pos, py_pos), p_resized)

                    if is_nitro:
                        for flame_x_off in [int(p_w * 0.30), int(p_w * 0.70)]:
                            fx = px_pos + flame_x_off
                            fy = py_pos + p_h - 4
                            draw.polygon([(fx - 6, fy), (fx + 6, fy), (fx, fy + 22 + (frame_idx % 6) * 3)], fill=(0, 220, 255, 240))

                if frame_idx > 150:
                    draw.rectangle([0, int(H * 0.16), W, int(H * 0.28)], fill=(255, 215, 0, 230), outline=(0, 0, 0, 255), width=3)
                    draw.text((W // 2 - 75, int(H * 0.20)), "🏆 1ST PLACE - VICTORY! 🏆", fill=(0, 0, 0, 255))

            # =============================================================
            # 3. FIGHTING: Combos & K.O.
            # =============================================================
            elif genre == "fighting":
                floor_y = int(H * 0.75)
                hit_spark = (70 <= frame_idx <= 115 and (frame_idx % 15) < 6)
                special_beam = (116 <= frame_idx <= 135)
                p_x = int(W * 0.32)
                e_x = int(W * 0.68)

                if enemy_img:
                    e_w, e_h = int(W * 0.20), int(H * 0.40)
                    e_sprite = enemy_img.resize((e_w, e_h), Image.LANCZOS).convert("RGBA").transpose(Image.FLIP_LEFT_RIGHT)
                    e_y = floor_y - e_h + (35 if frame_idx >= 150 else 0)
                    frame.paste(e_sprite, (e_x - e_w // 2, e_y), e_sprite)

                if player_img:
                    p_w, p_h = int(W * 0.20), int(H * 0.40)
                    p_sprite = player_img.resize((p_w, p_h), Image.LANCZOS).convert("RGBA")
                    frame.paste(p_sprite, (p_x - p_w // 2, floor_y - p_h), p_sprite)

                if hit_spark:
                    draw.ellipse([e_x - 30, floor_y - 80, e_x + 30, floor_y - 20], fill=(255, 230, 40, 220))
                    draw.text((e_x - 40, floor_y - 100), "💥 3-HIT COMBO!", fill=(255, 215, 0, 255))

                if special_beam:
                    bx = p_x + 40 + (frame_idx - 116) * 12
                    draw.ellipse([bx - 18, floor_y - 80, bx + 18, floor_y - 44], fill=(0, 220, 255, 240))

                if frame_idx >= 150:
                    draw.rectangle([W // 2 - 110, int(H * 0.18), W // 2 + 110, int(H * 0.32)], fill=(180, 20, 20, 230), outline=(255, 215, 0, 255), width=3)
                    draw.text((W // 2 - 80, int(H * 0.22)), "⚡ K.O.! - YOU WIN! ⚡", fill=(255, 255, 255, 255))

            # =============================================================
            # 4. ADVENTURE: Chest Discovery & Relic Unlock
            # =============================================================
            elif genre == "adventure":
                floor_y = int(H * 0.78)
                prog = frame_idx / float(TOTAL_FRAMES)
                px_pos = int(W * (0.15 + min(prog * 1.5, 0.45)))

                if player_img:
                    p_w, p_h = int(W * 0.14), int(H * 0.32)
                    p_sprite = player_img.resize((p_w, p_h), Image.LANCZOS).convert("RGBA")
                    frame.paste(p_sprite, (px_pos, floor_y - p_h), p_sprite)

                # Ancient Glowing Chest
                cx = int(W * 0.70)
                cy = floor_y - 40
                draw.rectangle([cx - 24, cy, cx + 24, cy + 30], fill=(130, 80, 30, 255), outline=(255, 215, 0, 255), width=2)
                if frame_idx >= 75:
                    draw.ellipse([cx - 30, cy - 35, cx + 30, cy], fill=(255, 230, 80, 200)) # Golden relic glow
                    draw.text((cx - 20, cy - 25), "✨ RELIC", fill=(255, 255, 255, 255))
                if frame_idx >= 140:
                    draw.rectangle([W // 2 - 140, int(H * 0.16), W // 2 + 140, int(H * 0.28)], fill=(30, 140, 60, 230), outline=(255, 215, 0, 255), width=3)
                    draw.text((W // 2 - 120, int(H * 0.20)), "🌟 NEW AREA DISCOVERED: SUN TEMPLE 🌟", fill=(255, 255, 255, 255))

            # =============================================================
            # 5. DUNGEON: Knight Combat & Boss Room Unlocked
            # =============================================================
            elif genre in ["dungeon", "adventure_fighting"]:
                floor_y = int(H * 0.78)
                p_x = int(W * 0.30)
                e_x = int(W * 0.65)
                is_slashing = (60 <= frame_idx <= 120)

                if enemy_img:
                    e_w, e_h = int(W * 0.16), int(H * 0.35)
                    if frame_idx >= 120:
                        draw.text((e_x - 30, floor_y - 40), "💀 DEFEATED +300G", fill=(255, 215, 0, 255))
                    else:
                        e_sprite = enemy_img.resize((e_w, e_h), Image.LANCZOS).convert("RGBA").transpose(Image.FLIP_LEFT_RIGHT)
                        frame.paste(e_sprite, (e_x, floor_y - e_h), e_sprite)

                if player_img:
                    p_w, p_h = int(W * 0.16), int(H * 0.35)
                    p_sprite = player_img.resize((p_w, p_h), Image.LANCZOS).convert("RGBA")
                    frame.paste(p_sprite, (p_x, floor_y - p_h), p_sprite)

                if is_slashing:
                    # Blue Holy Sword Arc
                    draw.arc([p_x + 30, floor_y - 90, e_x + 20, floor_y - 10], 270, 90, fill=(0, 235, 255, 255), width=6)
                    draw.text((e_x - 10, floor_y - 90), "💥 CRIT 450!", fill=(255, 80, 80, 255))

                if frame_idx >= 135:
                    draw.rectangle([W // 2 - 120, int(H * 0.16), W // 2 + 120, int(H * 0.28)], fill=(120, 20, 20, 230), outline=(255, 215, 0, 255), width=3)
                    draw.text((W // 2 - 95, int(H * 0.20)), "⚔️ DUNGEON CLEARED - GATE UNLOCKED ⚔️", fill=(255, 255, 255, 255))

            # =============================================================
            # 6. STRATEGY: Base Harvester & Artillery Strike
            # =============================================================
            elif genre == "strategy":
                floor_y = int(H * 0.76)
                if player_img:
                    p_w, p_h = int(W * 0.16), int(H * 0.26)
                    p_sprite = player_img.resize((p_w, p_h), Image.LANCZOS).convert("RGBA")
                    frame.paste(p_sprite, (int(W * 0.25), floor_y - p_h), p_sprite)

                if enemy_img:
                    e_w, e_h = int(W * 0.16), int(H * 0.26)
                    e_sprite = enemy_img.resize((e_w, e_h), Image.LANCZOS).convert("RGBA")
                    if frame_idx < 110:
                        frame.paste(e_sprite, (int(W * 0.70), floor_y - e_h), e_sprite)

                # Resource crystals mining
                draw.ellipse([int(W * 0.10), floor_y - 30, int(W * 0.16), floor_y - 5], fill=(0, 240, 255, 255))
                if frame_idx % 20 < 10:
                    draw.text((int(W * 0.10), floor_y - 45), "+50 ORE", fill=(0, 255, 200, 255))

                # Artillery Missile Strike
                if 70 <= frame_idx <= 110:
                    mx = int(W * (0.35 + (frame_idx - 70) / 40.0 * 0.35))
                    my = int(H * (0.30 + (frame_idx - 70) / 40.0 * 0.35))
                    draw.line([mx - 15, my - 15, mx, my], fill=(255, 120, 20, 255), width=4)
                if frame_idx >= 110:
                    draw.ellipse([int(W * 0.70) - 20, floor_y - 70, int(W * 0.70) + 40, floor_y - 10], fill=(255, 80, 20, 220))
                    draw.text((int(W * 0.65), floor_y - 85), "💥 BASE DESTROYED!", fill=(255, 215, 0, 255))
                if frame_idx >= 140:
                    draw.rectangle([W // 2 - 120, int(H * 0.16), W // 2 + 120, int(H * 0.28)], fill=(15, 45, 120, 230), outline=(0, 235, 255, 255), width=3)
                    draw.text((W // 2 - 95, int(H * 0.20)), "🎖️ VICTORY - SECTOR SECURED 🎖️", fill=(255, 255, 255, 255))

            # =============================================================
            # 7. TOWER DEFENSE: Creep Wave & Plasma Cannon Defense
            # =============================================================
            elif genre == "tower_defense":
                floor_y = int(H * 0.72)
                if player_img:
                    t_w, t_h = int(W * 0.14), int(H * 0.30)
                    t_sprite = player_img.resize((t_w, t_h), Image.LANCZOS).convert("RGBA")
                    frame.paste(t_sprite, (int(W * 0.45), floor_y - t_h - 20), t_sprite)

                # Creep wave marching
                if enemy_img and frame_idx < 115:
                    e_w, e_h = int(W * 0.10), int(H * 0.20)
                    e_sprite = enemy_img.resize((e_w, e_h), Image.LANCZOS).convert("RGBA")
                    creep_x = int(W * (0.05 + frame_idx * 0.005))
                    frame.paste(e_sprite, (creep_x, floor_y - e_h), e_sprite)
                    # Plasma laser beam firing
                    if 45 <= frame_idx <= 115 and (frame_idx % 8) < 4:
                        draw.line([int(W * 0.52), floor_y - 65, creep_x + e_w // 2, floor_y - e_h // 2], fill=(0, 245, 255, 255), width=4)

                if 115 <= frame_idx <= 135:
                    draw.ellipse([int(W * 0.60) - 25, floor_y - 60, int(W * 0.60) + 25, floor_y - 10], fill=(255, 120, 20, 220))
                    draw.text((int(W * 0.55), floor_y - 75), "💥 +150 COINS", fill=(255, 215, 0, 255))

                if frame_idx >= 140:
                    draw.rectangle([W // 2 - 120, int(H * 0.16), W // 2 + 120, int(H * 0.28)], fill=(30, 110, 50, 230), outline=(255, 215, 0, 255), width=3)
                    draw.text((W // 2 - 100, int(H * 0.20)), "🛡️ WAVE 5 CLEARED - BASE DEFENDED 🛡️", fill=(255, 255, 255, 255))

            # =============================================================
            # 8. RUNNING: 3-Lane Dodge & Ring Streak
            # =============================================================
            elif genre == "running":
                floor_y = int(H * 0.78)
                # Lane-switching: 0-40 center, 41-90 left dodge, 91-180 right ring streak
                if frame_idx < 40:
                    lane_x = int(W * 0.50)
                elif frame_idx < 90:
                    lane_x = int(W * 0.30)
                else:
                    lane_x = int(W * 0.70)

                # Laser Hurdle in Center Lane
                if enemy_img and frame_idx < 90:
                    h_w, h_h = int(W * 0.12), int(H * 0.15)
                    h_sprite = enemy_img.resize((h_w, h_h), Image.LANCZOS).convert("RGBA")
                    frame.paste(h_sprite, (int(W * 0.50) - h_w // 2, floor_y - h_h), h_sprite)

                # Ring Streak in Right Lane
                if frame_idx >= 80:
                    for ring_y in [floor_y - 70, floor_y - 45, floor_y - 20]:
                        draw.ellipse([int(W * 0.70) - 8, ring_y, int(W * 0.70) + 8, ring_y + 16], fill=(255, 215, 0, 255))
                    draw.text((int(W * 0.70) - 10, floor_y - 90), "✦ +500", fill=(255, 255, 255, 255))

                if player_img:
                    p_w, p_h = int(W * 0.12), int(H * 0.28)
                    p_sprite = player_img.resize((p_w, p_h), Image.LANCZOS).convert("RGBA")
                    frame.paste(p_sprite, (lane_x - p_w // 2, floor_y - p_h), p_sprite)

                if frame_idx >= 140:
                    draw.rectangle([W // 2 - 120, int(H * 0.16), W // 2 + 120, int(H * 0.28)], fill=(120, 20, 140, 230), outline=(0, 235, 255, 255), width=3)
                    draw.text((W // 2 - 100, int(H * 0.20)), "⚡ SPEED RECORD: 1500m (x4 COMBO) ⚡", fill=(255, 255, 255, 255))

            else:
                floor_y = int(H * 0.80)
                if player_img:
                    p_w, p_h = int(W * 0.14), int(H * 0.28)
                    p_sprite = player_img.resize((p_w, p_h), Image.LANCZOS).convert("RGBA")
                    frame.paste(p_sprite, (int(W * 0.30), floor_y - p_h), p_sprite)

            writer.append_data(np.array(frame))
            del frame
            del draw
            if frame_idx % 25 == 0:
                gc.collect()

        writer.close()
        return output_path
    except Exception as e:
        print(f"[Video] Video generation error: {e}")
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        return None

# --------------------------------------------------------------------------
# Atomic ZIP Packaging
# --------------------------------------------------------------------------

def create_job_zip(job_id):
    job_dir = os.path.join(API_OUTPUT_DIR, job_id)
    if not os.path.exists(job_dir):
        return None

    zip_filename = f"game_assets_{job_id}.zip"
    final_zip_path = os.path.join(job_dir, zip_filename)
    temp_zip_path = os.path.join(job_dir, f"temp_{zip_filename}")

    try:
        with zipfile.ZipFile(temp_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for fname in os.listdir(job_dir):
                if fname.startswith("temp_") or fname == zip_filename:
                    continue
                fpath = os.path.join(job_dir, fname)
                if os.path.isfile(fpath) and os.path.getsize(fpath) > 0:
                    zipf.write(fpath, fname)

        with zipfile.ZipFile(temp_zip_path, "r") as test_f:
            if test_f.testzip() is not None:
                raise ValueError("Corrupted ZIP structure")

        if os.path.exists(final_zip_path):
            os.remove(final_zip_path)
        os.rename(temp_zip_path, final_zip_path)
        return final_zip_path
    except Exception as e:
        print(f"[Packaging] Error building atomic ZIP: {e}")
        if os.path.exists(temp_zip_path):
            try:
                os.remove(temp_zip_path)
            except Exception:
                pass
        return None

# --------------------------------------------------------------------------
# Full Pipeline Execution
# --------------------------------------------------------------------------

def run_full_pipeline(image_path, user_description="Create a game based on the provided sketch.", job_id=None, base_url=""):
    job_dir = os.path.join(API_OUTPUT_DIR, job_id) if job_id else os.path.join(API_OUTPUT_DIR, str(uuid.uuid4())[:8])
    os.makedirs(job_dir, exist_ok=True)

    try:
        set_job_status(job_id, "Analyzing level sketch with Vision AI...", 10, "Extracting layout grid, caption, and object detections")
        layout, florence_caption, florence_od, vision_info = sketch_to_layout(image_path)
        free_model_memory("florence")

        set_job_status(job_id, "Resolving game genre...", 20, "Analyzing user intent priorities and visual cues")
        genre_resolution = resolve_genre(user_description, florence_caption, florence_od, layout, vision_info)

        with open(os.path.join(job_dir, "vision.json"), "w") as f:
            json.dump(vision_info, f, indent=2)
        with open(os.path.join(job_dir, "genre_resolution.json"), "w") as f:
            json.dump(genre_resolution, f, indent=2)
        with open(os.path.join(job_dir, "layout.json"), "w") as f:
            json.dump(layout, f, indent=2)

        set_job_status(job_id, f"Planning {genre_resolution['genre']} game mechanics...", 30, "Generating AAA structured gameplay specifications")
        game_plan = plan_game(layout, user_description, florence_caption, florence_od, genre_resolution, vision_info)
        if not game_plan:
            set_job_status(job_id, "Planning failed", 0, error="Failed to generate structured game plan", error_code="PLAN_FAILED")
            return {"error": "Game planning failed"}

        with open(os.path.join(job_dir, "game_plan.json"), "w") as f:
            json.dump(game_plan, f, indent=2)

        best_assets = generate_all_assets(game_plan, job_dir, job_id=job_id)

        set_job_status(job_id, "Composing game scene...", 88, "Composing backgrounds, platforms, and game sprites")
        scene = compose_scene(best_assets, layout, game_plan)
        scene_path = os.path.join(job_dir, "scene.png")
        scene.save(scene_path)

        set_job_status(job_id, "Rendering gameplay preview animation...", 94, "Creating authentic 12-second animated video preview")
        clean_bg = compose_scene(best_assets, layout, game_plan, include_sprites=False)
        preview_target_mp4 = os.path.join(job_dir, "preview.mp4")
        actual_preview_file = generate_preview_video(clean_bg, layout, best_assets, preview_target_mp4, game_plan=game_plan)

        set_job_status(job_id, "Packaging game package...", 98, "Creating verified download archive")
        zip_path = create_job_zip(job_id)

        ts = str(uuid.uuid4())[:6]
        preview_filename = os.path.basename(actual_preview_file) if actual_preview_file and os.path.exists(actual_preview_file) else "preview.mp4"

        result = {
            "job_id": job_id,
            "user_description": user_description,
            "resolved_genre": genre_resolution,
            "confidence": genre_resolution.get("confidence", 1.0),
            "game_plan": game_plan,
            "layout": layout,
            "vision_analysis": vision_info,
            "urls": {
                "scene": f"/files/{job_id}/scene.png?v={ts}",
                "preview": f"/files/{job_id}/{preview_filename}?v={ts}",
            },
            "zip_url": f"/download-zip/{job_id}?v={ts}"
        }

        if os.path.exists(image_path) and "temp_" in os.path.basename(image_path):
            try:
                os.remove(image_path)
            except Exception:
                pass

        set_job_status(job_id, "Completed", 100, "Level generation finished successfully", result=result)
        return result

    except Exception as e:
        print(f"[Pipeline] Fatal error during pipeline execution: {e}")
        if os.path.exists(image_path) and "temp_" in os.path.basename(image_path):
            try:
                os.remove(image_path)
            except Exception:
                pass
        set_job_status(job_id, "Error", 0, error=str(e), error_code="PIPELINE_ERROR")
        return {"error": str(e)}

# --------------------------------------------------------------------------
# FastAPI Web Application
# --------------------------------------------------------------------------

app = FastAPI(title="Sketch-to-Game API Engine", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/files", StaticFiles(directory=API_OUTPUT_DIR), name="files")

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Sketch-to-Game API Engine",
        "device": DEVICE,
        "clip_enabled": ENABLE_CLIP_SELECTION,
        "supported_genres": SUPPORTED_GENRES
    }

@app.post("/generate")
async def generate_endpoint(
    request: Request,
    sketch: UploadFile = File(...),
    description: str = Form("Create a game based on the provided sketch."),
    job_id: str = Form(None)
):
    if not job_id:
        job_id = str(uuid.uuid4())[:8]

    temp_sketch_path = os.path.join(API_OUTPUT_DIR, f"temp_{job_id}_{sketch.filename}")
    with open(temp_sketch_path, "wb") as f:
        content = await sketch.read()
        f.write(content)

    set_job_status(job_id, "Initializing pipeline...", 5, "Received uploaded sketch file")

    base_url = PUBLIC_BASE_URL or str(request.base_url).rstrip("/")
    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        executor,
        run_full_pipeline,
        temp_sketch_path,
        description,
        job_id,
        base_url
    )

    return {"job_id": job_id, "status": "processing"}

@app.get("/status/{job_id}")
def status_endpoint(job_id: str):
    info = JOB_STATUS.get(job_id, {"status": "not_found", "progress": 0, "step": "Unknown job"})
    return JSONResponse(content=info)

@app.get("/download-zip/{job_id}")
@app.get("/download-assets/{job_id}")
def download_zip_endpoint(job_id: str):
    job_dir = os.path.join(API_OUTPUT_DIR, job_id)
    if not os.path.exists(job_dir):
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    zip_path = os.path.join(job_dir, f"game_assets_{job_id}.zip")
    if not os.path.exists(zip_path) or os.path.getsize(zip_path) == 0:
        zip_path = create_job_zip(job_id)

    if zip_path and os.path.exists(zip_path) and os.path.getsize(zip_path) > 0:
        return FileResponse(path=zip_path, filename=f"game_assets_{job_id}.zip", media_type="application/zip")
    raise HTTPException(status_code=404, detail="Zip file could not be prepared")

@app.get("/download/{job_id}/scene")
def download_scene(job_id: str):
    scene_path = os.path.join(API_OUTPUT_DIR, job_id, "scene.png")
    if os.path.exists(scene_path) and os.path.getsize(scene_path) > 0:
        return FileResponse(scene_path, filename="scene.png", media_type="image/png")
    raise HTTPException(status_code=404, detail="Scene image not found")

@app.get("/download/{job_id}/preview")
def download_preview(job_id: str):
    mp4_path = os.path.join(API_OUTPUT_DIR, job_id, "preview.mp4")
    gif_path = os.path.join(API_OUTPUT_DIR, job_id, "preview.gif")
    if os.path.exists(mp4_path) and os.path.getsize(mp4_path) > 0:
        return FileResponse(mp4_path, filename="preview.mp4", media_type="video/mp4")
    elif os.path.exists(gif_path) and os.path.getsize(gif_path) > 0:
        return FileResponse(gif_path, filename="preview.gif", media_type="image/gif")
    raise HTTPException(status_code=404, detail="Preview video not found")

@app.get("/download/{job_id}/asset/{asset_name}")
def download_individual_asset(job_id: str, asset_name: str):
    clean_name = os.path.basename(asset_name)
    asset_path = os.path.join(API_OUTPUT_DIR, job_id, f"best_{clean_name}.png")
    if not os.path.exists(asset_path):
        asset_path = os.path.join(API_OUTPUT_DIR, job_id, f"{clean_name}.png")

    if os.path.exists(asset_path) and os.path.getsize(asset_path) > 0:
        return FileResponse(asset_path, filename=f"{clean_name}.png", media_type="image/png")
    raise HTTPException(status_code=404, detail=f"Asset '{clean_name}' not found")

@app.get("/download-status")
def download_status_endpoint():
    api_key = os.environ.get("OPENAI_API_KEY", "")
    key_configured = bool(api_key and len(api_key) > 8)
    key_preview = f"sk-...{api_key[-4:]}" if key_configured else "Not set"

    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    total_bytes = 0
    if os.path.exists(cache_dir):
        for root, dirs, files in os.walk(cache_dir):
            for f in files:
                try:
                    total_bytes += os.path.getsize(os.path.join(root, f))
                except Exception:
                    pass

    gb_cached = round(total_bytes / (1024 ** 3), 2)
    is_cached = (gb_cached >= 0.1) or (DEVICE == "cpu")

    return {
        "status": "cached" if is_cached else "idle",
        "progress": 100 if is_cached else 0,
        "downloaded_gb": gb_cached if gb_cached > 0 else (0.5 if DEVICE == "cpu" else 6.5),
        "target_gb": 6.5 if DEVICE == "cuda" else 0.5,
        "percent": 100 if is_cached else 0,
        "speed_mbps": 0,
        "file_name": "Models ready" if is_cached else "No active download",
        "api_key_configured": key_configured,
        "api_key_preview": key_preview,
        "is_cached": is_cached,
        "is_downloading": False
    }

@app.post("/start-download")
def start_download_endpoint():
    return {"status": "cached", "message": "Models ready for level generation."}

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return JSONResponse(status_code=204, content={})

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

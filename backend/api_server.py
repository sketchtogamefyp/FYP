"""
Sketch-to-Game pipeline API.
"""

import gc
import json
import os
import uuid
import math
import random
import time
from collections import deque
import asyncio
from concurrent.futures import ThreadPoolExecutor

import zipfile
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from openai import OpenAI

from prompt_config import (
    STYLE_LOCK_BLOCK,
    STYLE_BLOCKS,
    get_style_block,
    NEGATIVE_PROMPT_BLOCK,
    STEPS,
    CFG_SCALE,
    LORA_TRIGGER,
    build_full_prompt,
    validate_payload,
)

# Enable high-speed Rust multi-threaded parallel downloads
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

# Check for fine-tuned LoRA weights in parent or local folder
DEFAULT_FLORENCE_LORA = "../florence_game_lora/final" if os.path.exists("../florence_game_lora/final") else "./models/florence_lora"
FLORENCE_LORA_PATH = os.environ.get("FLORENCE_LORA_PATH", DEFAULT_FLORENCE_LORA)
SDXL_LORA_PATH = os.environ.get("SDXL_LORA_PATH", "./models/sdxl_lora")

def get_openai_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Please set it in backend/.env or environment variables."
        )
    return OpenAI(api_key=api_key)

API_OUTPUT_DIR = os.environ.get("API_OUTPUT_DIR", "./api_output")
os.makedirs(API_OUTPUT_DIR, exist_ok=True)

def get_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

DEVICE = os.environ.get("DEVICE", get_device())

executor = ThreadPoolExecutor(max_workers=2)
JOB_STATUS = {}


def set_job_status(job_id, step, progress, details="", result=None, error=None):
    if not job_id:
        return
    JOB_STATUS[job_id] = {
        "status": "error" if error else ("completed" if progress >= 100 else "processing"),
        "step": step,
        "progress": progress,
        "details": details,
        "result": result,
        "error": error,
    }

# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------

florence_processor = None
florence_base = None
florence_model = None
sdxl_pipe = None
clip_model = None
clip_preprocess = None
clip_tokenizer = None


def ensure_florence_loaded():
    global florence_processor, florence_base, florence_model, sdxl_pipe
    import torch
    from transformers import AutoProcessor, AutoModelForCausalLM
    from peft import PeftModel

    if florence_model is not None:
        return

    if sdxl_pipe is not None:
        print("Freeing SDXL from memory before loading Florence-2...")
        sdxl_pipe = None
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    print("Loading Florence-2...")
    florence_processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    # Load in bfloat16 to halve memory usage on CPU (critical for 512MB Render limit)
    # low_cpu_mem_usage prevents RAM spiking during initialization
    dtype = torch.float16 if DEVICE == "cuda" else torch.bfloat16
    florence_base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True,
        torch_dtype=dtype,
        device_map=DEVICE,
        low_cpu_mem_usage=True
    )
    if os.path.exists(FLORENCE_LORA_PATH) and len(os.listdir(FLORENCE_LORA_PATH)) > 0:
        print(f"Loading fine-tuned Florence-2 LoRA from {FLORENCE_LORA_PATH}...")
        florence_model = PeftModel.from_pretrained(florence_base, FLORENCE_LORA_PATH)
    else:
        print(f"No custom Florence LoRA found at {FLORENCE_LORA_PATH} - using base Florence-2 model...")
        florence_model = florence_base

    florence_model.eval()
    print("Florence-2 ready.")


def _load_sdxl_pipeline():
    import torch
    from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
    print("Loading SDXL...")
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
        print("Loading custom-trained pixel art LoRA...")
        pipe.load_lora_weights(SDXL_LORA_PATH)
    else:
        print("No custom LoRA found - loading pretrained nerijs/pixel-art-xl...")
        pipe.load_lora_weights("nerijs/pixel-art-xl", weight_name="pixel-art-xl.safetensors")
    pipe.fuse_lora(lora_scale=0.90)
    return pipe


def ensure_sdxl_loaded():
    global sdxl_pipe, florence_model, florence_base
    import torch

    if sdxl_pipe is not None:
        return

    if florence_model is not None:
        print("Freeing Florence-2 from memory before loading SDXL...")
        florence_model = None
        florence_base = None
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    sdxl_pipe = _load_sdxl_pipeline()


def ensure_clip_loaded():
    global clip_model, clip_preprocess, clip_tokenizer
    import open_clip
    if clip_model is not None:
        return
    print("Loading CLIP model...")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
    clip_model.eval()

# --------------------------------------------------------------------------
# Phase 2 - Sketch -> Layout JSON & Rich Visual Analysis (2-Pass Florence-2)
# --------------------------------------------------------------------------

def free_model_memory(target="all"):
    global florence_model, florence_base, florence_processor
    global sdxl_pipe
    global clip_model, clip_preprocess, clip_tokenizer
    import gc
    import torch

    print(f"Aggressively freeing memory for target: {target}...")

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
    print("Memory cleanup complete.")



def gpt4o_sketch_to_layout(image_path):
    """Uses GPT-4o Vision to extract layout, objects, and caption, bypassing Florence-2 entirely on low RAM."""
    import base64
    from PIL import Image
    import json
    
    # 1. Base64 encode the image
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
    client = get_openai_client()
    
    prompt = """Analyze this hand-drawn game level sketch. 
Return ONLY a valid JSON block with these exact keys:
- "caption": A detailed description of what the sketch depicts.
- "objects": A list of detected objects (e.g., character, enemy, chest, spikes). For each, give {"type": "name", "position": [col, row], "bbox": [ymin, xmin, ymax, xmax]} where col is 0-23, row is 0-11, and bbox values are 0-1000.
- "layout": {"platforms": [[col, row], ...], "player": [col, row], "enemies": [[col, row], ...], "items": [[col, row], ...]} where col is 0-23 and row is 0-11.
- "scene": {"environment": "fantasy/city/dungeon/etc", "camera": "side_view/top_down/behind_car"}
Do not include markdown blocks or any other text, just the raw JSON."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            max_tokens=1000,
            temperature=0.2
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)
        
        layout = data.get("layout", {"platforms": []})
        caption = data.get("caption", "A hand-drawn game level sketch")
        
        vision_info = {
            "caption": caption,
            "objects": data.get("objects", []),
            "scene": data.get("scene", {"environment": "fantasy", "camera": "side_view"}),
            "spatial_relations": []
        }
        
        return layout, caption, json.dumps(data.get("objects", [])), vision_info
        
    except Exception as e:
        print(f"GPT-4o Vision analysis failed: {e}")
        return None, "A hand-drawn game level sketch", "", {"objects": [], "scene": {"environment": "fantasy", "camera": "side_view"}}

def sketch_to_layout(image_path):
    if DEVICE == "cpu":
        print("[Memory-Save] Bypassing Florence-2 and using GPT-4o Vision for sketch analysis.")
        return gpt4o_sketch_to_layout(image_path)

    raw_caption = "A hand-drawn game level sketch"
    raw_od = ""
    vision_status = "unavailable"
    objects = []
    scene = {"environment": "fantasy", "camera": "side_view"}
    visual_genre_evidence = []
    layout = None

    if florence_model is not None:
        vision_status = "available"
        try:
            import torch
            img = Image.open(image_path).convert("RGB")
            dtype = torch.float16 if DEVICE == "cuda" else torch.float32
            
            # 1. Run Fine-tuned LoRA for Layout Extraction
            inputs_layout = florence_processor(
                text="<DETAILED_CAPTION>", images=img, return_tensors="pt"
            ).to(DEVICE, dtype)
            with torch.no_grad():
                gen_layout = florence_model.generate(**inputs_layout, max_new_tokens=1024, do_sample=False)
            raw_layout_str = florence_processor.tokenizer.decode(gen_layout[0], skip_special_tokens=True)
            try:
                layout = json.loads(raw_layout_str)
                print("[AI-Layout] Successfully parsed LoRA-predicted layout.")
            except Exception as le:
                print(f"[AI-Layout-Error] Failed to parse LoRA layout JSON: {le}. Raw string: {raw_layout_str[:200]}")
                layout = None

            # 2. Run detailed caption
            inputs_cap = florence_processor(
                text="<MORE_DETAILED_CAPTION>", images=img, return_tensors="pt"
            ).to(DEVICE, dtype)
            with torch.no_grad():
                gen_cap = florence_model.generate(**inputs_cap, max_new_tokens=256, do_sample=False)
            raw_caption = florence_processor.tokenizer.decode(gen_cap[0], skip_special_tokens=True)

            # 3. Run Object Detection
            inputs_od = florence_processor(
                text="<OD>", images=img, return_tensors="pt"
            ).to(DEVICE, dtype)
            with torch.no_grad():
                gen_od = florence_model.generate(**inputs_od, max_new_tokens=256, do_sample=False)
            raw_od = florence_processor.tokenizer.decode(gen_od[0], skip_special_tokens=True)
            
            # Post-process OD to extract clean objects
            try:
                parsed_od = florence_processor.post_process_generation(
                    raw_od, task="<OD>", image_size=img.size
                )
                if parsed_od and "<OD>" in parsed_od:
                    od_data = parsed_od["<OD>"]
                    for bbox, label in zip(od_data.get("bboxes", []), od_data.get("labels", [])):
                        ymin, xmin, ymax, xmax = bbox
                        norm_bbox = [
                            int(ymin / img.size[1] * 1000),
                            int(xmin / img.size[0] * 1000),
                            int(ymax / img.size[1] * 1000),
                            int(xmax / img.size[0] * 1000)
                        ]
                        cx = (xmin + xmax) / 2
                        cy = (ymin + ymax) / 2
                        grid_col = int(cx / img.size[0] * 24)
                        grid_row = int(cy / img.size[1] * 12)
                        objects.append({
                            "type": label.replace(" ", "_"),
                            "position": [grid_col, grid_row],
                            "bbox": norm_bbox
                        })
                        visual_genre_evidence.append(label.lower())
            except Exception as ode:
                print(f"OD parsing note: {ode}")

        except Exception as e:
            print(f"Florence-2 pipeline exception: {e}")
            vision_status = "unavailable"

    # Infer scene attributes
    cap_lower = raw_caption.lower()
    if "forest" in cap_lower or "tree" in cap_lower or "outdoor" in cap_lower:
        scene["environment"] = "forest"
    elif "city" in cap_lower or "street" in cap_lower or "neon" in cap_lower or "highway" in cap_lower:
        scene["environment"] = "city"
    elif "dungeon" in cap_lower or "cave" in cap_lower or "stone" in cap_lower or "wall" in cap_lower:
        scene["environment"] = "dungeon"
    elif "space" in cap_lower or "star" in cap_lower or "nebula" in cap_lower:
        scene["environment"] = "space"
    elif "dojo" in cap_lower or "ring" in cap_lower or "arena" in cap_lower:
        scene["environment"] = "arena"

    if "behind" in cap_lower or "car" in cap_lower or "driving" in cap_lower:
        scene["camera"] = "behind_car"
    elif "top" in cap_lower or "dungeon" in cap_lower or "strategy" in cap_lower or "ortho" in cap_lower:
        scene["camera"] = "top_down"
    else:
        scene["camera"] = "side_view"

    # Collect additional visual evidence from caption words
    for word in cap_lower.split():
        cleaned_word = "".join(c for c in word if c.isalnum())
        if cleaned_word in ["car", "road", "track", "race", "fighter", "brawler", "chest", "treasure", "key", "sword", "tower", "castle", "runner", "lane", "spikes"]:
            visual_genre_evidence.append(cleaned_word)

    visual_genre_evidence = list(set(visual_genre_evidence))

    # OpenCV Fallback Layout Extraction if AI Layout is missing or invalid
    if layout is None:
        try:
            img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img_gray is None:
                img_pil = Image.open(image_path).convert("L")
                img_gray = np.array(img_pil)

            h, w = img_gray.shape
            grid_cols, grid_rows = 24, 12
            cell_w, cell_h = w / grid_cols, h / grid_rows

            _, thresh = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY_INV)
            h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 30, 15), 2))
            h_thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel)

            platform_set = set()
            for r in range(grid_rows):
                for c in range(grid_cols):
                    x1, y1 = int(c * cell_w), int(r * cell_h)
                    x2, y2 = int((c + 1) * cell_w), int((r + 1) * cell_h)
                    
                    cell_full = thresh[y1:y2, x1:x2]
                    cell_h_layer = h_thresh[y1:y2, x1:x2]
                    
                    if np.mean(cell_full) > 10 or np.mean(cell_h_layer) > 4:
                        platform_set.add((c, r))

            platforms = [list(p) for p in platform_set]

            if len(platforms) > 5:
                sorted_for_player = sorted(platforms, key=lambda p: (-p[1], p[0]))
                player = [sorted_for_player[0][0], max(0, sorted_for_player[0][1] - 1)]

                sorted_for_goal = sorted(platforms, key=lambda p: (p[1], -p[0]))
                goal = [sorted_for_goal[0][0], max(0, sorted_for_goal[0][1] - 1)]

                enemies = []
                if len(sorted_for_player) > 10:
                    enemies = [
                        sorted_for_player[len(sorted_for_player) // 3],
                        sorted_for_player[2 * len(sorted_for_player) // 3]
                    ]

                layout = {
                    "player": player,
                    "goal": goal,
                    "platforms": platforms,
                    "enemies": enemies
                }
        except Exception as e:
            print(f"Fallback layout extraction note: {e}")

    if layout is None:
        layout = {
            "player": [1, 10],
            "goal": [22, 3],
            "platforms": [
                [0, 11], [1, 11], [2, 11], [3, 11], [4, 11], [5, 11], [6, 11], [7, 11], [8, 11], [9, 11], [10, 11],
                [3, 8], [4, 8], [5, 8],
                [8, 6], [9, 6], [10, 6],
                [14, 5], [15, 5], [16, 5],
                [19, 4], [20, 4], [21, 4], [22, 4], [23, 4]
            ],
            "enemies": [[5, 7], [15, 4]]
        }

    # Populate object positions from extracted layout
    if "player" in layout and not any(o["type"] == "player_character" for o in objects):
        objects.append({"type": "player_character", "position": layout["player"], "bbox": [0,0,0,0]})
    if "goal" in layout and not any(o["type"] == "goal_portal" for o in objects):
        objects.append({"type": "goal_portal", "position": layout["goal"], "bbox": [0,0,0,0]})
    for enemy in layout.get("enemies", []):
        if not any(o["type"] == "enemy_patrol" and o["position"] == enemy for o in objects):
            objects.append({"type": "enemy_patrol", "position": enemy, "bbox": [0,0,0,0]})

    vision_info = {
        "vision_status": vision_status,
        "caption": raw_caption if vision_status == "available" else None,
        "objects": objects,
        "scene": scene,
        "spatial_relations": ["player_near_ground", "enemy_facing_player"],
        "visual_genre_evidence": visual_genre_evidence
    }

    return layout, raw_caption, raw_od, vision_info


def resolve_genre(user_description, florence_caption, florence_od, layout, vision_info):
    user_desc_lower = user_description.lower()
    
    # 1. User Intent Mapping
    user_genre = None
    reason = ""
    source = "default"
    confidence = 0.5
    
    # Check hybrids first to prevent partial matching (e.g. "adventure fighting")
    if "adventure fighting" in user_desc_lower or "action adventure" in user_desc_lower or "hack and slash" in user_desc_lower:
        user_genre = "fighting" # Use fighting mechanics in an adventure setting
        reason = "User requested a hybrid 'adventure fighting' genre; prioritizing combat mechanics."
        source = "user_instruction"
        confidence = 0.99
    
    if not user_genre:
        genre_keywords = {
            "racing": ["racing", "race", "car", "drive", "driving", "track", "vehicle", "kart"],
            "fighting": ["fighting", "fight", "brawler", "combat", "arena", "beatemup", "beat 'em up", "smash"],
            "dungeon": ["dungeon", "crawler", "maze", "basement", "corridor", "roguelike"],
            "strategy": ["strategy", "rts", "base", "units", "territory", "build", "tactics"],
            "platformer": ["platformer", "mario", "jumping", "jump", "megaman", "sonic", "platform"],
            "tower_defense": ["tower defense", "tower", "td", "defense"],
            "running": ["running", "runner", "infinite run", "temple run", "subway surfers", "endless runner"],
            "adventure": ["adventure", "explore", "quest", "chest", "treasure", "forest", "rpg", "fantasy", "zelda"]
        }
        
        for g, keywords in genre_keywords.items():
            for kw in keywords:
                if kw in user_desc_lower:
                    user_genre = g
                    reason = f"User explicitly requested a genre matching keyword '{kw}'."
                    source = "user_instruction"
                    confidence = 0.95
                    break
            if user_genre:
                break
            
    # 2. Visual Evidence Analysis
    visual_evidence = vision_info.get("visual_genre_evidence", [])
    visual_candidates = []
    # Using the same mapping for visual checking
    genre_keywords_vis = {
        "racing": ["racing", "race", "car", "drive", "driving", "track", "vehicle"],
        "fighting": ["fighting", "fight", "brawler", "combat", "arena", "beatemup", "beat 'em up"],
        "dungeon": ["dungeon", "crawler", "maze", "basement", "corridor"],
        "strategy": ["strategy", "rts", "base", "units", "territory", "build"],
        "platformer": ["platformer", "mario", "jumping", "jump"],
        "tower_defense": ["tower defense", "tower", "td", "defense"],
        "running": ["running", "runner", "infinite run"],
        "adventure": ["adventure", "explore", "quest", "chest", "treasure", "forest", "rpg", "fantasy"]
    }
    
    for g, keywords in genre_keywords_vis.items():
        score = 0.0
        for kw in keywords:
            if any(kw in ev.lower() for ev in visual_evidence):
                score += 0.4
            if florence_caption and kw in florence_caption.lower():
                score += 0.3
        if score > 0:
            visual_candidates.append({"genre": g, "score": min(score, 1.0)})
            
    visual_candidates = sorted(visual_candidates, key=lambda x: x["score"], reverse=True)
    vis_genre = visual_candidates[0]["genre"] if visual_candidates else None
    vis_score = visual_candidates[0]["score"] if visual_candidates else 0.0
    
    # 3. Structural Heuristics
    platforms = layout.get("platforms", [])
    enemies = layout.get("enemies", [])
    
    if not platforms and len(enemies) > 2:
        struct_genre = "running"
    else:
        struct_genre = "platformer"
        
    # Combine signals
    resolved_genre = "platformer"
    visual_conflict = False
    
    if user_genre:
        resolved_genre = user_genre
        if vis_genre and vis_genre != user_genre and vis_score > 0.6:
            visual_conflict = True
            reason += f" Note: Visual evidence suggests '{vis_genre}' (score {vis_score}), indicating a conflict with user request."
    elif vis_genre and vis_score > 0.5:
        resolved_genre = vis_genre
        source = "visual_evidence"
        confidence = 0.8
        reason = f"Visual analysis detected strong evidence of '{vis_genre}' (score {vis_score})."
    else:
        resolved_genre = struct_genre or "platformer"
        source = "structural_heuristics"
        confidence = 0.6
        reason = "No clear text or visual cues. Inferred from spatial coordinates layout."
        
    return {
        "genre": resolved_genre,
        "confidence": confidence,
        "source": source,
        "user_requested_genre": user_genre,
        "visual_genre_candidates": visual_candidates,
        "visual_conflict": visual_conflict,
        "reason": reason
    }



GAME_PLAN_SYSTEM_PROMPT = """You are an expert AAA Game Designer, Technical Director, and AI Planning Engine.

You receive a resolved genre, visual scene understanding data, layout coordinates, and user requests.
Your absolute first command is to obey the resolved genre. Do not change it.

Under the "genre_specific" field in the JSON output, you must provide genre-appropriate parameters:
- For racing: {"track_type": "city circuit", "laps": 3, "boost": true, "opponents": 3, "checkpoints": 5}
- For fighting: {"rounds": 3, "health": 100, "special_meter": 100, "arena_boundary": true}
- For adventure: {"quests": 3, "npc_count": 2, "map_size": "medium", "keys_required": 1}
- For dungeon: {"rooms": 4, "boss_health": 200, "keys": 2, "has_minimap": true}
- For strategy: {"max_units": 50, "resource_types": ["gold", "wood"], "has_fog_of_war": true}
- For platformer: {"jump_height": 3, "checkpoint_count": 2, "lives": 3}
- For tower_defense: {"waves": 10, "tower_types": ["cannon", "slow", "rapid"], "enemy_path": true, "base_health": 20}
- For running: {"lanes": 3, "initial_speed": 5, "multiplier": 1.1, "has_obstacles": true}

Provide your response in raw JSON format (no markdown, no backticks):
{
  "genre": "Exact Resolved Genre",
  "genre_confidence": 0.0,
  "sketch_interpretation": "One clear sentence explaining what you understood the sketch to depict and why you chose this genre.",
  "user_intent": "Summary of what the user requested",
  "camera": {
    "style": "Side Camera / Behind Car / Fixed Arena / Orthographic Top Down / Follow Camera",
    "fov": 60,
    "smoothing": 0.15
  },
  "title": "Creative AAA Game Title",
  "theme": "Creative Theme Name",
  "description": "Engaging single sentence game description",
  "color_palette": {
    "sky": "#hex",
    "ground": "#hex",
    "platform": "#hex",
    "accent": "#hex"
  },
  "gameplay_systems": {
    "health": 100,
    "lives": 3,
    "movement_style": "physics_driven",
    "attack_key": "SPACE / J / CLICK",
    "attack_action": "Sword Slash / Laser / Nitro Boost / Magic Spell / Stomp",
    "win_condition": "Clear win objective derived from genre",
    "lose_condition": "Lose condition derived from genre",
    "scoring": "Score system"
  },
  "animations": {
    "player": ["idle", "walk", "run", "jump", "attack", "hit", "death"],
    "enemy": ["idle", "patrol", "attack", "hit", "death"],
    "environment": ["wind_sway", "water_flow", "torch_fire", "glow_pulse"]
  },
  "physics_data": {
    "gravity": 9.8,
    "move_speed": 6.0,
    "jump_force": 12.5,
    "friction": 0.85,
    "restitution": 0.1
  },
  "level_structure": {
    "learning_area": "Introduction zone",
    "challenge_area": "Main obstacle trajectory",
    "reward_area": "Power-up placement",
    "goal_area": "Finish area"
  },
  "assets": {
    "player": "detailed 16-bit SNES arcade pixel art prompt",
    "enemy": "detailed 16-bit SNES arcade pixel art prompt",
    "platform_tile": "detailed 16-bit SNES arcade pixel art prompt",
    "background": "detailed 16-bit SNES arcade pixel art prompt"
  },
  "video_prompt": "detailed 16-bit game video preview description",
  "asset_metadata": [
    { "name": "player", "category": "character/hero", "collision": "solid", "gameplay": "user_controlled" },
    { "name": "enemy", "category": "hazard/opponent", "collision": "trigger_damage", "gameplay": "patrol_ai" }
  ],
  "difficulty": "medium",
  "enemy_count": 3,
  "genre_specific": {}
}"""


def plan_game(layout_json, user_description, florence_caption, florence_od, genre_resolution, vision_info):
    user_msg = (
        f"USER REQUEST / THEME:\n{user_description}\n\n"
        f"RESOLVED GENRE:\n{genre_resolution['genre']}\n\n"
        f"GENRE CONFIDENCE:\n{genre_resolution['confidence']}\n\n"
        f"SKETCH CAPTION:\n{florence_caption}\n\n"
        f"DETECTED OBJECTS:\n{json.dumps(vision_info.get('objects', []), indent=2)}\n\n"
        f"SCENE:\n{json.dumps(vision_info.get('scene', {}), indent=2)}\n\n"
        f"SPATIAL RELATIONS:\n{json.dumps(vision_info.get('spatial_relations', []), indent=2)}\n\n"
        f"LAYOUT:\n{json.dumps(layout_json, indent=2)}\n\n"
        f"VISUAL GENRE EVIDENCE:\n{json.dumps(genre_resolution.get('visual_genre_candidates', []), indent=2)}"
    )
    client = get_openai_client()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": GAME_PLAN_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.7,
        max_tokens=1200,
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        plan = json.loads(raw)
        if "sketch_interpretation" not in plan or not plan["sketch_interpretation"]:
            plan["sketch_interpretation"] = f"The sketch was interpreted as a {plan.get('genre', 'action')} scene based on visual layout and drawing patterns."
        # Ensure exact resolved genre is kept
        plan["genre"] = genre_resolution["genre"]
        plan["genre_confidence"] = genre_resolution["confidence"]
        return plan
    except json.JSONDecodeError:
        print(f"GPT-4o returned invalid JSON:\n{raw[:500]}")
        return None

# --------------------------------------------------------------------------
# Phase 4 - High-Detail Organic Procedural Sprite Renderer
# --------------------------------------------------------------------------

def draw_parallax_sky(width=1024, height=512, game_plan=None):
    import random as _rnd
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    genre_lower = (game_plan.get("genre", "") if game_plan else "").lower()
    theme_lower = (game_plan.get("theme", "") if game_plan else "").lower()
    is_car = "rac" in genre_lower or "car" in genre_lower or "vehicle" in theme_lower or "racer" in theme_lower
    is_fighting = "fight" in genre_lower or "arena" in genre_lower or "beat" in genre_lower or "brawl" in genre_lower or "fight" in theme_lower
    is_shooter = "shoot" in genre_lower or "top down" in genre_lower

    if is_car:
        # ═══ RACING: Cyberpunk Night City 3-Layer Parallax ═══════════════
        # Deep indigo-to-purple sky gradient
        for y in range(height):
            t = y / height
            r = int(5 + t * 20)
            g = int(2 + t * 12)
            b = int(18 + t * 50)
            draw.line([0, y, width, y], fill=(r, g, b, 255))

        # Glowing crescent moon with soft halo
        mx, my = int(width * 0.88), int(height * 0.08)
        for ring in range(6, 0, -1):
            alpha = int(15 + ring * 5)
            draw.ellipse([mx - 40 - ring*6, my - 40 - ring*6, mx + 40 + ring*6, my + 40 + ring*6],
                         fill=(60, 50, 90, alpha))
        draw.ellipse([mx - 30, my - 30, mx + 30, my + 30], fill=(255, 248, 220, 255))
        draw.ellipse([mx - 18, my - 34, mx + 22, my + 26], fill=(5, 2, 18, 255))

        # Stars with twinkling variety
        _rnd.seed(777)
        for _ in range(100):
            sx = _rnd.randint(0, width)
            sy = _rnd.randint(0, int(height * 0.55))
            br = _rnd.randint(140, 255)
            sz = _rnd.choice([1, 1, 1, 2, 2])
            color = _rnd.choice([(br, br, br, 255), (br, br-20, br-60, 255), (br-40, br-20, br, 255)])
            draw.rectangle([sx, sy, sx + sz, sy + sz], fill=color)

        # ── Layer 1: Far distant buildings (small, dark) ──
        _rnd.seed(101)
        bx = 0
        while bx < width:
            bw = _rnd.randint(30, 65)
            bh = _rnd.randint(int(height * 0.15), int(height * 0.35))
            by = int(height * 0.68) - bh
            shade = _rnd.randint(12, 22)
            draw.rectangle([bx, by, bx + bw - 2, int(height * 0.68)], fill=(shade, shade - 2, shade + 15, 255))
            # Tiny dim windows
            for wy in range(by + 6, int(height * 0.68) - 6, 12):
                for wx in range(bx + 4, bx + bw - 6, 10):
                    if _rnd.random() > 0.6:
                        wc = _rnd.choice([(80, 70, 40, 255), (40, 60, 80, 255)])
                        draw.rectangle([wx, wy, wx + 3, wy + 4], fill=wc)
            bx += bw + _rnd.randint(1, 6)

        # ── Layer 2: Mid-range buildings (medium, with neon) ──
        _rnd.seed(202)
        neon_colors = [(0, 230, 255), (255, 50, 180), (50, 255, 160), (255, 100, 40), (180, 80, 255)]
        bx = _rnd.randint(0, 20)
        while bx < width:
            bw = _rnd.randint(45, 100)
            bh = _rnd.randint(int(height * 0.25), int(height * 0.50))
            by = int(height * 0.72) - bh
            shade = _rnd.randint(18, 32)
            draw.rectangle([bx, by, bx + bw - 3, int(height * 0.72)], fill=(shade, shade - 4, shade + 20, 255))
            # Lit windows (warm yellow/orange/cyan)
            for wy in range(by + 8, int(height * 0.72) - 8, 14):
                for wx in range(bx + 5, bx + bw - 8, 12):
                    if _rnd.random() > 0.35:
                        wc = _rnd.choice([(255, 215, 70, 255), (255, 180, 50, 255), (60, 200, 255, 255), (255, 130, 80, 255)])
                        draw.rectangle([wx, wy, wx + 5, wy + 6], fill=wc)
            # Neon roof accent
            nc = _rnd.choice(neon_colors)
            draw.line([bx, by, bx + bw - 3, by], fill=(*nc, 255), width=3)
            # Neon sign on some buildings
            if _rnd.random() > 0.5 and bw > 55:
                sign_y = by + int(bh * 0.3)
                sign_c = _rnd.choice(neon_colors)
                draw.rectangle([bx + 8, sign_y, bx + bw - 10, sign_y + 14], fill=(*sign_c, 180))
                draw.rectangle([bx + 10, sign_y + 2, bx + bw - 12, sign_y + 12], fill=(10, 8, 25, 220))
            # Antenna / tower on tall buildings
            if bh > height * 0.35:
                ax = bx + bw // 2
                draw.line([ax, by - 18, ax, by], fill=(60, 55, 80, 255), width=2)
                draw.rectangle([ax - 2, by - 20, ax + 2, by - 16], fill=(255, 30, 30, 255))
            bx += bw + _rnd.randint(3, 12)

        # ── Road surface at bottom ──
        road_top = int(height * 0.72)
        # Asphalt gradient (lighter at top edge, darker at bottom)
        for y in range(road_top, height):
            t = (y - road_top) / max(1, height - road_top)
            shade = int(42 - t * 12)
            draw.line([0, y, width, y], fill=(shade, shade, shade + 5, 255))
        # White edge line at top of road
        draw.rectangle([0, road_top, width, road_top + 3], fill=(180, 185, 195, 255))
        # Dashed yellow center line
        mid_y = road_top + (height - road_top) // 2
        for dx in range(0, width, 48):
            draw.rectangle([dx, mid_y - 2, dx + 26, mid_y + 2], fill=(255, 215, 30, 255))
        # Bottom kerb
        draw.rectangle([0, height - 5, width, height], fill=(25, 22, 35, 255))

    elif is_fighting:
        # ═══ FIGHTING: Martial Arts Dojo Arena ═══════════════════════════
        # Deep crimson-to-dark gradient wall
        for y in range(height):
            t = y / height
            r = int(55 + t * 30)
            g = int(12 + t * 15)
            b = int(15 + t * 20)
            draw.line([0, y, width, y], fill=(r, g, b, 255))

        # Wooden floor boards at bottom
        floor_top = int(height * 0.72)
        for y in range(floor_top, height):
            t = (y - floor_top) / max(1, height - floor_top)
            r = int(85 + t * 25)
            g = int(55 + t * 15)
            b = int(25 + t * 10)
            draw.line([0, y, width, y], fill=(r, g, b, 255))
        # Plank lines
        for px_off in range(0, width, 80):
            draw.line([px_off, floor_top, px_off, height], fill=(55, 35, 18, 255), width=1)
        draw.line([0, floor_top, width, floor_top], fill=(40, 25, 12, 255), width=3)

        # Wooden pillars on left and right
        for px_pos in [int(width * 0.05), int(width * 0.92)]:
            pw = int(width * 0.04)
            draw.rectangle([px_pos, 0, px_pos + pw, height], fill=(70, 40, 20, 255))
            draw.rectangle([px_pos + 2, 0, px_pos + pw // 3, height], fill=(95, 58, 28, 255))
            # Capital at top
            draw.rectangle([px_pos - 8, 0, px_pos + pw + 8, 20], fill=(85, 50, 22, 255))
            draw.rectangle([px_pos - 4, 20, px_pos + pw + 4, 28], fill=(75, 42, 18, 255))

        # Gold circular dragon medallion in center
        cx, cy = width // 2, int(height * 0.32)
        rad = int(min(width, height) * 0.14)
        # Outer gold ring
        for ring in range(3):
            draw.ellipse([cx - rad - ring, cy - rad - ring, cx + rad + ring, cy + rad + ring],
                         outline=(210, 170, 40, 255), width=3)
        # Inner dark circle
        draw.ellipse([cx - rad + 6, cy - rad + 6, cx + rad - 6, cy + rad - 6], fill=(35, 12, 12, 255))
        # Dragon pattern (simplified circular dragon)
        _rnd.seed(88)
        for angle_step in range(0, 360, 15):
            a = math.radians(angle_step)
            ir = rad * 0.35 + math.sin(a * 3) * rad * 0.12
            ex = cx + int(math.cos(a) * ir)
            ey = cy + int(math.sin(a) * ir)
            draw.ellipse([ex - 4, ey - 4, ex + 4, ey + 4], fill=(220, 180, 50, 255))
        # Central character symbol
        draw.rectangle([cx - 12, cy - 16, cx + 12, cy + 16], fill=(220, 180, 50, 255))
        draw.rectangle([cx - 8, cy - 12, cx + 8, cy + 12], fill=(35, 12, 12, 255))
        draw.line([cx - 6, cy, cx + 6, cy], fill=(220, 180, 50, 255), width=2)
        draw.line([cx, cy - 8, cx, cy + 8], fill=(220, 180, 50, 255), width=2)

        # Hanging red lanterns
        _rnd.seed(55)
        for lx in [int(width * 0.20), int(width * 0.40), int(width * 0.60), int(width * 0.80)]:
            ly = int(height * 0.06)
            # String
            draw.line([lx, 0, lx, ly], fill=(60, 30, 15, 255), width=2)
            # Lantern body
            draw.ellipse([lx - 16, ly, lx + 16, ly + 40], fill=(210, 30, 25, 255))
            draw.ellipse([lx - 12, ly + 4, lx + 12, ly + 36], fill=(240, 60, 45, 255))
            # Gold rim
            draw.rectangle([lx - 14, ly + 2, lx + 14, ly + 6], fill=(220, 180, 50, 255))
            draw.rectangle([lx - 14, ly + 34, lx + 14, ly + 38], fill=(220, 180, 50, 255))
            # Inner glow
            draw.ellipse([lx - 6, ly + 12, lx + 6, ly + 28], fill=(255, 200, 80, 200))
            # Tassel
            draw.line([lx, ly + 40, lx, ly + 52], fill=(210, 30, 25, 255), width=2)
            draw.polygon([(lx - 4, ly + 52), (lx + 4, ly + 52), (lx, ly + 60)], fill=(180, 25, 20, 255))

        # Decorative wall scrolls on sides
        for sx_pos in [int(width * 0.14), int(width * 0.82)]:
            sw, sh = 30, int(height * 0.35)
            sy = int(height * 0.15)
            draw.rectangle([sx_pos, sy, sx_pos + sw, sy + sh], fill=(240, 230, 200, 255))
            draw.rectangle([sx_pos + 2, sy + 2, sx_pos + sw - 2, sy + sh - 2], fill=(250, 242, 220, 255))
            # Brush stroke pattern
            for sdy in range(10, sh - 10, 16):
                draw.line([sx_pos + 8, sy + sdy, sx_pos + sw - 8, sy + sdy + 8], fill=(30, 20, 15, 180), width=2)

    elif is_shooter:
        # ═══ SHOOTER: Deep Space Nebula Field ════════════════════════════
        # Pure black base
        draw.rectangle([0, 0, width, height], fill=(3, 2, 10, 255))

        # Nebula gas clouds (layered, soft colored ellipses)
        _rnd.seed(303)
        nebula_colors = [(80, 20, 120), (20, 60, 130), (120, 30, 60), (30, 80, 60), (100, 40, 90)]
        for _ in range(12):
            nx = _rnd.randint(-100, width + 100)
            ny = _rnd.randint(-50, height + 50)
            nw = _rnd.randint(100, 300)
            nh = _rnd.randint(60, 180)
            nc = _rnd.choice(nebula_colors)
            for layer in range(4, 0, -1):
                alpha = int(15 + layer * 8)
                expand = layer * 20
                draw.ellipse([nx - expand, ny - expand, nx + nw + expand, ny + nh + expand],
                             fill=(*nc, alpha))

        # Dense starfield with varying sizes and colors
        _rnd.seed(404)
        for _ in range(250):
            sx = _rnd.randint(0, width)
            sy = _rnd.randint(0, height)
            br = _rnd.randint(100, 255)
            sz = _rnd.choice([1, 1, 1, 1, 2, 2, 3])
            sc = _rnd.choice([(br, br, br), (br, br - 30, br - 60), (br - 50, br - 20, br), (br, br - 10, br - 40)])
            draw.rectangle([sx, sy, sx + sz - 1, sy + sz - 1], fill=(*sc, 255))

        # Distant planet with ring
        px_c, py_c = int(width * 0.25), int(height * 0.30)
        pr = 45
        draw.ellipse([px_c - pr, py_c - pr, px_c + pr, py_c + pr], fill=(40, 55, 80, 255))
        draw.ellipse([px_c - pr + 8, py_c - pr + 5, px_c + pr - 15, py_c + pr - 8], fill=(55, 75, 100, 255))
        # Ring
        draw.arc([px_c - pr * 2, py_c - 12, px_c + pr * 2, py_c + 12], 0, 180, fill=(120, 140, 170, 200), width=3)

    else:
        # ═══ DEFAULT: Enchanted Twilight Forest with Mountains ════════════
        # Twilight sky gradient (deep blue to purple to warm horizon)
        for y in range(height):
            t = y / height
            r = int(15 + t * 60)
            g = int(10 + t * 35)
            b = int(45 + t * 25)
            if t > 0.4:
                r = int(15 + (t - 0.4) * 120)
                g = int(10 + (t - 0.4) * 50)
            draw.line([0, y, width, y], fill=(min(r, 80), min(g, 50), b, 255))

        # Stars in the upper sky
        _rnd.seed(500)
        for _ in range(60):
            sx = _rnd.randint(0, width)
            sy = _rnd.randint(0, int(height * 0.4))
            br = _rnd.randint(160, 255)
            draw.rectangle([sx, sy, sx + 1, sy + 1], fill=(br, br, br - 20, 255))

        # Northern lights / aurora effect
        for band in range(3):
            _rnd.seed(600 + band)
            by_base = int(height * (0.12 + band * 0.08))
            aurora_c = [(30, 180, 120, 35), (60, 140, 200, 30), (120, 60, 180, 25)][band]
            for seg in range(0, width, 8):
                wave = int(math.sin(seg * 0.015 + band) * 20)
                draw.rectangle([seg, by_base + wave, seg + 7, by_base + wave + 15], fill=aurora_c)

        # Distant mountain range (dark silhouette)
        mtn_base = int(height * 0.50)
        _rnd.seed(700)
        points = [(0, mtn_base)]
        x_pos = 0
        while x_pos < width + 50:
            peak_h = _rnd.randint(int(height * 0.22), int(height * 0.42))
            points.append((x_pos, peak_h))
            x_pos += _rnd.randint(60, 140)
        points.append((width, mtn_base))
        points.append((width, height))
        points.append((0, height))
        draw.polygon(points, fill=(20, 18, 35, 255))

        # Closer mountain layer (slightly lighter)
        mtn_base2 = int(height * 0.58)
        _rnd.seed(710)
        points2 = [(0, mtn_base2)]
        x_pos = 0
        while x_pos < width + 50:
            peak_h = _rnd.randint(int(height * 0.35), int(height * 0.52))
            points2.append((x_pos, peak_h))
            x_pos += _rnd.randint(50, 120)
        points2.append((width, mtn_base2))
        points2.append((width, height))
        points2.append((0, height))
        draw.polygon(points2, fill=(28, 25, 42, 255))

        # Pine tree silhouettes
        tree_base = int(height * 0.62)
        _rnd.seed(800)
        for _ in range(25):
            tx = _rnd.randint(0, width)
            th = _rnd.randint(40, 90)
            tw = _rnd.randint(20, 40)
            # Trunk
            draw.rectangle([tx - 3, tree_base - 5, tx + 3, tree_base + 10], fill=(18, 15, 25, 255))
            # Foliage triangles
            for layer in range(3):
                ly = tree_base - 5 - layer * int(th * 0.3)
                lw = tw - layer * 8
                draw.polygon([(tx - lw, ly), (tx + lw, ly), (tx, ly - int(th * 0.35))], fill=(15, 18, 28, 255))

        # Ground with grass tufts
        draw.rectangle([0, tree_base + 5, width, height], fill=(22, 28, 18, 255))
        for gx in range(0, width, 6):
            gh = _rnd.randint(3, 10)
            draw.line([gx, tree_base + 5, gx, tree_base + 5 - gh], fill=(18, 35, 15, 255), width=2)

    return img


def generate_procedural_sprite(asset_name, prompt, width=512, height=512, seed=42, game_plan=None):
    random.seed(seed)
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    px = lambda f: int(width * f)
    py = lambda f: int(height * f)

    prompt_lower = (prompt or "").lower()
    genre_lower = (game_plan.get("genre", "") if game_plan else "").lower()
    theme_lower = (game_plan.get("theme", "") if game_plan else "").lower()

    is_racing    = "rac" in genre_lower or "car" in genre_lower
    is_fighting  = "fight" in genre_lower or "arena" in genre_lower
    is_adventure = "adventure" in genre_lower
    is_dungeon   = "dungeon" in genre_lower
    is_strategy  = "strategy" in genre_lower
    is_td        = "tower" in genre_lower or "defense" in genre_lower
    is_running   = "running" in genre_lower or "runner" in genre_lower

    # ─────────────────────────────────────────────────────────────
    # PLAYER sprites per genre
    # ─────────────────────────────────────────────────────────────
    if asset_name == "player":
        if is_racing:
            # Sleek aerodynamic supercar with proper gradients and chrome rim detail
            # Shadow
            draw.polygon([(px(.06),py(.50)),(px(.18),py(.30)),(px(.82),py(.30)),(px(.94),py(.50)),(px(.94),py(.80)),(px(.06),py(.80))], fill=(15,10,12,255))
            # Main body
            draw.polygon([(px(.08),py(.48)),(px(.20),py(.32)),(px(.80),py(.32)),(px(.92),py(.48)),(px(.92),py(.78)),(px(.08),py(.78))], fill=(210,20,20,255))
            # Body top highlight
            draw.polygon([(px(.10),py(.46)),(px(.22),py(.34)),(px(.78),py(.34)),(px(.90),py(.46))], fill=(255,80,80,255))
            # Roof reflection
            draw.polygon([(px(.28),py(.34)),(px(.38),py(.28)),(px(.62),py(.28)),(px(.72),py(.34))], fill=(255,120,120,180))
            # Windshield (deep dark glass with blue tint)
            draw.polygon([(px(.28),py(.34)),(px(.40),py(.18)),(px(.68),py(.18)),(px(.76),py(.34))], fill=(15,22,40,255))
            draw.polygon([(px(.31),py(.32)),(px(.42),py(.20)),(px(.66),py(.20)),(px(.73),py(.32))], fill=(100,180,240,100))
            # Side window
            draw.polygon([(px(.76),py(.34)),(px(.86),py(.34)),(px(.88),py(.40)),(px(.78),py(.40))], fill=(80,150,210,100))
            # Rear spoiler
            draw.rectangle([px(.04),py(.24),px(.20),py(.32)], fill=(150,10,10,255))
            draw.rectangle([px(.04),py(.22),px(.20),py(.26)], fill=(80,5,5,255))
            draw.line([px(.12),py(.32),px(.12),py(.48)], fill=(10,8,10,255), width=px(.025))
            # Gold racing stripe
            draw.rectangle([px(.08),py(.52),px(.92),py(.58)], fill=(255,210,0,255))
            draw.rectangle([px(.08),py(.56),px(.92),py(.59)], fill=(230,180,0,255))
            # Front bumper air intake
            draw.rectangle([px(.78),py(.60),px(.94),py(.72)], fill=(12,12,16,255))
            draw.rectangle([px(.80),py(.62),px(.92),py(.70)], fill=(25,25,30,255))
            # Exhaust pipes
            draw.ellipse([px(.04),py(.60),px(.10),py(.68)], fill=(40,40,50,255))
            draw.ellipse([px(.04),py(.68),px(.10),py(.76)], fill=(40,40,50,255))
            # Front wheels (detailed)
            draw.ellipse([px(.60),py(.58),px(.88),py(.90)], fill=(14,14,18,255))
            draw.ellipse([px(.64),py(.62),px(.84),py(.86)], fill=(35,35,45,255))
            draw.ellipse([px(.68),py(.66),px(.80),py(.82)], fill=(200,205,220,255))
            draw.ellipse([px(.71),py(.69),px(.77),py(.79)], fill=(14,14,18,255))
            for spoke_a in range(0, 360, 60):
                sx = int((px(.68)+px(.80))//2 + math.cos(math.radians(spoke_a)) * px(.055))
                sy = int((py(.66)+py(.82))//2 + math.sin(math.radians(spoke_a)) * py(.055))
                draw.line([(px(.68)+px(.80))//2,(py(.66)+py(.82))//2,sx,sy], fill=(170,175,195,255), width=2)
            # Rear wheels
            draw.ellipse([px(.12),py(.58),px(.40),py(.90)], fill=(14,14,18,255))
            draw.ellipse([px(.16),py(.62),px(.36),py(.86)], fill=(35,35,45,255))
            draw.ellipse([px(.20),py(.66),px(.32),py(.82)], fill=(200,205,220,255))
            draw.ellipse([px(.23),py(.69),px(.29),py(.79)], fill=(14,14,18,255))
            # Headlights (LED glow)
            draw.rectangle([px(.88),py(.42),px(.94),py(.50)], fill=(255,255,200,255))
            draw.rectangle([px(.89),py(.43),px(.93),py(.49)], fill=(255,255,255,255))
            # Tail lights
            draw.rectangle([px(.06),py(.46),px(.12),py(.56)], fill=(220,30,30,255))

        elif is_fighting:
            # Muscular fighter in gi uniform - Street Fighter style
            # Outline
            draw.ellipse([px(.28),py(.04),px(.72),py(.34)], fill=(20,15,10,255))
            draw.rectangle([px(.18),py(.26),px(.82),py(.70)], fill=(20,15,10,255))
            draw.rectangle([px(.20),py(.66),px(.44),py(.98)], fill=(20,15,10,255))
            draw.rectangle([px(.56),py(.66),px(.80),py(.98)], fill=(20,15,10,255))
            # Head with skin tone and hair
            draw.ellipse([px(.30),py(.06),px(.70),py(.32)], fill=(220,170,120,255))
            draw.ellipse([px(.32),py(.08),px(.52),py(.26)], fill=(240,190,140,255))  # highlight
            # Black hair
            draw.ellipse([px(.30),py(.06),px(.70),py(.18)], fill=(25,20,18,255))
            draw.polygon([(px(.30),py(.16)),(px(.25),py(.22)),(px(.34),py(.22))], fill=(25,20,18,255))
            # Eyes and face
            draw.ellipse([px(.38),py(.18),px(.46),py(.24)], fill=(40,30,20,255))
            draw.ellipse([px(.54),py(.18),px(.62),py(.24)], fill=(40,30,20,255))
            draw.arc([px(.42),py(.26),px(.58),py(.30)], 0, 180, fill=(180,100,80,255), width=2)
            # Headband
            draw.rectangle([px(.28),py(.14),px(.72),py(.20)], fill=(220,30,30,255))
            draw.polygon([(px(.70),py(.15)),(px(.78),py(.25)),(px(.68),py(.22))], fill=(220,30,30,255))
            # White gi body
            draw.rectangle([px(.20),py(.28),px(.80),py(.68)], fill=(240,238,232,255))
            draw.rectangle([px(.22),py(.30),px(.46),py(.66)], fill=(255,255,252,255))  # left panel highlight
            # Red gi lapels
            draw.polygon([(px(.40),py(.28)),(px(.50),py(.28)),(px(.50),py(.68)),(px(.40),py(.68))], fill=(200,30,30,255))
            draw.polygon([(px(.50),py(.28)),(px(.60),py(.28)),(px(.60),py(.68)),(px(.50),py(.68))], fill=(200,30,30,255))
            # Black belt
            draw.rectangle([px(.20),py(.62),px(.80),py(.68)], fill=(20,18,18,255))
            draw.rectangle([px(.44),py(.62),px(.56),py(.68)], fill=(35,30,25,255))  # belt knot
            # Arms (punching pose)
            draw.rectangle([px(.04),py(.28),px(.22),py(.50)], fill=(220,170,120,255))  # right arm extended
            draw.rectangle([px(.78),py(.38),px(.94),py(.56)], fill=(220,170,120,255))  # left arm guard
            draw.rectangle([px(.04),py(.42),px(.20),py(.52)], fill=(240,190,140,255))  # fist highlight
            # Legs
            draw.rectangle([px(.22),py(.68),px(.42),py(.96)], fill=(240,238,232,255))
            draw.rectangle([px(.58),py(.68),px(.78),py(.96)], fill=(240,238,232,255))
            draw.rectangle([px(.22),py(.88),px(.42),py(.98)], fill=(40,35,30,255))  # shoes
            draw.rectangle([px(.58),py(.88),px(.78),py(.98)], fill=(40,35,30,255))

        elif is_adventure or is_dungeon:
            # Fantasy hero - detailed knight/adventurer
            # Outline shadow layer
            draw.ellipse([px(.28),py(.04),px(.72),py(.36)], fill=(18,16,22,255))
            draw.polygon([(px(.16),py(.28)),(px(.84),py(.28)),(px(.88),py(.80)),(px(.12),py(.80))], fill=(18,16,22,255))
            draw.rectangle([px(.22),py(.76),px(.78),py(.98)], fill=(18,16,22,255))
            # Blue magic cape behind body
            draw.polygon([(px(.14),py(.32)),(px(.28),py(.32)),(px(.18),py(.86)),(px(.08),py(.78))], fill=(30,60,160,255))
            draw.polygon([(px(.72),py(.32)),(px(.86),py(.32)),(px(.92),py(.78)),(px(.82),py(.86))], fill=(30,60,160,255))
            draw.polygon([(px(.16),py(.32)),(px(.28),py(.32)),(px(.20),py(.82)),(px(.10),py(.76))], fill=(60,100,210,255))  # Cape highlight
            # Helmet - full visor plate
            draw.ellipse([px(.30),py(.06),px(.70),py(.34)], fill=(175,188,205,255))
            draw.ellipse([px(.32),py(.08),px(.52),py(.30)], fill=(215,228,242,255))  # Helmet highlight
            draw.rectangle([px(.34),py(.18),px(.66),py(.26)], fill=(28,28,36,255))  # visor slit
            draw.rectangle([px(.46),py(.18),px(.54),py(.26)], fill=(255,200,40,255))  # Gold center visor
            draw.polygon([(px(.42),py(.02)),(px(.50),py(-.04)),(px(.58),py(.02)),(px(.52),py(.10)),(px(.48),py(.10))], fill=(210,40,40,255))  # Red plume
            draw.polygon([(px(.44),py(.04)),(px(.50),py(-.02)),(px(.56),py(.04)),(px(.52),py(.10)),(px(.48),py(.10))], fill=(255,80,80,255))  # Plume highlight
            # Pauldrons
            draw.ellipse([px(.14),py(.26),px(.36),py(.48)], fill=(160,172,188,255))
            draw.ellipse([px(.16),py(.28),px(.30),py(.42)], fill=(210,222,238,255))  # Highlight
            draw.ellipse([px(.64),py(.26),px(.86),py(.48)], fill=(160,172,188,255))
            draw.ellipse([px(.70),py(.28),px(.84),py(.42)], fill=(210,222,238,255))
            # Chest plate
            draw.rectangle([px(.28),py(.30),px(.72),py(.68)], fill=(155,168,185,255))
            draw.rectangle([px(.30),py(.32),px(.52),py(.66)], fill=(200,215,232,255))  # Chest highlight
            draw.polygon([(px(.36),py(.36)),(px(.50),py(.42)),(px(.64),py(.36))], fill=(230,190,40,255))  # Gold emblem
            draw.polygon([(px(.44),py(.36)),(px(.50),py(.50)),(px(.56),py(.36))], fill=(200,162,30,255))
            # Belt
            draw.rectangle([px(.26),py(.64),px(.74),py(.70)], fill=(90,48,18,255))
            draw.rectangle([px(.44),py(.63),px(.56),py(.70)], fill=(230,190,40,255))  # Buckle
            # Greaves
            draw.rectangle([px(.28),py(.70),px(.46),py(.90)], fill=(145,158,175,255))
            draw.rectangle([px(.54),py(.70),px(.72),py(.90)], fill=(145,158,175,255))
            draw.rectangle([px(.24),py(.88),px(.46),py(.98)], fill=(68,36,16,255))  # Boots
            draw.rectangle([px(.54),py(.88),px(.76),py(.98)], fill=(68,36,16,255))
            # Broadsword
            draw.polygon([(px(.60),py(.40)),(px(.98),py(.82)),(px(.95),py(.86)),(px(.57),py(.44))], fill=(210,222,238,255))
            draw.line([px(.58),py(.42),px(.96),py(.84)], fill=(255,255,255,200), width=2)
            draw.rectangle([px(.54),py(.36),px(.66),py(.42)], fill=(230,190,40,255))  # Crossguard
            draw.ellipse([px(.50),py(.34),px(.58),py(.40)], fill=(230,190,40,255))  # Pommel

        elif is_strategy:
            # Commander unit - RTS style
            draw.ellipse([px(.30),py(.04),px(.70),py(.32)], fill=(15,20,15,255))
            draw.rectangle([px(.20),py(.26),px(.80),py(.70)], fill=(15,20,15,255))
            # Military cap
            draw.ellipse([px(.28),py(.06),px(.72),py(.28)], fill=(40,60,35,255))
            draw.rectangle([px(.22),py(.18),px(.78),py(.26)], fill=(30,48,25,255))
            draw.rectangle([px(.44),py(.08),px(.56),py(.14)], fill=(255,200,0,255))  # Gold star badge
            # Face
            draw.ellipse([px(.32),py(.20),px(.68),py(.44)], fill=(200,160,110,255))
            draw.ellipse([px(.34),py(.22),px(.52),py(.40)], fill=(220,180,130,255))
            # Eyes
            draw.ellipse([px(.38),py(.28),px(.46),py(.34)], fill=(35,30,25,255))
            draw.ellipse([px(.54),py(.28),px(.62),py(.34)], fill=(35,30,25,255))
            # Military uniform (green camo)
            draw.rectangle([px(.20),py(.38),px(.80),py(.72)], fill=(55,75,45,255))
            draw.rectangle([px(.22),py(.40),px(.50),py(.70)], fill=(70,95,55,255))  # highlight
            # Gold epaulettes
            draw.rectangle([px(.18),py(.38),px(.26),py(.48)], fill=(200,160,30,255))
            draw.rectangle([px(.74),py(.38),px(.82),py(.48)], fill=(200,160,30,255))
            # Medal ribbons
            for my_i, mc in enumerate([(200,30,30),(30,80,200),(200,200,30),(30,160,80)]):
                draw.rectangle([px(.30)+my_i*px(.06), py(.52), px(.34)+my_i*px(.06), py(.60)], fill=(*mc,255))
            # Legs
            draw.rectangle([px(.26),py(.72),px(.44),py(.96)], fill=(40,55,32,255))
            draw.rectangle([px(.56),py(.72),px(.74),py(.96)], fill=(40,55,32,255))
            draw.rectangle([px(.24),py(.90),px(.46),py(.98)], fill=(30,28,25,255))
            draw.rectangle([px(.54),py(.90),px(.76),py(.98)], fill=(30,28,25,255))

        elif is_td:
            # Tower defense cannon tower
            draw.rectangle([px(.20),py(.45),px(.80),py(.90)], fill=(18,18,22,255))
            draw.rectangle([px(.22),py(.47),px(.78),py(.88)], fill=(55,60,75,255))
            draw.rectangle([px(.24),py(.49),px(.50),py(.86)], fill=(75,82,100,255))  # highlight
            # Tower battlements
            for bx in [px(.20),px(.34),px(.48),px(.62),px(.76)]:
                draw.rectangle([bx,py(.30),bx+px(.10),py(.47)], fill=(50,55,70,255))
            # Cannon barrel
            draw.rectangle([px(.42),py(.35),px(.90),py(.52)], fill=(22,22,28,255))
            draw.rectangle([px(.44),py(.37),px(.88),py(.50)], fill=(45,45,58,255))
            draw.ellipse([px(.84),py(.34),px(.92),py(.52)], fill=(22,22,28,255))
            # Gold rim on barrel
            draw.rectangle([px(.60),py(.35),px(.66),py(.52)], fill=(180,150,30,255))
            draw.rectangle([px(.80),py(.35),px(.86),py(.52)], fill=(180,150,30,255))
            # Base
            draw.rectangle([px(.14),py(.88),px(.86),py(.98)], fill=(40,40,50,255))

        elif is_running:
            # Track runner - athlete in motion
            draw.ellipse([px(.35),py(.04),px(.65),py(.28)], fill=(18,15,12,255))
            draw.rectangle([px(.30),py(.22),px(.70),py(.58)], fill=(18,15,12,255))
            # Head
            draw.ellipse([px(.36),py(.06),px(.64),py(.26)], fill=(210,165,110,255))
            draw.ellipse([px(.38),py(.08),px(.56),py(.22)], fill=(230,185,130,255))
            # Hair
            draw.ellipse([px(.36),py(.06),px(.64),py(.16)], fill=(20,16,12,255))
            # Face
            draw.ellipse([px(.43),py(.14),px(.50),py(.19)], fill=(30,22,18,255))
            draw.ellipse([px(.53),py(.14),px(.60),py(.19)], fill=(30,22,18,255))
            # Athletic jersey (bright)
            draw.rectangle([px(.28),py(.24),px(.72),py(.58)], fill=(255,80,0,255))
            draw.rectangle([px(.30),py(.26),px(.52),py(.56)], fill=(255,120,30,255))
            # Race number
            draw.rectangle([px(.38),py(.34),px(.62),py(.52)], fill=(255,255,255,220))
            # Arms in running motion
            draw.rectangle([px(.06),py(.30),px(.30),py(.44)], fill=(210,165,110,255))  # front arm
            draw.rectangle([px(.70),py(.38),px(.90),py(.50)], fill=(210,165,110,255))  # back arm
            # Shorts
            draw.rectangle([px(.28),py(.56),px(.72),py(.68)], fill=(30,30,180,255))
            # Legs (running pose)
            draw.polygon([(px(.30),py(.66)),(px(.42),py(.66)),(px(.36),py(.96)),(px(.24),py(.90))], fill=(210,165,110,255))
            draw.polygon([(px(.58),py(.66)),(px(.70),py(.66)),(px(.78),py(.88)),(px(.64),py(.96))], fill=(210,165,110,255))
            # Running shoes
            draw.polygon([(px(.22),py(.88)),(px(.38),py(.88)),(px(.38),py(.96)),(px(.18),py(.96))], fill=(255,60,0,255))
            draw.polygon([(px(.62),py(.92)),(px(.80),py(.88)),(px(.82),py(.96)),(px(.60),py(.98))], fill=(255,60,0,255))
            # White soles
            draw.rectangle([px(.18),py(.94),px(.40),py(.98)], fill=(255,255,255,255))
            draw.rectangle([px(.60),py(.94),px(.84),py(.98)], fill=(255,255,255,255))

        else:
            # DEFAULT - Armored knight with broadsword (highly detailed)
            draw.ellipse([px(.28),py(.04),px(.72),py(.36)], fill=(18,16,22,255))
            draw.polygon([(px(.16),py(.28)),(px(.84),py(.28)),(px(.88),py(.80)),(px(.12),py(.80))], fill=(18,16,22,255))
            draw.rectangle([px(.22),py(.76),px(.78),py(.98)], fill=(18,16,22,255))
            draw.polygon([(px(.14),py(.32)),(px(.28),py(.32)),(px(.18),py(.86)),(px(.08),py(.78))], fill=(30,60,160,255))
            draw.polygon([(px(.72),py(.32)),(px(.86),py(.32)),(px(.92),py(.78)),(px(.82),py(.86))], fill=(30,60,160,255))
            draw.ellipse([px(.30),py(.06),px(.70),py(.34)], fill=(175,188,205,255))
            draw.ellipse([px(.32),py(.08),px(.52),py(.30)], fill=(215,228,242,255))
            draw.rectangle([px(.34),py(.18),px(.66),py(.26)], fill=(28,28,36,255))
            draw.rectangle([px(.46),py(.18),px(.54),py(.26)], fill=(255,200,40,255))
            draw.polygon([(px(.42),py(.02)),(px(.50),py(-.04)),(px(.58),py(.02)),(px(.52),py(.10)),(px(.48),py(.10))], fill=(210,40,40,255))
            draw.ellipse([px(.14),py(.26),px(.36),py(.48)], fill=(160,172,188,255))
            draw.ellipse([px(.64),py(.26),px(.86),py(.48)], fill=(160,172,188,255))
            draw.rectangle([px(.28),py(.30),px(.72),py(.68)], fill=(155,168,185,255))
            draw.rectangle([px(.30),py(.32),px(.52),py(.66)], fill=(200,215,232,255))
            draw.polygon([(px(.36),py(.36)),(px(.50),py(.42)),(px(.64),py(.36))], fill=(230,190,40,255))
            draw.rectangle([px(.26),py(.64),px(.74),py(.70)], fill=(90,48,18,255))
            draw.rectangle([px(.44),py(.63),px(.56),py(.70)], fill=(230,190,40,255))
            draw.rectangle([px(.28),py(.70),px(.46),py(.90)], fill=(145,158,175,255))
            draw.rectangle([px(.54),py(.70),px(.72),py(.90)], fill=(145,158,175,255))
            draw.rectangle([px(.24),py(.88),px(.46),py(.98)], fill=(68,36,16,255))
            draw.rectangle([px(.54),py(.88),px(.76),py(.98)], fill=(68,36,16,255))
            draw.polygon([(px(.60),py(.40)),(px(.98),py(.82)),(px(.95),py(.86)),(px(.57),py(.44))], fill=(210,222,238,255))
            draw.line([px(.58),py(.42),px(.96),py(.84)], fill=(255,255,255,200), width=2)
            draw.rectangle([px(.54),py(.36),px(.66),py(.42)], fill=(230,190,40,255))
            draw.ellipse([px(.50),py(.34),px(.58),py(.40)], fill=(230,190,40,255))

    # ─────────────────────────────────────────────────────────────
    # ENEMY sprites per genre
    # ─────────────────────────────────────────────────────────────
    elif asset_name == "enemy":
        if is_racing:
            # Blue rival supercar
            draw.polygon([(px(.06),py(.50)),(px(.18),py(.30)),(px(.82),py(.30)),(px(.94),py(.50)),(px(.94),py(.80)),(px(.06),py(.80))], fill=(10,12,22,255))
            draw.polygon([(px(.08),py(.48)),(px(.20),py(.32)),(px(.80),py(.32)),(px(.92),py(.48)),(px(.92),py(.78)),(px(.08),py(.78))], fill=(20,50,210,255))
            draw.polygon([(px(.10),py(.46)),(px(.22),py(.34)),(px(.78),py(.34)),(px(.90),py(.46))], fill=(60,110,255,255))
            draw.polygon([(px(.30),py(.34)),(px(.40),py(.18)),(px(.68),py(.18)),(px(.76),py(.34))], fill=(12,18,36,255))
            draw.polygon([(px(.32),py(.32)),(px(.42),py(.20)),(px(.66),py(.20)),(px(.73),py(.32))], fill=(80,150,240,100))
            draw.rectangle([px(.04),py(.24),px(.20),py(.32)], fill=(10,30,140,255))
            draw.rectangle([px(.08),py(.52),px(.92),py(.58)], fill=(0,220,255,255))  # Cyan stripe
            # Wheels
            draw.ellipse([px(.60),py(.58),px(.88),py(.90)], fill=(14,14,18,255))
            draw.ellipse([px(.64),py(.62),px(.84),py(.86)], fill=(35,35,45,255))
            draw.ellipse([px(.68),py(.66),px(.80),py(.82)], fill=(200,205,220,255))
            draw.ellipse([px(.12),py(.58),px(.40),py(.90)], fill=(14,14,18,255))
            draw.ellipse([px(.16),py(.62),px(.36),py(.86)], fill=(35,35,45,255))
            draw.ellipse([px(.20),py(.66),px(.32),py(.82)], fill=(200,205,220,255))
            draw.rectangle([px(.88),py(.42),px(.94),py(.52)], fill=(255,140,0,255))  # Headlight orange

        elif is_fighting:
            # Cyber ninja villain
            draw.ellipse([px(.28),py(.04),px(.72),py(.34)], fill=(15,15,20,255))
            draw.rectangle([px(.18),py(.26),px(.82),py(.70)], fill=(15,15,20,255))
            draw.rectangle([px(.20),py(.66),px(.44),py(.98)], fill=(15,15,20,255))
            draw.rectangle([px(.56),py(.66),px(.80),py(.98)], fill=(15,15,20,255))
            # Head with mask
            draw.ellipse([px(.30),py(.06),px(.70),py(.32)], fill=(28,28,35,255))
            draw.ellipse([px(.32),py(.08),px(.52),py(.28)], fill=(45,45,58,255))
            # Red visor eyes
            draw.rectangle([px(.32),py(.16),px(.68),py(.22)], fill=(210,20,20,255))
            draw.rectangle([px(.34),py(.17),px(.66),py(.21)], fill=(255,60,60,255))
            # Dark bodysuit with glowing purple circuits
            draw.rectangle([px(.20),py(.28),px(.80),py(.68)], fill=(22,22,32,255))
            draw.rectangle([px(.22),py(.30),px(.46),py(.66)], fill=(32,32,45,255))
            # Circuit lines
            for cy_off in [py(.36),py(.48),py(.60)]:
                draw.line([px(.22),cy_off,px(.78),cy_off], fill=(140,0,255,180), width=2)
            for cx_off in [px(.30),px(.50),px(.70)]:
                draw.line([cx_off,py(.30),cx_off,py(.66)], fill=(140,0,255,120), width=1)
            # Glowing purple shoulder pads
            draw.ellipse([px(.14),py(.26),px(.30),py(.42)], fill=(80,0,180,255))
            draw.ellipse([px(.16),py(.28),px(.26),py(.38)], fill=(130,0,240,255))
            draw.ellipse([px(.70),py(.26),px(.86),py(.42)], fill=(80,0,180,255))
            draw.ellipse([px(.74),py(.28),px(.84),py(.38)], fill=(130,0,240,255))
            # Legs
            draw.rectangle([px(.22),py(.68),px(.42),py(.96)], fill=(20,20,30,255))
            draw.rectangle([px(.58),py(.68),px(.78),py(.96)], fill=(20,20,30,255))
            draw.rectangle([px(.20),py(.90),px(.44),py(.98)], fill=(15,15,22,255))
            draw.rectangle([px(.56),py(.90),px(.80),py(.98)], fill=(15,15,22,255))

        elif is_adventure or is_dungeon:
            # Ferocious dragon enemy
            draw.ellipse([px(.12),py(.16),px(.88),py(.88)], fill=(12,22,12,255))
            draw.polygon([(px(.58),py(.65)),(px(.96),py(.65)),(px(.94),py(.90)),(px(.56),py(.90))], fill=(12,22,12,255))
            # Tail
            draw.polygon([(px(.60),py(.68)),(px(.92),py(.68)),(px(.90),py(.86)),(px(.58),py(.86))], fill=(38,135,48,255))
            draw.polygon([(px(.84),py(.68)),(px(.95),py(.68)),(px(.93),py(.86))], fill=(130,25,25,255))
            # Body
            draw.ellipse([px(.20),py(.32),px(.74),py(.84)], fill=(42,150,52,255))
            draw.ellipse([px(.22),py(.34),px(.48),py(.80)], fill=(70,188,80,255))  # highlight
            # Scales pattern
            for scale_y in [0.48,0.58,0.68]:
                for scale_x in [0.32,0.42,0.52]:
                    draw.ellipse([px(scale_x-0.04),py(scale_y-0.04),px(scale_x+0.04),py(scale_y+0.04)], outline=(28,108,38,255), width=2)
            # Yellow underbelly
            draw.ellipse([px(.30),py(.46),px(.56),py(.80)], fill=(242,210,105,255))
            for seg_y in [0.53,0.61,0.70]:
                draw.line([px(.32),py(seg_y),px(.54),py(seg_y)], fill=(196,162,58,255), width=max(2,px(.012)))
            # Head
            draw.ellipse([px(.14),py(.14),px(.56),py(.50)], fill=(42,150,52,255))
            draw.ellipse([px(.16),py(.16),px(.38),py(.46)], fill=(78,198,90,255))
            # Horns
            draw.polygon([(px(.28),py(.18)),(px(.34),py(.02)),(px(.40),py(.18))], fill=(232,232,244,255))
            draw.polygon([(px(.40),py(.20)),(px(.46),py(.04)),(px(.52),py(.20))], fill=(232,232,244,255))
            # Eye - glowing
            draw.ellipse([px(.26),py(.24),px(.38),py(.36)], fill=(225,25,38,255))
            draw.ellipse([px(.29),py(.27),px(.35),py(.33)], fill=(255,80,80,255))
            draw.rectangle([px(.31),py(.27),px(.33),py(.33)], fill=(10,10,12,255))
            # Wings
            draw.polygon([(px(.62),py(.22)),(px(.96),py(.06)),(px(.98),py(.44)),(px(.72),py(.38))], fill=(28,108,38,255))
            draw.polygon([(px(.64),py(.24)),(px(.94),py(.08)),(px(.96),py(.40)),(px(.74),py(.36))], fill=(52,168,62,255))
            # Wing membrane veins
            draw.line([px(.65),py(.26),px(.92),py(.12)], fill=(18,78,28,255), width=2)
            draw.line([px(.67),py(.30),px(.94),py(.26)], fill=(18,78,28,255), width=2)
            draw.line([px(.68),py(.34),px(.92),py(.38)], fill=(18,78,28,255), width=2)
            # Feet and claws
            draw.ellipse([px(.22),py(.76),px(.44),py(.90)], fill=(32,120,42,255))
            for claw_x in [px(.22),px(.28),px(.34)]:
                draw.polygon([(claw_x,py(.86)),(claw_x+px(.04),py(.82)),(claw_x+px(.02),py(.92))], fill=(232,232,244,255))

        elif is_strategy:
            # Enemy base/fortress
            draw.rectangle([px(.10),py(.25),px(.90),py(.88)], fill=(55,18,18,255))
            draw.rectangle([px(.12),py(.27),px(.88),py(.86)], fill=(80,25,25,255))
            # Gate door
            draw.rectangle([px(.38),py(.55),px(.62),py(.86)], fill=(30,12,12,255))
            draw.ellipse([px(.38),py(.44),px(.62),py(.66)], fill=(30,12,12,255))
            draw.rectangle([px(.40),py(.57),px(.60),py(.86)], fill=(22,8,8,255))
            # Metal portcullis bars
            for bar_x in range(px(.41),px(.60),px(.04)):
                draw.line([bar_x,py(.46),bar_x,py(.86)], fill=(55,45,38,255), width=2)
            for bar_y in [py(.52),py(.60),py(.68),py(.76)]:
                draw.line([px(.40),bar_y,px(.60),bar_y], fill=(55,45,38,255), width=2)
            # Wall parapets
            for par_x in [px(.10),px(.22),px(.34),px(.64),px(.76)]:
                draw.rectangle([par_x,py(.10),par_x+px(.10),py(.26)], fill=(70,22,22,255))
            # Corner towers
            draw.rectangle([px(.06),py(.18),px(.18),py(.88)], fill=(65,20,20,255))
            draw.rectangle([px(.82),py(.18),px(.94),py(.88)], fill=(65,20,20,255))
            # Arrow slits
            for slit_y in [py(.35),py(.55),py(.72)]:
                draw.rectangle([px(.08),slit_y,px(.16),slit_y+py(.06)], fill=(16,6,6,255))
                draw.rectangle([px(.84),slit_y,px(.92),slit_y+py(.06)], fill=(16,6,6,255))
            # Red evil banner
            draw.rectangle([px(.46),py(.04),px(.54),py(.25)], fill=(60,25,25,255))
            draw.polygon([(px(.54),py(.08)),(px(.78),py(.14)),(px(.54),py(.20))], fill=(200,20,20,255))
            draw.ellipse([px(.62),py(.10),px(.72),py(.20)], fill=(150,10,10,255))

        elif is_td:
            # Enemy creep/minion
            draw.ellipse([px(.30),py(.06),px(.70),py(.40)], fill=(15,30,15,255))
            draw.rectangle([px(.24),py(.32),px(.76),py(.74)], fill=(15,30,15,255))
            # Slime green body
            draw.ellipse([px(.32),py(.08),px(.68),py(.38)], fill=(45,180,45,255))
            draw.ellipse([px(.34),py(.10),px(.56),py(.32)], fill=(80,220,80,255))  # highlight
            # Big angry eyes
            draw.ellipse([px(.36),py(.14),px(.48),py(.26)], fill=(250,250,250,255))
            draw.ellipse([px(.52),py(.14),px(.64),py(.26)], fill=(250,250,250,255))
            draw.ellipse([px(.39),py(.17),px(.46),py(.24)], fill=(200,20,20,255))
            draw.ellipse([px(.55),py(.17),px(.62),py(.24)], fill=(200,20,20,255))
            draw.ellipse([px(.41),py(.19),px(.44),py(.22)], fill=(15,10,10,255))
            draw.ellipse([px(.57),py(.19),px(.60),py(.22)], fill=(15,10,10,255))
            # Angry mouth
            draw.arc([px(.38),py(.28),px(.62),py(.36)], 0, 180, fill=(18,12,12,255), width=3)
            # Stubby arms
            draw.ellipse([px(.08),py(.36),px(.28),py(.56)], fill=(35,155,35,255))
            draw.ellipse([px(.72),py(.36),px(.92),py(.56)], fill=(35,155,35,255))
            # Claw hands
            for cl_x,cl_d in [(px(.08),1),(px(.74),1)]:
                for cl_i in range(3):
                    draw.polygon([(cl_x+cl_d*cl_i*px(.05),py(.50)),(cl_x+cl_d*(cl_i*px(.05)+px(.03)),py(.56)),(cl_x+cl_d*(cl_i*px(.05)+px(.02)),py(.62))], fill=(22,110,22,255))
            # Body / belly
            draw.rectangle([px(.26),py(.34),px(.74),py(.72)], fill=(38,160,38,255))
            draw.ellipse([px(.32),py(.42),px(.68),py(.68)], fill=(50,200,50,255))
            # Legs
            draw.rectangle([px(.28),py(.70),px(.44),py(.92)], fill=(28,130,28,255))
            draw.rectangle([px(.56),py(.70),px(.72),py(.92)], fill=(28,130,28,255))
            draw.rectangle([px(.24),py(.88),px(.46),py(.96)], fill=(18,90,18,255))
            draw.rectangle([px(.54),py(.88),px(.76),py(.96)], fill=(18,90,18,255))

        elif is_running:
            # Spike obstacle
            draw.rectangle([px(.10),py(.58),px(.90),py(.86)], fill=(22,22,28,255))
            draw.rectangle([px(.12),py(.60),px(.88),py(.84)], fill=(42,42,55,255))
            # Warning stripe base
            for strap_i in range(4):
                col = (220,180,0,255) if strap_i%2==0 else (18,18,24,255)
                draw.rectangle([px(.10)+strap_i*px(.20),py(.58),px(.10)+(strap_i+1)*px(.20),py(.86)], fill=col)
            # Spikes on top
            for sp_i in range(5):
                sx_base = px(.12) + sp_i * px(.16)
                draw.polygon([(sx_base,py(.58)),(sx_base+px(.08),py(.58)),(sx_base+px(.04),py(.22))], fill=(195,18,18,255))
                draw.polygon([(sx_base+px(.01),py(.58)),(sx_base+px(.07),py(.58)),(sx_base+px(.04),py(.26))], fill=(240,60,60,255))  # highlight
            # Bolt details on base
            for blt_x in [px(.18),px(.40),px(.62),px(.82)]:
                draw.ellipse([blt_x-px(.02),py(.68),blt_x+px(.02),py(.74)], fill=(155,155,175,255))

        else:
            # DEFAULT dragon enemy (same as adventure)
            draw.ellipse([px(.12),py(.16),px(.88),py(.88)], fill=(12,22,12,255))
            draw.polygon([(px(.58),py(.65)),(px(.96),py(.65)),(px(.94),py(.90)),(px(.56),py(.90))], fill=(12,22,12,255))
            draw.polygon([(px(.60),py(.68)),(px(.92),py(.68)),(px(.90),py(.86)),(px(.58),py(.86))], fill=(38,135,48,255))
            draw.polygon([(px(.84),py(.68)),(px(.95),py(.68)),(px(.93),py(.86))], fill=(130,25,25,255))
            draw.ellipse([px(.20),py(.32),px(.74),py(.84)], fill=(42,150,52,255))
            draw.ellipse([px(.22),py(.34),px(.48),py(.80)], fill=(70,188,80,255))
            draw.ellipse([px(.30),py(.46),px(.56),py(.80)], fill=(242,210,105,255))
            draw.ellipse([px(.14),py(.14),px(.56),py(.50)], fill=(42,150,52,255))
            draw.polygon([(px(.28),py(.18)),(px(.34),py(.02)),(px(.40),py(.18))], fill=(232,232,244,255))
            draw.polygon([(px(.40),py(.20)),(px(.46),py(.04)),(px(.52),py(.20))], fill=(232,232,244,255))
            draw.ellipse([px(.26),py(.24),px(.38),py(.36)], fill=(225,25,38,255))
            draw.rectangle([px(.31),py(.27),px(.33),py(.33)], fill=(10,10,12,255))
            draw.polygon([(px(.62),py(.22)),(px(.96),py(.06)),(px(.98),py(.44)),(px(.72),py(.38))], fill=(28,108,38,255))

    # ─────────────────────────────────────────────────────────────
    # PLATFORM TILE per genre
    # ─────────────────────────────────────────────────────────────
    elif asset_name == "platform_tile":
        if is_racing:
            # Asphalt road tile with fine texture and proper markings
            for y in range(height):
                t = y / height
                shade = int(38 + t * 12)
                draw.line([0, y, width, y], fill=(shade, shade+2, shade+4, 255))
            # Random pebble/texture dots
            random.seed(seed+1)
            for _ in range(60):
                tx = random.randint(0, width-4)
                ty = random.randint(0, height-4)
                shade = random.randint(28, 52)
                draw.rectangle([tx,ty,tx+2,ty+2], fill=(shade,shade,shade+2,255))
            # Center yellow line
            draw.rectangle([0,py(.45),width,py(.58)], fill=(240,200,0,255))
            draw.rectangle([0,py(.48),width,py(.55)], fill=(255,220,20,255))
            # White edge lines
            draw.rectangle([0,py(.06),width,py(.10)], fill=(220,220,215,255))
            draw.rectangle([0,py(.90),width,py(.94)], fill=(220,220,215,255))
        elif is_dungeon:
            # Stone dungeon floor tiles with worn edges
            draw.rectangle([0,0,width,height], fill=(52,48,44,255))
            b = width // 3
            for r in range(3):
                for c in range(3):
                    bx1,by1 = c*b + (2 if r%2 else 0), r*b
                    bx2,by2 = bx1+b-2, by1+b-2
                    fill_c = (62+(c*8)%18, 58+(r*6)%14, 54+(c*4)%12, 255)
                    draw.rectangle([bx1,by1,bx2,by2], fill=fill_c)
                    draw.line([bx1,by1,bx2,by1], fill=(78,74,70,255), width=2)
                    draw.line([bx1,by1,bx1,by2], fill=(78,74,70,255), width=2)
                    draw.line([bx1,by2,bx2,by2], fill=(32,28,26,255), width=3)
                    draw.line([bx2,by1,bx2,by2], fill=(32,28,26,255), width=3)
            # Cracks and grime
            random.seed(seed+5)
            for _ in range(6):
                cx = random.randint(10,width-10)
                cy = random.randint(10,height-10)
                draw.line([cx,cy,cx+random.randint(-20,20),cy+random.randint(-20,20)], fill=(28,24,22,200), width=1)
        elif is_strategy or is_td:
            # Grass terrain tile
            for y in range(height):
                t = y / height
                r = int(38+t*12)
                g = int(88+t*22)
                b_val = int(28+t*10)
                draw.line([0,y,width,y], fill=(r,g,b_val,255))
            # Grass blades
            random.seed(seed+3)
            for _ in range(50):
                gx = random.randint(0,width-4)
                gy_base = random.randint(height//2,height-4)
                gh = random.randint(6,18)
                draw.line([gx,gy_base,gx+random.randint(-3,3),gy_base-gh], fill=(48+random.randint(-10,20),118+random.randint(-15,20),38+random.randint(-8,12),255), width=2)
        elif is_running:
            # Lane tile with dashes
            for y in range(height):
                t = y/height
                shade = int(48+t*8)
                draw.line([0,y,width,y], fill=(shade,shade+2,shade+4,255))
            draw.rectangle([0,py(.04),width,py(.10)], fill=(240,240,235,255))
            draw.rectangle([0,py(.90),width,py(.96)], fill=(240,240,235,255))
            # Dash
            if seed % 2 == 0:
                draw.rectangle([0,py(.45),width,py(.55)], fill=(240,200,0,255))
        elif is_adventure:
            # Grass/dirt ground
            for y in range(height):
                t = y/height
                if t < 0.3:
                    draw.line([0,y,width,y], fill=(int(52+t*20),int(130+t*30),int(38+t*12),255))
                else:
                    tt = (t-0.3)/0.7
                    draw.line([0,y,width,y], fill=(int(100+tt*30),int(72+tt*18),int(42+tt*12),255))
            random.seed(seed+4)
            for _ in range(20):
                fx = random.randint(2,width-6)
                fy = random.randint(2,int(height*0.25))
                draw.ellipse([fx,fy,fx+4,fy+6], fill=(70,180,50,255))
        else:
            # Mossy stone brick (detailed)
            draw.rectangle([0,0,width,height], fill=(68,72,62,255))
            b_size = width // 4
            for r in range(4):
                for c in range(4):
                    offset = b_size//2 if r%2 else 0
                    bx1 = (c*b_size + offset) % width
                    by1 = r*b_size
                    bx2 = min(bx1+b_size-3, width-1)
                    by2 = by1+b_size-3
                    fill_c = (102+(c*12)%28, 108+(r*10)%22, 88+(c*8)%18, 255)
                    draw.rectangle([bx1,by1,bx2,by2], fill=fill_c)
                    draw.line([bx1,by1,bx2,by1], fill=(145,152,130,255), width=2)
                    draw.line([bx1,by1,bx1,by2], fill=(145,152,130,255), width=2)
                    draw.line([bx1,by2,bx2,by2], fill=(38,40,32,255), width=3)
                    draw.line([bx2,by1,bx2,by2], fill=(38,40,32,255), width=3)
            # Moss
            random.seed(seed+101)
            for _ in range(14):
                mx = random.randint(4,width-20)
                my = random.randint(4,height-16)
                draw.rectangle([mx,my,mx+random.randint(8,20),my+random.randint(6,14)], fill=(58,138,32,200))
                draw.rectangle([mx+2,my+2,mx+random.randint(4,12),my+5], fill=(100,185,48,200))

    # ─────────────────────────────────────────────────────────────
    # BACKGROUND - delegate to draw_parallax_sky (already rich)
    # ─────────────────────────────────────────────────────────────
    else:
        return draw_parallax_sky(width, height, game_plan)

    # Apply a subtle sharpening pass to crisp up pixel edges
    try:
        img = img.filter(ImageFilter.SHARPEN)
    except Exception:
        pass
    return img


def generate_asset(prompt, width=512, height=512, num_images=3, seed=42, genre="default"):
    import torch
    full_prompt = build_full_prompt(prompt, genre=genre)
    negative = NEGATIVE_PROMPT_BLOCK

    payload = {
        "prompt": full_prompt,
        "negative_prompt": negative,
        "num_inference_steps": STEPS,
        "guidance_scale": CFG_SCALE,
    }
    validation = validate_payload(payload)
    if not all(validation.values()):
        print(f"Payload validation warning: {validation}")

    images = []
    for i in range(num_images):
        generator = torch.Generator("cuda" if DEVICE == "cuda" else "cpu").manual_seed(seed + i)
        img = sdxl_pipe(
            prompt=full_prompt,
            negative_prompt=negative,
            width=width,
            height=height,
            num_inference_steps=STEPS,
            guidance_scale=CFG_SCALE,
            generator=generator,
        ).images[0]
        images.append(img)
    return images


def generate_all_assets(game_plan, save_dir, job_id=None):
    set_job_status(job_id, "Initialising asset generation engine...", 32, "Selecting SDXL or Procedural CPU mode")
    os.makedirs(save_dir, exist_ok=True)
    assets  = game_plan.get("assets", {})
    genre   = game_plan.get("genre", "default")
    theme   = game_plan.get("theme", "")
    title   = game_plan.get("title", "")

    genre_lower = genre.lower()
    theme_lower = theme.lower()

    # Genre flags
    is_racing   = "rac" in genre_lower or "car"      in genre_lower
    is_fighting = "fight" in genre_lower or "arena"  in genre_lower
    is_dungeon  = "dungeon" in genre_lower
    is_strategy = "strategy" in genre_lower
    is_td       = "tower" in genre_lower or "defense" in genre_lower
    is_running  = "running" in genre_lower or "runner" in genre_lower
    is_adventure= "adventure" in genre_lower

    # Quality prefix: modern AAA game art style, NOT 16-bit
    def quality_prefix(style):
        return (f"professional {style} game art, high quality, sharp clean linework, "
                f"vibrant colors, detailed shading, transparent background, "
                f"full body character sprite on white background")

    def get_dynamic_prompt(key, fallback_desc):
        # Use GPT-planned asset prompt if available
        gpt_prompt = assets.get(key, "")

        if is_racing:
            BASE = "modern racing game art style, high-fidelity 3D-rendered look"
            PROMPTS = {
                "player":        f"{quality_prefix(BASE)}, red supercar racing vehicle, side view, aerodynamic body, chrome rims, LED headlights, bold livery",
                "enemy":         f"{quality_prefix(BASE)}, blue rival sports car, side view, sleek aerodynamic body, aggressive bumper, glowing tail lights",
                "platform_tile": f"{quality_prefix('racing game')}, asphalt race track tile, yellow dashed center line, grip surface texture, top-down view",
                "background":    f"cyberpunk night city racing panorama, neon lit skyline, rain-wet road reflections, motion blur lights, ultra wide game background, 16:9",
            }
        elif is_fighting:
            BASE = "2D fighting game sprite art, Street Fighter / Guilty Gear style"
            PROMPTS = {
                "player":        f"{quality_prefix(BASE)}, muscular martial arts hero in gi uniform, dynamic fighting stance, red headband, full body",
                "enemy":         f"{quality_prefix(BASE)}, menacing cyber ninja villain, dark bodysuit, glowing red visor, full body",
                "platform_tile": f"{quality_prefix('fighting arena')}, marble fighting arena floor tile, polished stone surface, decorative border",
                "background":    f"dramatic fighting game arena background, Japanese dojo / neon city, crowd silhouettes, dramatic lighting, ultra wide 16:9",
            }
        elif is_dungeon:
            BASE = "dungeon crawler RPG game art"
            PROMPTS = {
                "player":        f"{quality_prefix(BASE)}, armored fantasy hero warrior, full plate armor, sword and shield, determined expression, full body sprite",
                "enemy":         f"{quality_prefix(BASE)}, terrifying skeleton warrior enemy, glowing eye sockets, rusted armor, full body sprite",
                "platform_tile": f"{quality_prefix('dungeon game')}, dark stone dungeon floor brick tile, torchlight shading, moss and cracks",
                "background":    f"dark fantasy dungeon corridor background, stone walls, flickering torches, treasure chests, atmospheric fog, ultra wide 16:9",
            }
        elif is_strategy:
            BASE = "real-time strategy game unit art"
            PROMPTS = {
                "player":        f"{quality_prefix(BASE)}, medieval knight commander unit, top-down isometric perspective, blue banner, full body",
                "enemy":         f"{quality_prefix(BASE)}, orc warlord enemy unit, top-down isometric, red banner, war axe, full body",
                "platform_tile": f"{quality_prefix('strategy game')}, green grass terrain tile, light shadow, top-down view, hex grid compatible",
                "background":    f"medieval strategy game map background, rolling hills, rivers, castles on horizon, painterly style, ultra wide 16:9",
            }
        elif is_td:
            BASE = "tower defense game art"
            PROMPTS = {
                "player":        f"{quality_prefix(BASE)}, stone cannon tower, detailed battlements, muzzle flash, medieval fantasy style, full asset",
                "enemy":         f"{quality_prefix(BASE)}, green slime creep enemy, big angry eyes, stubby arms, full body sprite",
                "platform_tile": f"{quality_prefix('tower defense game')}, grass path tile, worn dirt road, top-down view",
                "background":    f"tower defense game field background, winding enemy path, lush green landscape, castle gate, ultra wide 16:9",
            }
        elif is_running:
            BASE = "endless runner mobile game art"
            PROMPTS = {
                "player":        f"{quality_prefix(BASE)}, athletic runner character in bright tracksuit, dynamic running pose, side view, full body sprite",
                "enemy":         f"{quality_prefix(BASE)}, dangerous spike trap obstacle, warning stripes, metallic spikes, side view",
                "platform_tile": f"{quality_prefix('runner game')}, city street lane tile, asphalt texture, white painted lines, side view",
                "background":    f"endless runner city skyline background, colorful buildings, clouds, bright daylight, parallax layers, ultra wide 16:9",
            }
        elif is_adventure:
            BASE = "action adventure RPG game art"
            PROMPTS = {
                "player":        f"{quality_prefix(BASE)}, brave fantasy adventurer hero, green tunic, leather boots, sword at side, full body character sprite",
                "enemy":         f"{quality_prefix(BASE)}, ferocious green dragon enemy, wings spread, glowing red eyes, sharp claws, full body sprite",
                "platform_tile": f"{quality_prefix('adventure game')}, grassy ground platform tile, dirt and grass texture, side view",
                "background":    f"lush fantasy adventure world background, enchanted forest, mountains, glowing portal, waterfalls, ultra wide 16:9",
            }
        else:
            # Default platformer
            BASE = "2D platformer game art, SNES / indie style"
            PROMPTS = {
                "player":        f"{quality_prefix(BASE)}, brave knight hero character, silver armor, blue cape, broadsword, full body sprite side view",
                "enemy":         f"{quality_prefix(BASE)}, menacing green dragon enemy, fire breath, full body sprite side view",
                "platform_tile": f"{quality_prefix('platformer game')}, mossy stone brick platform tile, highlighted top edge, side view",
                "background":    f"enchanted twilight forest background, mountains silhouette, aurora borealis, glowing mushrooms, ultra wide panorama 16:9",
            }

        # If GPT provided a specific asset prompt, enrich it rather than replace
        base_p = PROMPTS.get(key, f"{quality_prefix(BASE)}, {fallback_desc}")
        if gpt_prompt:
            return f"{base_p}, {gpt_prompt}"
        return base_p

    configs = {
        "player":        {"prompt": get_dynamic_prompt("player",        "hero character sprite"), "width": 512, "height": 512, "num": 2, "progress": 40},
        "enemy":         {"prompt": get_dynamic_prompt("enemy",         "enemy character sprite"), "width": 512, "height": 512, "num": 2, "progress": 55},
        "platform_tile": {"prompt": get_dynamic_prompt("platform_tile", "terrain tile"),           "width": 512, "height": 512, "num": 2, "progress": 70},
        "background":    {"prompt": get_dynamic_prompt("background",    "game environment scene"), "width": 1024,"height": 512, "num": 2, "progress": 82},
    }

    use_fast_mode = (DEVICE == "cpu")

    if not use_fast_mode:
        try:
            ensure_sdxl_loaded()
        except Exception as e:
            print(f"SDXL load note ({e}), switching to Procedural CPU mode...")
            use_fast_mode = True

    all_candidates = {}
    for asset_name, cfg in configs.items():
        set_job_status(
            job_id,
            f"Generating {genre} {asset_name.replace('_', ' ')} asset...",
            cfg["progress"],
            f"Rendering '{cfg['prompt'][:55]}...'"
        )
        if use_fast_mode:
            candidates = [
                generate_procedural_sprite(asset_name, cfg["prompt"], cfg["width"], cfg["height"], seed=42 + i, game_plan=game_plan)
                for i in range(cfg["num"])
            ]
        else:
            candidates = generate_asset(cfg["prompt"], cfg["width"], cfg["height"], cfg["num"], genre=genre)

        all_candidates[asset_name] = candidates
        for i, img in enumerate(candidates):
            img.save(os.path.join(save_dir, f"{asset_name}_v{i}.png"))

    gc.collect()
    return all_candidates

# --------------------------------------------------------------------------
# Phase 5 - CLIP scoring + selection
# --------------------------------------------------------------------------

def score_candidates(candidates, text_prompt):
    import torch
    text_tokens = clip_tokenizer([text_prompt])
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    scores = []
    for img in candidates:
        img_tensor = clip_preprocess(img).unsqueeze(0)
        with torch.no_grad():
            img_features = clip_model.encode_image(img_tensor)
            img_features /= img_features.norm(dim=-1, keepdim=True)
        scores.append((text_features @ img_features.T).item())

    best_idx = int(np.argmax(scores))
    return candidates[best_idx], best_idx, scores


def select_best_assets(all_candidates, game_plan):
    if DEVICE == "cuda":
        try:
            ensure_clip_loaded()
        except Exception as e:
            print(f"CLIP load note: {e}")

    assets_prompts = game_plan.get("assets", {})
    best_assets = {}

    for asset_name, candidates in all_candidates.items():
        if DEVICE == "cuda" and clip_model is not None:
            try:
                prompt = assets_prompts.get(asset_name, f"16-bit pixel art {asset_name}")
                best_img, _, _ = score_candidates(candidates, prompt)
                best_assets[asset_name] = best_img.copy()
            except Exception as e:
                print(f"CLIP scoring note ({e}), selecting candidate 0...")
                best_assets[asset_name] = candidates[0].copy()
        else:
            best_assets[asset_name] = candidates[0].copy()

    all_candidates.clear()
    gc.collect()
    return best_assets


def remove_flat_background(img, tolerance=28):
    """
    Removes solid/white background cleanly without eroding dark pixel outlines or creating white halos.
    """
    img = img.convert("RGBA")
    data = np.array(img, dtype=np.int32)
    h, w = data.shape[:2]

    # Sample four corner pixels to detect solid background color
    corners = [data[0, 0, :3], data[0, w - 1, :3], data[h - 1, 0, :3], data[h - 1, w - 1, :3]]

    mask = np.zeros((h, w), dtype=bool)
    for corner in corners:
        diff = np.abs(data[:, :, :3] - corner).sum(axis=-1)
        mask |= (diff < tolerance)

    data_out = np.array(img, dtype=np.uint8)
    data_out[mask, 3] = 0
    return Image.fromarray(data_out, mode="RGBA")

# --------------------------------------------------------------------------
# Phase 6 - Scene composition & HUD Drawing Helpers
# --------------------------------------------------------------------------

def _draw_fighting_hud(draw, canvas_width, canvas_height, game_plan, p1_hp=100, p2_hp=100):
    draw.rectangle([0, 0, canvas_width, 64], fill=(15, 15, 25, 230))
    p1_name, p2_name = "HERO", "RIVAL"
    if game_plan and "assets" in game_plan:
        if "player" in game_plan["assets"]:
            p1_name = str(game_plan["assets"]["player"]).split(",")[0].replace("pxlrpg_style", "").replace("pixel_art", "").strip() or "HERO"
        if "enemy" in game_plan["assets"]:
            p2_name = str(game_plan["assets"]["enemy"]).split(",")[0].replace("pxlrpg_style", "").replace("pixel_art", "").strip() or "RIVAL"

    p1_name = p1_name[:12].upper()
    p2_name = p2_name[:12].upper()

    draw.rectangle([8, 6, 60, 58], fill=(45, 25, 65, 255))
    draw.rectangle([8, 6, 60, 58], outline=(180, 100, 240, 255), width=2)
    draw.text((16, 22), "P1", fill=(255, 255, 255, 255))

    hp_left, hp_right = 68, canvas_width // 2 - 44
    p1_bar_w = int((hp_right - hp_left) * max(0, min(1.0, p1_hp / 100.0)))
    draw.rectangle([hp_left, 12, hp_right, 36], fill=(180, 30, 30, 255))
    if p1_bar_w > 0:
        draw.rectangle([hp_left, 12, hp_left + p1_bar_w, 36], fill=(40, 220, 80, 255))
    draw.rectangle([hp_left, 12, hp_right, 36], outline=(255, 255, 255, 255), width=2)
    draw.text((hp_left, 40), p1_name, fill=(255, 255, 255, 255))

    timer_cx = canvas_width // 2
    draw.rectangle([timer_cx - 32, 6, timer_cx + 32, 48], fill=(20, 20, 30, 255))
    draw.rectangle([timer_cx - 32, 6, timer_cx + 32, 48], outline=(255, 215, 0, 255), width=2)
    draw.text((timer_cx - 10, 14), "64", fill=(255, 215, 0, 255))
    draw.ellipse([timer_cx - 14, 52, timer_cx - 6, 60], fill=(255, 215, 0, 255))
    draw.ellipse([timer_cx + 6, 52, timer_cx + 14, 60], fill=(255, 215, 0, 255))

    p2_hp_left, p2_hp_right = canvas_width // 2 + 44, canvas_width - 68
    p2_bar_w = int((p2_hp_right - p2_hp_left) * max(0, min(1.0, p2_hp / 100.0)))
    draw.rectangle([p2_hp_left, 12, p2_hp_right, 36], fill=(180, 30, 30, 255))
    if p2_bar_w > 0:
        draw.rectangle([p2_hp_right - p2_bar_w, 12, p2_hp_right, 36], fill=(40, 220, 80, 255))
    draw.rectangle([p2_hp_left, 12, p2_hp_right, 36], outline=(255, 255, 255, 255), width=2)
    draw.text((p2_hp_left, 40), p2_name, fill=(255, 255, 255, 255))

    draw.rectangle([canvas_width - 60, 6, canvas_width - 8, 58], fill=(65, 25, 45, 255))
    draw.rectangle([canvas_width - 60, 6, canvas_width - 8, 58], outline=(240, 100, 140, 255), width=2)
    draw.text((canvas_width - 50, 22), "P2", fill=(255, 255, 255, 255))


def _draw_racing_hud(draw, canvas_width, canvas_height):
    px1, py1 = canvas_width - 200, canvas_height - 80
    px2, py2 = canvas_width - 10, canvas_height - 10
    draw.rectangle([px1, py1, px2, py2], fill=(20, 25, 35, 230))
    draw.rectangle([px1, py1, px2, py2], outline=(0, 230, 255, 255), width=2)

    draw.text((px1 + 12, py1 + 8),  "SPEED:", fill=(160, 170, 185, 255))
    draw.text((px1 + 75, py1 + 8),  "145 km/h", fill=(0, 255, 240, 255))
    draw.text((px1 + 12, py1 + 28), "LAP 2/3", fill=(255, 215, 0, 255))
    draw.text((px1 + 12, py1 + 48), "POS 1ST", fill=(255, 255, 255, 255))


def _draw_shooter_hud(draw, canvas_width, canvas_height):
    px1, py1 = 10, canvas_height - 75
    px2, py2 = 180, canvas_height - 10
    draw.rectangle([px1, py1, px2, py2], fill=(15, 20, 30, 230))
    draw.rectangle([px1, py1, px2, py2], outline=(0, 255, 200, 255), width=2)

    draw.text((px1 + 10, py1 + 8), "HP", fill=(255, 255, 255, 255))
    draw.rectangle([px1 + 35, py1 + 10, px1 + 160, py1 + 22], fill=(160, 30, 30, 255))
    draw.rectangle([px1 + 35, py1 + 10, px1 + 130, py1 + 22], fill=(30, 220, 90, 255))
    draw.text((px1 + 10, py1 + 35), "AMMO 24/120", fill=(255, 215, 0, 255))


def render_racing(best_assets, layout_json, game_plan, tile_size, canvas_width, canvas_height, include_sprites):
    bg = best_assets.get("background")
    if bg:
        canvas = bg.resize((canvas_width, canvas_height), Image.LANCZOS).convert("RGBA")
    else:
        canvas = draw_parallax_sky(canvas_width, canvas_height, game_plan)
    draw = ImageDraw.Draw(canvas)

    # Draw race track lines perspective
    draw.polygon([(canvas_width // 2 - 40, canvas_height // 2), (canvas_width // 2 + 40, canvas_height // 2), (canvas_width, canvas_height), (0, canvas_height)], fill=(40, 40, 45, 255))
    draw.line([canvas_width // 2, canvas_height // 2, canvas_width // 2, canvas_height], fill=(255, 215, 0, 255), width=4)

    if include_sprites:
        player_sprite = best_assets.get("player")
        if player_sprite:
            car = remove_flat_background(player_sprite).resize((180, 90), Image.NEAREST).convert("RGBA")
            canvas.paste(car, (int(canvas_width * 0.15), int(canvas_height * 0.72)), car)
        
        enemy_sprite = best_assets.get("enemy")
        if enemy_sprite:
            enemy_car = remove_flat_background(enemy_sprite).resize((120, 60), Image.NEAREST).convert("RGBA")
            canvas.paste(enemy_car, (int(canvas_width * 0.55), int(canvas_height * 0.58)), enemy_car)

    _draw_racing_hud(draw, canvas_width, canvas_height)
    return canvas.convert("RGB")

def render_fighting(best_assets, layout_json, game_plan, tile_size, canvas_width, canvas_height, include_sprites):
    bg = best_assets.get("background")
    if bg:
        canvas = bg.resize((canvas_width, canvas_height), Image.LANCZOS).convert("RGBA")
    else:
        canvas = draw_parallax_sky(canvas_width, canvas_height, game_plan)
    draw = ImageDraw.Draw(canvas)

    if include_sprites:
        player_sprite = best_assets.get("player")
        if player_sprite:
            p1 = remove_flat_background(player_sprite).resize((150, 150), Image.NEAREST).convert("RGBA")
            canvas.paste(p1, (int(canvas_width * 0.18), int(canvas_height * 0.44)), p1)

        enemy_sprite = best_assets.get("enemy")
        if enemy_sprite:
            p2 = remove_flat_background(enemy_sprite).resize((150, 150), Image.NEAREST).convert("RGBA")
            p2 = p2.transpose(Image.FLIP_LEFT_RIGHT)
            canvas.paste(p2, (int(canvas_width * 0.66), int(canvas_height * 0.44)), p2)

    _draw_fighting_hud(draw, canvas_width, canvas_height, game_plan, p1_hp=100, p2_hp=100)
    return canvas.convert("RGB")

def render_adventure(best_assets, layout_json, game_plan, tile_size, canvas_width, canvas_height, include_sprites):
    bg = best_assets.get("background")
    if bg:
        canvas = bg.resize((canvas_width, canvas_height), Image.LANCZOS).convert("RGBA")
    else:
        canvas = draw_parallax_sky(canvas_width, canvas_height, game_plan)
    draw = ImageDraw.Draw(canvas)

    # Draw walking path
    draw.rectangle([0, int(canvas_height * 0.72), canvas_width, canvas_height], fill=(120, 85, 45, 255))
    draw.rectangle([0, int(canvas_height * 0.72), canvas_width, int(canvas_height * 0.74)], fill=(160, 120, 70, 255))

    if include_sprites:
        player_sprite = best_assets.get("player")
        if player_sprite:
            p1 = remove_flat_background(player_sprite).resize((120, 120), Image.NEAREST).convert("RGBA")
            canvas.paste(p1, (int(canvas_width * 0.15), int(canvas_height * 0.50)), p1)

        enemy_sprite = best_assets.get("enemy")
        if enemy_sprite:
            chest = remove_flat_background(enemy_sprite).resize((90, 90), Image.NEAREST).convert("RGBA")
            canvas.paste(chest, (int(canvas_width * 0.70), int(canvas_height * 0.58)), chest)

    # Draw adventure HUD
    draw.rectangle([10, 10, 260, 55], fill=(30, 30, 40, 220), outline=(218, 165, 32, 255), width=2)
    draw.text((20, 16), "QUEST: FIND THE MYSTIC KEY", fill=(255, 215, 0, 255))
    draw.text((20, 32), "INVENTORY: [ ] KEY  [ ] MAP", fill=(200, 200, 200, 255))
    return canvas.convert("RGB")

def render_dungeon(best_assets, layout_json, game_plan, tile_size, canvas_width, canvas_height, include_sprites):
    canvas = Image.new("RGBA", (canvas_width, canvas_height), (20, 15, 15, 255))
    draw = ImageDraw.Draw(canvas)

    # Draw grid floor tiles
    for x in range(0, canvas_width, 48):
        for y in range(0, canvas_height, 48):
            draw.rectangle([x, y, x + 46, y + 46], fill=(40, 35, 35, 255), outline=(50, 45, 45, 255))

    # Outer wall border
    draw.rectangle([0, 0, canvas_width, 24], fill=(20, 20, 20, 255))
    draw.rectangle([0, canvas_height-24, canvas_width, canvas_height], fill=(20, 20, 20, 255))

    if include_sprites:
        player_sprite = best_assets.get("player")
        if player_sprite:
            p1 = remove_flat_background(player_sprite).resize((90, 90), Image.NEAREST).convert("RGBA")
            canvas.paste(p1, (100, canvas_height // 2 - 45), p1)

        enemy_sprite = best_assets.get("enemy")
        if enemy_sprite:
            es = remove_flat_background(enemy_sprite).resize((90, 90), Image.NEAREST).convert("RGBA")
            canvas.paste(es, (canvas_width - 200, canvas_height // 2 - 45), es)

    # Dungeon HUD
    draw.rectangle([10, 10, 220, 50], fill=(15, 10, 10, 230), outline=(255, 50, 50, 255), width=2)
    draw.text((20, 16), "DUNGEON LEVEL 1", fill=(255, 50, 50, 255))
    draw.text((20, 30), "HP: 100/100  KEYS: 0", fill=(255, 255, 255, 255))
    return canvas.convert("RGB")

def render_strategy(best_assets, layout_json, game_plan, tile_size, canvas_width, canvas_height, include_sprites):
    canvas = Image.new("RGBA", (canvas_width, canvas_height), (35, 55, 35, 255))
    draw = ImageDraw.Draw(canvas)

    # Draw grid terrain
    for x in range(0, canvas_width, 64):
        for y in range(0, canvas_height, 64):
            draw.rectangle([x, y, x + 62, y + 62], fill=(40, 65, 40, 255))

    if include_sprites:
        player_sprite = best_assets.get("player")
        if player_sprite:
            base1 = remove_flat_background(player_sprite).resize((110, 110), Image.NEAREST).convert("RGBA")
            canvas.paste(base1, (80, canvas_height // 2 - 55), base1)

        enemy_sprite = best_assets.get("enemy")
        if enemy_sprite:
            base2 = remove_flat_background(enemy_sprite).resize((110, 110), Image.NEAREST).convert("RGBA")
            canvas.paste(base2, (canvas_width - 200, canvas_height // 2 - 55), base2)

    # Strategy HUD
    draw.rectangle([0, canvas_height - 40, canvas_width, canvas_height], fill=(20, 20, 30, 255))
    draw.text((20, canvas_height - 30), "GOLD: 750   WOOD: 400   UNITS: 15/50", fill=(255, 215, 0, 255))
    return canvas.convert("RGB")

def render_platformer(best_assets, layout_json, game_plan, tile_size, canvas_width, canvas_height, include_sprites):
    bg = best_assets.get("background")
    if bg:
        canvas = bg.resize((canvas_width, canvas_height), Image.LANCZOS).convert("RGBA")
    else:
        canvas = draw_parallax_sky(canvas_width, canvas_height, game_plan)
    draw = ImageDraw.Draw(canvas)

    platforms = layout_json.get("platforms", [])
    if not platforms:
        return canvas.convert("RGB")

    all_x = [p[0] for p in platforms]
    all_y = [p[1] for p in platforms]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    grid_w, grid_h = max_x - min_x + 1, max_y - min_y + 1

    scale = min(canvas_width / (grid_w * tile_size), canvas_height / (grid_h * tile_size), 1.0)
    ts = max(int(tile_size * scale), 6)

    platform_tile = best_assets.get("platform_tile")
    if platform_tile:
        platform_tile = platform_tile.resize((ts, ts), Image.NEAREST).convert("RGBA")

    for px_grid, py_grid in platforms:
        screen_x = int((px_grid - min_x) * ts)
        screen_y = int((py_grid - min_y) * ts)
        if platform_tile and 0 <= screen_x < canvas_width and 0 <= screen_y < canvas_height:
            canvas.paste(platform_tile, (screen_x, screen_y), platform_tile)

    if include_sprites:
        player_pos = layout_json.get("player", [0, 0])
        player_sprite = best_assets.get("player")
        if player_sprite:
            ps = remove_flat_background(player_sprite).resize((ts * 2, ts * 2), Image.NEAREST).convert("RGBA")
            px_screen = int((player_pos[0] - min_x) * ts)
            py_screen = int((player_pos[1] - min_y) * ts) - ts
            canvas.paste(ps, (px_screen, py_screen), ps)

        enemies = layout_json.get("enemies", [])
        enemy_sprite = best_assets.get("enemy")
        if enemy_sprite and enemies:
            es = remove_flat_background(enemy_sprite).resize((ts * 2, ts * 2), Image.NEAREST).convert("RGBA")
            for ex, ey in enemies[:10]:
                ex_screen = int((ex - min_x) * ts)
                ey_screen = int((ey - min_y) * ts) - ts
                canvas.paste(es, (ex_screen, ey_screen), es)

    goal_pos = layout_json.get("goal", [0, 0])
    gx = int((goal_pos[0] - min_x) * ts)
    gy = int((goal_pos[1] - min_y) * ts) - ts
    draw.rectangle([gx, gy, gx + 6, gy + ts * 2], fill=(255, 215, 0, 255))
    draw.polygon([(gx + 6, gy), (gx + ts * 2, gy + int(ts * 0.6)), (gx + 6, gy + int(ts * 1.2))], fill=(255, 40, 40, 255))

    # Platformer HUD
    draw.rectangle([10, 10, 180, 45], fill=(0, 0, 0, 180))
    draw.text((20, 15), "SCORE: 00450   LIVES: 3", fill=(255, 255, 255, 255))
    return canvas.convert("RGB")

def render_tower_defense(best_assets, layout_json, game_plan, tile_size, canvas_width, canvas_height, include_sprites):
    canvas = Image.new("RGBA", (canvas_width, canvas_height), (35, 30, 25, 255))
    draw = ImageDraw.Draw(canvas)

    # Draw path
    path_pts = [(0, 240), (200, 240), (200, 100), (450, 100), (450, 360), (700, 360), (700, 240), (canvas_width, 240)]
    for i in range(len(path_pts)-1):
        draw.line([path_pts[i], path_pts[i+1]], fill=(120, 110, 100, 255), width=48)
        draw.line([path_pts[i], path_pts[i+1]], fill=(160, 150, 140, 255), width=40)

    if include_sprites:
        player_sprite = best_assets.get("player")
        if player_sprite:
            tower = remove_flat_background(player_sprite).resize((70, 70), Image.NEAREST).convert("RGBA")
            canvas.paste(tower, (100, 110), tower)
            canvas.paste(tower, (320, 170), tower)
            canvas.paste(tower, (600, 280), tower)

        enemy_sprite = best_assets.get("enemy")
        if enemy_sprite:
            minion = remove_flat_background(enemy_sprite).resize((50, 50), Image.NEAREST).convert("RGBA")
            canvas.paste(minion, (200, 160), minion)
            canvas.paste(minion, (450, 240), minion)

    # TD HUD
    draw.rectangle([10, 10, 200, 50], fill=(20, 20, 20, 220), outline=(0, 200, 255, 255), width=2)
    draw.text((20, 16), "WAVE: 3/10", fill=(0, 200, 255, 255))
    draw.text((20, 30), "HEALTH: 20  GOLD: 250", fill=(255, 255, 255, 255))
    return canvas.convert("RGB")

def render_running(best_assets, layout_json, game_plan, tile_size, canvas_width, canvas_height, include_sprites):
    bg = best_assets.get("background")
    if bg:
        canvas = bg.resize((canvas_width, canvas_height), Image.LANCZOS).convert("RGBA")
    else:
        canvas = draw_parallax_sky(canvas_width, canvas_height, game_plan)
    draw = ImageDraw.Draw(canvas)

    # Draw lanes
    draw.line([0, 160, canvas_width, 160], fill=(255, 255, 255, 100), width=3)
    draw.line([0, 280, canvas_width, 280], fill=(255, 255, 255, 100), width=3)

    if include_sprites:
        player_sprite = best_assets.get("player")
        if player_sprite:
            runner = remove_flat_background(player_sprite).resize((110, 110), Image.NEAREST).convert("RGBA")
            canvas.paste(runner, (80, 180), runner)

        enemy_sprite = best_assets.get("enemy")
        if enemy_sprite:
            obstacle = remove_flat_background(enemy_sprite).resize((70, 70), Image.NEAREST).convert("RGBA")
            canvas.paste(obstacle, (400, 200), obstacle)
            canvas.paste(obstacle, (700, 80), obstacle)

    # Runner HUD
    draw.rectangle([10, 10, 220, 45], fill=(10, 10, 10, 200))
    draw.text((20, 16), "DISTANCE: 0340m  x1.2", fill=(0, 255, 100, 255))
    return canvas.convert("RGB")

def compose_scene(best_assets, layout_json, game_plan, tile_size=32, canvas_width=1024, canvas_height=480, include_sprites=True):
    genre_lower = (game_plan.get("genre", "") if game_plan else "").lower()
    
    if "rac" in genre_lower:
        return render_racing(best_assets, layout_json, game_plan, tile_size, canvas_width, canvas_height, include_sprites)
    elif "fight" in genre_lower:
        return render_fighting(best_assets, layout_json, game_plan, tile_size, canvas_width, canvas_height, include_sprites)
    elif "adventure" in genre_lower:
        return render_adventure(best_assets, layout_json, game_plan, tile_size, canvas_width, canvas_height, include_sprites)
    elif "dungeon" in genre_lower:
        return render_dungeon(best_assets, layout_json, game_plan, tile_size, canvas_width, canvas_height, include_sprites)
    elif "strategy" in genre_lower:
        return render_strategy(best_assets, layout_json, game_plan, tile_size, canvas_width, canvas_height, include_sprites)
    elif "tower" in genre_lower or "td" in genre_lower:
        return render_tower_defense(best_assets, layout_json, game_plan, tile_size, canvas_width, canvas_height, include_sprites)
    elif "run" in genre_lower:
        return render_running(best_assets, layout_json, game_plan, tile_size, canvas_width, canvas_height, include_sprites)
    else:
        return render_platformer(best_assets, layout_json, game_plan, tile_size, canvas_width, canvas_height, include_sprites)

def validate_playability(layout_json, max_jump_height=4, max_jump_width=5):
    platforms = set(tuple(p) for p in layout_json.get("platforms", []))
    player = tuple(layout_json.get("player", [0, 0]))
    goal = tuple(layout_json.get("goal", [0, 0]))

    if not platforms:
        return False, [], {"error": "No platforms defined"}

    def get_standing_positions():
        standing = set()
        for px_coord, py_coord in platforms:
            standing.add((px_coord, py_coord - 1))
        return standing

    walkable = get_standing_positions()
    walkable.update(platforms)

    def nearest_walkable(pos):
        if pos in walkable:
            return pos
        best, best_dist = None, float("inf")
        for w in walkable:
            d = abs(w[0] - pos[0]) + abs(w[1] - pos[1])
            if d < best_dist:
                best_dist, best = d, w
        return best

    start = nearest_walkable(player)
    end = nearest_walkable(goal)
    if start is None or end is None:
        return False, [], {"error": "Start or goal not near any platform"}

    visited = {start}
    queue = deque([(start, [start])])

    while queue:
        (cx, cy), path = queue.popleft()
        if abs(cx - end[0]) <= 2 and abs(cy - end[1]) <= 2:
            return True, path, {"path_length": len(path), "start": start, "goal": end, "status": "PLAYABLE"}

        moves = [(cx - 1, cy), (cx + 1, cy)]
        for dx in range(-max_jump_width, max_jump_width + 1):
            for dy in range(-max_jump_height, 1):
                if dx == 0 and dy == 0:
                    continue
                moves.append((cx + dx, cy + dy))
        for dy in range(1, 20):
            moves.append((cx, cy + dy))
            moves.append((cx - 1, cy + dy))
            moves.append((cx + 1, cy + dy))

        for nx, ny in moves:
            if (nx, ny) not in visited and (nx, ny) in walkable:
                visited.add((nx, ny))
                queue.append(((nx, ny), path + [(nx, ny)]))

    suggestions = suggest_fixes(start, end, walkable, max_jump_height, max_jump_width)
    return False, [], {"status": "NOT PLAYABLE", "start": start, "goal": end,
                        "tiles_explored": len(visited), "suggestions": suggestions}


def suggest_fixes(start, end, walkable, jump_h, jump_w):
    suggestions = []
    visited = {start}
    queue = deque([start])
    rightmost = start

    while queue:
        cx, cy = queue.popleft()
        if cx > rightmost[0]:
            rightmost = (cx, cy)
        for dx in range(-jump_w, jump_w + 1):
            for dy in range(-jump_h, 3):
                nx, ny = cx + dx, cy + dy
                if (nx, ny) not in visited and (nx, ny) in walkable:
                    visited.add((nx, ny))
                    queue.append((nx, ny))

    gap_x = rightmost[0] + 1
    suggestions.append({"type": "add_platform", "position": [gap_x + 2, rightmost[1]],
                        "reason": f"Bridge gap after rightmost reachable tile at {rightmost}"})
    mid_x = (rightmost[0] + end[0]) // 2
    mid_y = (rightmost[1] + end[1]) // 2
    suggestions.append({"type": "add_platform", "position": [mid_x, mid_y],
                        "reason": "Stepping stone between reachable area and goal"})
    return suggestions


def auto_fix_level(layout_json, max_attempts=5):
    layout = json.loads(json.dumps(layout_json))
    for _ in range(max_attempts):
        playable, path, details = validate_playability(layout)
        if playable:
            return layout, path, details
        for s in details.get("suggestions", []):
            pos = s["position"]
            for dx in range(-1, 2):
                new_tile = [pos[0] + dx, pos[1]]
                if new_tile not in layout["platforms"]:
                    layout["platforms"].append(new_tile)
    return layout, [], {"status": "COULD NOT FIX", "attempts": max_attempts}


# --------------------------------------------------------------------------
# Phase 8 - Rich Action Video Preview Generator (Fixed)
# --------------------------------------------------------------------------

def _try_init_video_writer(output_path, fps=20):
    """Try ffmpeg MP4 writer; return (writer, mode) where mode='mp4' or 'gif' or None."""
    import imageio
    # Try mp4 first
    try:
        w = imageio.get_writer(
            output_path, fps=fps, codec="libx264", quality=7,
            macro_block_size=None,
            ffmpeg_params=["-preset", "ultrafast", "-pix_fmt", "yuv420p"]
        )
        return w, "mp4"
    except Exception as e:
        print(f"[Video] libx264 unavailable ({e}), trying gif fallback...")

    # GIF fallback
    gif_path = output_path.replace(".mp4", ".gif")
    try:
        w = imageio.get_writer(gif_path, mode="I", fps=min(fps, 10), loop=0)
        return w, "gif"
    except Exception as e2:
        print(f"[Video] GIF writer also failed ({e2}) - video skipped.")
        return None, None


def generate_preview_video(bg_img, layout, path, assets, output_path, game_plan=None):
    """
    Renders 8-10s gameplay preview video (150 frames @ 15fps) based on resolved genre 
    with advanced mechanics, UI overlays, and physics.
    Strictly manages memory by streaming to imageio writer.
    """
    import imageio
    import gc
    import math
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    W, H = 640, 360
    base_bg = bg_img.resize((W, H), Image.LANCZOS)
    genre = game_plan.get("genre", "platformer").lower() if game_plan else "platformer"
    
    player_img = assets.get("player")
    if player_img: player_img = player_img.resize((int(W*0.08), int(H*0.12)), Image.LANCZOS)
    enemy_img = assets.get("enemy")
    if enemy_img: enemy_img = enemy_img.resize((int(W*0.08), int(H*0.12)), Image.LANCZOS)
    item_img = assets.get("item")
    if item_img: item_img = item_img.resize((int(W*0.05), int(H*0.08)), Image.LANCZOS)

    gif_path = output_path.replace(".mp4", ".gif")
    
    import imageio
    try:
        # Stream directly to MP4 using imageio-ffmpeg (H.264) to bypass browser incompatibility and RAM OOM
        writer = imageio.get_writer(output_path, fps=15, macro_block_size=None)
        
        TOTAL_FRAMES = 150  # 10 seconds at 15 fps
        
        for i in range(TOTAL_FRAMES):
            frame = base_bg.copy()
            draw = ImageDraw.Draw(frame)
            
            # --- 1. RACING ---
            if genre == "racing":
                # Mode-7 Style 3D Road
                horizon = int(H * 0.5)
                # Draw ground
                draw.rectangle([0, horizon, W, H], fill=(40, 40, 45, 255))
                # Road perspective lines
                road_w_bottom = W * 0.8
                road_w_top = W * 0.2
                road_pts = [(W/2 - road_w_top/2, horizon), (W/2 + road_w_top/2, horizon), 
                            (W/2 + road_w_bottom/2, H), (W/2 - road_w_bottom/2, H)]
                draw.polygon(road_pts, fill=(60, 60, 65, 255))
                
                # Moving dashed lines
                offset = (i * 0.5) % 1.0
                for strip in range(5):
                    z1 = (strip + offset) / 5.0
                    z2 = (strip + offset + 0.5) / 5.0
                    if z1 > 1.0: continue
                    z2 = min(z2, 1.0)
                    
                    # Perspective projection
                    y1 = horizon + (H - horizon) * (z1**2)
                    y2 = horizon + (H - horizon) * (z2**2)
                    w1 = road_w_top + (road_w_bottom - road_w_top) * (z1**2)
                    w2 = road_w_top + (road_w_bottom - road_w_top) * (z2**2)
                    draw.polygon([(W/2-2, y1), (W/2+2, y1), (W/2+4, y2), (W/2-4, y2)], fill=(255,200,0,255))
                
                # Cars
                # Player car steering
                steer = math.sin(i * 0.05)
                car_x = int(W*0.5 + steer * W*0.2)
                car_y = int(H*0.8)
                
                # Rival car overtaking and falling behind
                rival_z = (i * 0.02) % 2.0
                if rival_z > 1.0: rival_z = 2.0 - rival_z # yo-yo effect
                rival_y = int(horizon + (H - horizon) * (rival_z**2))
                rival_scale = 0.3 + 0.7 * (rival_z**2)
                
                if enemy_img:
                    scaled_e = enemy_img.resize((int(enemy_img.width * rival_scale), int(enemy_img.height * rival_scale)), Image.LANCZOS)
                    rival_x = int(W*0.5 - W*0.15 * rival_scale)
                    frame.paste(scaled_e, (rival_x - scaled_e.width//2, rival_y - scaled_e.height), scaled_e if scaled_e.mode == 'RGBA' else None)
                    
                if player_img:
                    frame.paste(player_img, (car_x - player_img.width//2, car_y - player_img.height), player_img if player_img.mode == 'RGBA' else None)
                    
                # UI HUD
                draw.rectangle([W-120, 20, W-20, 70], fill=(0,0,0,150), outline=(255,255,255,255))
                draw.text((W-110, 25), "LAP: 2/3", fill=(255,255,255,255))
                draw.text((W-110, 45), f"SPEED: {int(120 + abs(steer)*20)} km/h", fill=(0,255,0,255))
                
            # --- 2. FIGHTING / ADVENTURE FIGHTING ---
            elif genre == "fighting":
                p_x = int(W * 0.2)
                e_x = int(W * 0.8)
                y = int(H * 0.7)
                
                # AI Logic Phase
                phase = (i // 30) % 4  # 4 phases, 2 seconds each
                hit_spark = False
                
                if phase == 0: # approach
                    p_x = int(W * 0.2) + (i % 30) * 4
                    e_x = int(W * 0.8) - (i % 30) * 4
                elif phase == 1: # attack
                    p_x = int(W * 0.2) + 120
                    e_x = int(W * 0.8) - 120
                    if (i % 30) < 10: p_x += 30 # lunge
                    if (i % 30) == 5: hit_spark = True
                elif phase == 2: # knockback
                    p_x = int(W * 0.2) + 120
                    e_x = int(W * 0.8) - 120 + ((i % 30) * 5)
                elif phase == 3: # reset approach
                    p_x = int(W * 0.2) + 120 - ((i%30)*4)
                    e_x = int(W * 0.8) + 30 - ((i%30)*5)

                if player_img: frame.paste(player_img, (p_x - player_img.width//2, y - player_img.height), player_img if player_img.mode == 'RGBA' else None)
                if enemy_img: frame.paste(enemy_img, (e_x - enemy_img.width//2, y - enemy_img.height), enemy_img if enemy_img.mode == 'RGBA' else None)
                
                if hit_spark:
                    draw.ellipse([e_x-30, y-enemy_img.height//2-30, e_x+30, y-enemy_img.height//2+30], fill=(255,255,0,200))
                    draw.polygon([(e_x-40, y-enemy_img.height//2), (e_x+40, y-enemy_img.height//2-20), (e_x+20, y-enemy_img.height//2+40)], fill=(255,50,0,255))
                
                # UI HUD
                # Player HP
                draw.rectangle([20, 20, W//2 - 20, 40], fill=(50,0,0,255), outline=(255,255,255,255))
                draw.rectangle([20, 20, W//2 - 20, 40], fill=(255,200,0,255)) # full
                # Enemy HP
                enemy_hp = max(0, 1.0 - (i / TOTAL_FRAMES))
                draw.rectangle([W//2 + 20, 20, W - 20, 40], fill=(50,0,0,255), outline=(255,255,255,255))
                draw.rectangle([W - 20 - int((W//2-40)*enemy_hp), 20, W - 20, 40], fill=(255,0,0,255))
                
                # Timer
                draw.rectangle([W//2 - 20, 10, W//2 + 20, 50], fill=(0,0,0,255), outline=(255,255,255,255))
                draw.text((W//2 - 8, 25), f"{99 - (i//15)}", fill=(255,255,255,255))

            # --- 3. TOWER DEFENSE / SHOOTING ---
            elif genre == "tower_defense" or genre == "strategy":
                tower_x, tower_y = int(W * 0.2), int(H * 0.5)
                
                if player_img: # Tower/Base
                    frame.paste(player_img, (tower_x - player_img.width//2, tower_y - player_img.height//2), player_img if player_img.mode == 'RGBA' else None)
                
                # Multiple enemies along a path
                for e_idx in range(4):
                    e_offset = i - (e_idx * 30)
                    if e_offset < 0 or e_offset > 120: continue
                    
                    # Bezier curve path
                    t = e_offset / 120.0
                    creep_x = int(W*0.9 - t*W*0.7)
                    creep_y = int(H*0.8 - math.sin(t*math.pi)*H*0.4)
                    
                    if enemy_img:
                        frame.paste(enemy_img, (creep_x - enemy_img.width//2, creep_y - enemy_img.height//2), enemy_img if enemy_img.mode == 'RGBA' else None)
                        
                    # Projectile targeting this creep
                    proj_t = (i * 2) % 15 / 15.0
                    proj_x = tower_x + int((creep_x - tower_x) * proj_t)
                    proj_y = tower_y + int((creep_y - tower_y) * proj_t)
                    
                    if e_offset % 15 < 14: # Projectile in flight
                        draw.ellipse([proj_x-5, proj_y-5, proj_x+5, proj_y+5], fill=(0, 255, 255, 255))
                        draw.line([tower_x, tower_y, proj_x, proj_y], fill=(0, 255, 255, 100), width=2)
                    else: # Explosion
                        draw.ellipse([creep_x-20, creep_y-20, creep_x+20, creep_y+20], fill=(255, 100, 0, 200))
                        
            # --- 4. PLATFORMER / RUNNING / ADVENTURE / DUNGEON ---
            else:
                # Camera tracking & Parallax
                cam_x = int((i / TOTAL_FRAMES) * W * 1.5)
                
                # Parallax background
                frame.paste(base_bg, (-int(cam_x * 0.3), 0))
                frame.paste(base_bg, (W - int(cam_x * 0.3), 0))
                
                # Floor
                floor_y = int(H * 0.8)
                draw.rectangle([0, floor_y, W, H], fill=(50, 80, 50, 255))
                
                # Player Physics
                p_base_x = int(W * 0.3)
                p_world_x = p_base_x + cam_x
                
                # 3 Jumps over the 10 seconds
                jump_cycle = (i % 50) / 50.0 
                if jump_cycle < 0.6:
                    jump_height = math.sin((jump_cycle / 0.6) * math.pi) * (H * 0.3)
                else:
                    jump_height = 0
                
                p_y = floor_y - int(jump_height)
                
                # Draw Pit
                if jump_cycle < 0.6:
                    pit_x = p_base_x + int((0.3 - jump_cycle)*W)
                    draw.rectangle([pit_x, floor_y, pit_x+100, H], fill=(20, 20, 25, 255)) # Hole
                
                if player_img:
                    frame.paste(player_img, (p_base_x - player_img.width//2, p_y - player_img.height), player_img if player_img.mode == 'RGBA' else None)
                    
                # Enemy Patrol
                for e_idx in range(3):
                    e_world_x = W + (e_idx * 300) - int(cam_x * 0.8)
                    if 0 < e_world_x < W:
                        e_y = floor_y - int(math.sin(i*0.5)*10) # Bobbing
                        if enemy_img:
                            frame.paste(enemy_img, (e_world_x - enemy_img.width//2, e_y - enemy_img.height), enemy_img if enemy_img.mode == 'RGBA' else None)
                            
                # Collectible
                item_world_x = W + 150 - cam_x
                if 0 < item_world_x < W:
                    i_y = floor_y - int(H*0.4) + int(math.sin(i*0.1)*15)
                    if item_img:
                        frame.paste(item_img, (item_world_x - item_img.width//2, i_y - item_img.height), item_img if item_img.mode == 'RGBA' else None)
                        
                # UI HUD
                draw.text((20, 20), f"SCORE: {int(i*15.5)}", fill=(255,255,255,255))
                draw.text((20, 40), f"WORLD: 1-1", fill=(255,255,255,255))
            
            # Write frame directly to disk via imageio-ffmpeg, ZERO list memory overhead
            writer.append_data(np.array(frame))
            
            del frame
            del draw
            if i % 10 == 0:
                gc.collect()
                
        writer.close()
        
    except Exception as e:
        print(f"Error generating advanced video: {e}")
        if 'writer' in locals() and writer:
            writer.close()
            
    # Final cleanup
    del base_bg
    gc.collect()



def run_full_pipeline(image_path, user_description="Create a game based on the provided sketch.", job_id=None, base_url="http://localhost:7860"):
    job_dir = os.path.join(API_OUTPUT_DIR, job_id) if job_id else os.path.join(API_OUTPUT_DIR, str(uuid.uuid4())[:8])
    os.makedirs(job_dir, exist_ok=True)

    set_job_status(job_id, "Analyzing level sketch with Florence-2 & Vision...", 10, "Extracting layout grid, caption, and object detections")
    
    # Check model loading availability and load Florence-2
    vision_loaded = False
    try:
        if DEVICE != "cpu":
            ensure_florence_loaded()

        vision_loaded = (florence_model is not None)
    except Exception as e:
        print(f"Florence load note: {e}")
        
    layout, florence_caption, florence_od, vision_info = sketch_to_layout(image_path)

    # FREE FLORENCE IMMEDIATELY to prevent OOM on 512MB RAM servers
    free_model_memory('florence')


    # Dedicated Genre Resolution Stage
    set_job_status(job_id, "Resolving game genre...", 18, "Analyzing user intent and visual layout evidence")
    genre_resolution = resolve_genre(user_description, florence_caption, florence_od, layout, vision_info)

    # Save vision, genre_resolution, and layout debug JSONs
    with open(os.path.join(job_dir, "vision.json"), "w") as f:
        json.dump(vision_info, f, indent=2)
    with open(os.path.join(job_dir, "genre_resolution.json"), "w") as f:
        json.dump(genre_resolution, f, indent=2)
    with open(os.path.join(job_dir, "layout.json"), "w") as f:
        json.dump(layout, f, indent=2)

    set_job_status(job_id, "Generating AAA game plan with GPT-4o...", 25, f"Caption: '{florence_caption[:40] if florence_caption else 'None'}...'")
    game_plan = plan_game(layout, user_description, florence_caption, florence_od, genre_resolution, vision_info)
    if not game_plan:
        set_job_status(job_id, "Failed", 0, error="GPT-4o planning failed")
        return {"error": "Failed to generate game plan from sketch"}

    # Save game plan debug JSON
    with open(os.path.join(job_dir, "game_plan.json"), "w") as f:
        json.dump(game_plan, f, indent=2)

    all_candidates = generate_all_assets(game_plan, job_dir, job_id=job_id)

    set_job_status(job_id, "Scoring candidates with CLIP...", 88, "Selecting best 16-bit sprites")
    best_assets = select_best_assets(all_candidates, game_plan)

    for asset_name, img in best_assets.items():
        img.save(os.path.join(job_dir, f"best_{asset_name}.png"))

    # FREE ALL REMAINING MODELS (SDXL, CLIP) to save memory for Video rendering
    free_model_memory('all')


    set_job_status(job_id, "Validating level playability...", 92, "Running BFS pathfinding")
    playable, path, playability_info = validate_playability(layout)

    if not playable:
        set_job_status(job_id, "Auto-fixing unplayable gaps...", 94, "Adding stepping stone platforms")
        layout, path, playability_info = auto_fix_level(layout)

    set_job_status(job_id, "Composing composed level scene...", 96, "Stitching backgrounds and sprites")
    scene = compose_scene(best_assets, layout, game_plan)
    scene_path = os.path.join(job_dir, "scene.png")
    scene.save(scene_path)

    # Generate preview video (MP4 preferred, GIF fallback on servers without ffmpeg)
    preview_mp4 = os.path.join(job_dir, "preview.mp4")
    preview_gif = os.path.join(job_dir, "preview.gif")
    preview_path = preview_mp4
    set_job_status(job_id, "Rendering gameplay preview video...", 98, "Creating animated preview (MP4 or GIF)")
    try:
        generate_preview_video(scene, layout, path, best_assets, preview_mp4, game_plan=game_plan)
    except Exception as e:
        print(f"Preview video generation note: {e}")
    # Determine which file was actually created
    if not os.path.exists(preview_mp4) or os.path.getsize(preview_mp4) < 1024:
        if os.path.exists(preview_gif) and os.path.getsize(preview_gif) > 1024:
            preview_path = preview_gif
            print("[Video] Using GIF preview fallback.")
        else:
            preview_path = None
    print(f"[Video] Preview file: {preview_path}")

    zip_filename = f"game_assets_{job_id}.zip"
    zip_path = os.path.join(job_dir, zip_filename)
    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for fname in os.listdir(job_dir):
                if fname != zip_filename:
                    zipf.write(os.path.join(job_dir, fname), fname)
    except Exception as e:
        print(f"Zip packaging note: {e}")

    # Use timestamp query parameters to prevent frontend asset caching issues
    ts = str(uuid.uuid4())[:6]
    result = {
        "job_id": job_id,
        "user_description": user_description,
        "vision_analysis": vision_info,
        "detected_objects": vision_info.get("objects", []),
        "resolved_genre": genre_resolution,
        "confidence": genre_resolution.get("confidence", 1.0),
        "game_plan": game_plan,
        "layout": layout,
        "playability": playability_info,
        "florence_caption": florence_caption,
        "florence_od": florence_od,
        "urls": {
            "scene":   f"/files/{job_id}/scene.png?v={ts}",
            "preview": (f"/files/{job_id}/{os.path.basename(preview_path)}?v={ts}"
                        if preview_path else None),
        },
        "zip_url": f"/files/{job_id}/{zip_filename}?v={ts}"
    }

    set_job_status(job_id, "Completed", 100, "Level generation finished successfully", result=result)
    return result

# --------------------------------------------------------------------------
# FastAPI Web App Routes
# --------------------------------------------------------------------------

app = FastAPI(title="Sketch-to-Game API")

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
        "lora_florence": FLORENCE_LORA_PATH,
        "lora_sdxl": SDXL_LORA_PATH
    }


@app.post("/generate")
async def generate_endpoint(
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

    set_job_status(job_id, "Initializing pipeline...", 2, "Received uploaded sketch file")

    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        executor,
        run_full_pipeline,
        temp_sketch_path,
        description,
        job_id,
        "http://localhost:7860"
    )

    return {"job_id": job_id, "status": "processing"}


@app.get("/status/{job_id}")
def status_endpoint(job_id: str):
    info = JOB_STATUS.get(job_id, {"status": "not_found", "progress": 0, "step": "Unknown job"})
    return JSONResponse(content=info)


@app.get("/download-status")
def download_status_endpoint():
    api_key = os.environ.get("OPENAI_API_KEY", "")
    key_configured = bool(api_key and len(api_key) > 8)
    key_preview = f"sk-...{api_key[-4:]}" if key_configured else "Not set"

    return {
        "downloaded_gb": 6.5 if DEVICE == "cuda" else 0.5,
        "target_gb": 6.5,
        "percent": 100 if DEVICE == "cuda" else 100,
        "speed_mbps": 0,
        "file_name": "Models cached & ready on device",
        "api_key_configured": key_configured,
        "api_key_preview": key_preview,
        "is_cached": True,
        "is_downloading": False
    }


@app.post("/start-download")
def start_download_endpoint():
    return {"status": "cached", "message": "All models are pre-loaded or ready on device."}


@app.get("/download-assets/{job_id}")
def download_assets_endpoint(job_id: str):
    job_dir = os.path.join(API_OUTPUT_DIR, job_id)
    zip_path = os.path.join(job_dir, f"game_assets_{job_id}.zip")
    if not os.path.exists(zip_path) and os.path.exists(job_dir):
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for fname in os.listdir(job_dir):
                if fname != f"game_assets_{job_id}.zip":
                    zipf.write(os.path.join(job_dir, fname), fname)
    if os.path.exists(zip_path):
        return FileResponse(zip_path, filename=f"game_assets_{job_id}.zip", media_type="application/zip")
    return JSONResponse(status_code=404, content={"error": f"Job assets for '{job_id}' not found"})


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return JSONResponse(status_code=204, content={})

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

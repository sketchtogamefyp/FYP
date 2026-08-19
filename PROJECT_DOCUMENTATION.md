# 🎮 Sketch-to-Game FYP: Complete Project Guide & Documentation

> **Simple, Easy-to-Understand Guide** explaining how the whole system works, every file and folder in the project, the technologies used, and which AI models/APIs are running under the hood.

---

## 📖 Table of Contents
1. [What is this Project?](#1-what-is-this-project)
2. [How Does it Work? (Step-by-Step Flow)](#2-how-does-it-work-step-by-step-flow)
3. [The 9 Supported Game Genres](#3-the-9-supported-game-genres)
4. [Tech Stack (Technologies Used)](#4-tech-stack-technologies-used)
5. [Which APIs & AI Models Are Used?](#5-which-apis--ai-models-are-used)
6. [Complete File & Folder Guide (FYP Main Folder)](#6-complete-file--folder-guide-fyp-main-folder)
7. [Complete File & Folder Guide (Extra Files & Research Folders)](#7-complete-file--folder-guide-extra-files--research-folders)
8. [How to Run the Project Locally](#8-how-to-run-the-project-locally)

---

## 1. What is this Project?

**Sketch-to-Game** is an AI-powered system that takes a **simple hand-drawn pencil/digital sketch** of a game level and a **short text description**, and automatically turns it into:
1. **Game Level Layout**: Detects platforms, jumping steps, hazards/spikes, start position, and goal flag.
2. **Individual Game Sprites**: Generates high-quality Player, Enemy/Rival, Ground/Platform Tile, and Background artwork.
3. **Full Scene Composite (`scene.png`)**: Puts all the generated assets together onto the extracted sketch layout with a retro arcade HUD.
4. **12-Second Gameplay Video (`preview.mp4`)**: Creates an authentic 180-frame animated gameplay video showing real player movement, jumping over spikes, hitting coin blocks, defeating enemies, and reaching the victory goal.
5. **Downloadable Game ZIP Package**: Packages all images, videos, and JSON game configuration files into a verified `.zip` file for game engines (Unity, Godot, Pygame, Web).

---

## 2. How Does it Work? (Step-by-Step Flow)

```
[ User Uploads Sketch & Description ]
                 │
                 ▼
[ Step 1: Computer Vision & AI Vision Analysis ]
  • OpenCV detects drawn platform bars, spikes, player circle, and goal flag.
  • GPT-4o Vision describes the visual context and theme.
                 │
                 ▼
[ Step 2: Intent & Genre Resolution ]
  • Resolves user text into 1 of 9 exact genres (Mario, Racing, Fighting, etc.).
                 │
                 ▼
[ Step 3: Structured Game Planning ]
  • GPT-4o-mini generates physics (gravity, jump force), health, lives, rules, and art prompts.
                 │
                 ▼
[ Step 4: Asset Generation ]
  • Generates 4 clean sprites: Player, Enemy, Platform Tile, Background.
  • Uses SDXL (GPU mode) or Custom Multi-Layer Procedural Art Engine (CPU mode).
  • Auto-crops solid white backgrounds into transparent RGBA sprites.
                 │
                 ▼
[ Step 5: Scene Composition ]
  • Places platforms, hazards, blocks, goal flagpole, characters, and HUD into scene.png.
                 │
                 ▼
[ Step 6: 12-Second Interactive Gameplay Video ]
  • Renders 180 frames @ 15fps streamed to ffmpeg/H.264 (jumping, nitro boosts, combos, K.O.).
                 │
                 ▼
[ Step 7: Packaging & Download Endpoints ]
  • Creates game_assets_<job_id>.zip with testzip() integrity verification.
  • Frontend displays live progress, interactive video preview, sprite gallery, and download button.
```

---

## 3. The 9 Supported Game Genres

| # | Genre | Description | Key Visuals & 12s Video Interaction |
|---|---|---|---|
| 1 | **Mario / Platformer** | Colorful jumping and obstacle platformer. | Plumber hero with red cap & blue overalls, Goomba enemy, `[ ? ]` coin blocks, spike dodging, stomping enemies, and goal flagpole banner (**★ COURSE CLEAR! ★**). |
| 2 | **Racing** | Fast-paced supercar circuit racing. | Red & Blue supercars (rear-3/4 angle, aero diffuser, quad exhausts, GT wings), "3..2..1..GO!" countdown, tire burnout smoke, nitro boost flames, and overtake (**🏆 1ST PLACE 🏆**). |
| 3 | **Fighting** | 1v1 martial arts combat. | Ryu martial artist vs Shadow Cyber Ninja in a Japanese dojo, "ROUND 1 - FIGHT!", 3-hit combo sparks, energy fireball blast, and K.O. slow-mo (**⚡ K.O.! - YOU WIN! ⚡**). |
| 4 | **Adventure** | Exploration and treasure discovery. | Explorer with fedora & glowing torch exploring an ancient forest, solving rune puzzles, opening glowing treasure chests (**🌟 NEW AREA DISCOVERED! 🌟**). |
| 5 | **Dungeon** | Dark fantasy dungeon combat. | Armored Paladin Knight with runic glowing sword & shield vs Skeleton Warrior in a stone dungeon hall (**⚔️ DUNGEON CLEARED! ⚔️**). |
| 6 | **Strategy** | Tactical base building & resource management. | Blue Mech Commander & Red Siege Tank on a grid map, mining crystals (+50 Ore), launching artillery barrage missiles (**🎖️ ENEMY BASE DESTROYED! 🎖️**). |
| 7 | **Tower Defense** | Strategic turret defense against creep waves. | High-tech plasma turrets on pedestals firing neon laser pulses at advancing creep monsters on a winding road (**🛡️ WAVE CLEARED - BASE DEFENDED! 🛡️**). |
| 8 | **Running** | Supersonic endless runner. | Cyber athlete sprinting on a 3-lane neon highway, dodging electric barriers, collecting gold energy ring streaks (**⚡ DISTANCE: 1500m ⚡**). |
| 9 | **Adventure Fighting** | Action RPG exploration with real-time combat. | Armored warrior exploring fortress ruins, dodging beast claws with backflips, and executing 3-hit Flame Whirlwind Combos (**🔥 ABILITY UNLOCKED! 🔥**). |

---

## 4. Tech Stack (Technologies Used)

### 🔹 Frontend (User Interface)
- **React.js (v18)**: Modern component-based web interface.
- **Vite**: Ultra-fast frontend development server and bundler.
- **Tailwind CSS**: Modern styling system for dark-themed, glassmorphic UI.
- **Lucide-React**: Clean vector icons for downloads, statuses, and controls.

### 🔹 Backend (Server & API)
- **Python 3.10**: Core programming language.
- **FastAPI**: Modern, asynchronous high-speed web framework for Python.
- **Uvicorn**: Lightning-fast ASGI production web server.
- **Asyncio & ThreadPoolExecutor**: Non-blocking background job queue handling heavy generation tasks smoothly.

### 🔹 Computer Vision & Image Processing
- **OpenCV (`cv2`)**: Binarization, contour detection, and shape analysis to extract platforms and obstacles from user sketches.
- **Pillow (`PIL`)**: Image drawing, multi-layer sprite creation, alpha masking, cropping, and scene composition.
- **NumPy**: Matrix and pixel array manipulations.

### 🔹 Video Generation & Streaming
- **imageio & imageio-ffmpeg**: Direct frame-by-frame streaming to H.264 MP4 (with automatic GIF fallback).
- **libx264 (ultrafast preset)**: Fast encoding that stays under 512MB RAM usage.

### 🔹 Deep Learning & AI
- **PyTorch**: Deep learning framework.
- **HuggingFace Transformers & Diffusers**: Model pipelines and scheduler configurations.
- **PEFT (LoRA)**: Parameter-Efficient Fine-Tuning for custom models.

---

## 5. Which APIs & AI Models Are Used?

1. **OpenAI API (`gpt-4o` & `gpt-4o-mini`)**:
   - **GPT-4o Vision**: Understands the visual contents of the uploaded sketch image.
   - **GPT-4o-mini**: Generates structured, genre-specific game design plans (physics constants, health, rules, asset prompts) as clean JSON.
2. **Microsoft Florence-2-base (`microsoft/Florence-2-base`)**:
   - Vision-language foundation model used for zero-shot object detection (`<OD>`) and detailed image captioning (`<MORE_DETAILED_CAPTION>`).
   - Fine-tuned with custom LoRA weights for sketch game level recognition.
3. **Stability AI SDXL 1.0 (`stabilityai/stable-diffusion-xl-base-1.0`)**:
   - State-of-the-art text-to-image model used on GPU environments to generate pixel-art game sprites with `nerijs/pixel-art-xl` LoRA.
4. **Custom Procedural 2D Generation Engine**:
   - High-performance, zero-RAM CPU engine that builds high-detail vector-shaded sprites (cars, plumbers, fighters, knights, mechs, turrets) instantly without requiring a dedicated graphics card.

---

## 6. Complete File & Folder Guide (`FYP/` Main Folder)

This is the main working codebase deployed to GitHub and production:

### 📁 `FYP/backend/`
- **`api_server.py`**: The **core brain of the backend**. Contains:
  - FastAPI server endpoints (`/generate`, `/status/{id}`, `/download-zip/{id}`, `/download/{id}/*`, `/download-status`).
  - Computer Vision sketch analyzer (`cv_extract_sketch_layout`).
  - Genre resolution logic for all 9 genres (`resolve_genre`).
  - Game planning system (`plan_game`).
  - High-detail procedural sprite generators for every genre.
  - Scene composition engine (`compose_scene`).
  - 12-second interactive video preview generator (`generate_preview_video`).
  - Atomic `.zip` packaging engine (`create_job_zip`).
- **`prompt_config.py`**: Prompt engineering kit defining prompt styles, negative prompts, guidance scale, and formatting rules for SDXL across all genres.
- **`requirements.txt`**: List of Python dependencies (FastAPI, uvicorn, opencv-python, pillow, imageio, openai, torch, diffusers, transformers).
- **`.env`**: Local environment variables file storing `OPENAI_API_KEY`, `DEVICE=cpu`, `PORT=7860`.
- **`download_all_models.py`**: Utility script to pre-download Florence-2 and SDXL weights locally.
- **`download_terminal.py`**: Terminal-based model download utility with progress display.
- **`test_pipeline_e2e.py`**: Automated end-to-end test suite testing procedural assets, scene composition, 120+ frame video generation, atomic ZIP packaging, and FastAPI routes.
- **`test_genre_resolution.py`**: 11 automated test cases verifying priority matching and keyword resolution for all genres.
- **`test_cv_layout.py`**: Unit test for OpenCV sketch contour and platform extraction.
- **`test_mario_sprites.py`**: Test script for Mario hero and Goomba sprite drawing.
- **`test_all_sprites.py`**: Test script validating procedural drawing for all 9 genres.
- **`test_api_mario_gen.py`**: Integration test sending a platformer sketch to the live `/generate` API.
- **`test_api_dungeon_gen.py`**: Integration test sending a dungeon sketch to the live `/generate` API.
- **`api_output/`**: Directory where generated game files (`scene.png`, `preview.mp4`, `best_player.png`, `game_assets_*.zip`) are saved for each job ID.

### 📁 `FYP/frontend/`
- **`src/App.jsx`**: The **main React component**. Contains:
  - Drag-and-drop sketch upload area and prompt text box.
  - Live generation status progress bar with step-by-step updates.
  - Interactive 12-second video player preview.
  - Scene composite viewer.
  - Individual sprite gallery (`Player`, `Enemy`, `Tile`, `Background`).
  - One-click ZIP download button.
  - Automatic fallback between local development (`http://localhost:7860`) and Render production backend (`https://fyp-1nwy.onrender.com`).
- **`src/main.jsx`**: React root entry point that mounts `App.jsx` into the HTML DOM.
- **`src/index.css`**: Tailwind CSS styles, animations, and custom scrollbars.
- **`index.html`**: Main HTML page with viewport and title metadata.
- **`package.json`**: Frontend project metadata, npm scripts (`dev`, `build`, `preview`), and React dependencies.
- **`vite.config.js`**: Vite configuration file setting up React plugin and local dev server port.

### 📁 `FYP/florence_game_lora/`
- **`final/`**: Saved LoRA checkpoint weights for Florence-2, fine-tuned specifically on game sketches and level bounding boxes.

### 📁 `FYP/.gitignore`
- Git ignore file keeping git history clean by excluding `node_modules/`, `api_output/`, `*.zip`, `*.mp4`, `.env`, and model binaries.

---

## 7. Complete File & Folder Guide (Extra Files & Research Folders)

These folders contain the training scripts, research experiments, reports, and dataset tools used during the FYP development:

### 📁 `FYP_Extra_Files/`
- **`retrain_florence.py`**: PyTorch training script used to fine-tune Florence-2 with LoRA on sketch level datasets.
- **`generate_dataset_all_games.py`**: Synthetic data generator that creates paired sketch images and bounding-box layout annotations for training.
- **`Game_FYP_Week8_9_ram_fix.ipynb`**: Jupyter Notebook documenting weekly sprint progress, experiments with RAM optimization, and memory benchmark tests.
- **`backend_test_*.py`**: Experimental scratch test scripts used during the evolution of video generation and sprite renderers.
- **`florence_game_lora_checkpoints/`**: Intermediate training checkpoints (`checkpoint-300`, `checkpoint-350`, `checkpoint-360`) saved during Florence-2 fine-tuning.
- **`sample_*.png`**: Sample generated reference outputs for racing, platformer, fighting, and shooter genres.
- **`test_sketch_*.png`**: Test sketches drawn for all genres (adventure, dungeon, fighting, platformer, racing, running, strategy, tower defense).

### 📁 `dataset/`
- **`rebuild_dataset.py`**: Dataset builder script that parses synthetic and drawn level sketches, crops bounding boxes, and formats them into Florence-2 JSONL training format.
- **`dataset/`**: Folder containing images and ground-truth JSON files for training.

### 📁 Root Directory (`Game/`)
- **`Game FYP.ipynb`**: Complete master Jupyter notebook containing research experiments, early prototypes, and visual evaluation.
- **`Sketch-to-Game_Report.docx` & `FYP game.docx`**: Formal academic project reports and documentation written for final year evaluation.
- **`Car.mp4` & `preview.mp4`**: Sample exported gameplay video demonstrations.

---

## 8. How to Run the Project Locally

### 1. Start the Backend Server
```powershell
cd D:\Programming\FYP\Game\FYP\backend
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python api_server.py
```
*The backend will start at: `http://localhost:7860`*

### 2. Start the Frontend Application
```powershell
cd D:\Programming\FYP\Game\FYP\frontend
npm install
npm run dev
```
*The frontend will start at: `http://localhost:5173`*

### 3. Open in Browser
Visit **`http://localhost:5173`** in your browser, upload any sketch, type your prompt, and watch your game come to life!

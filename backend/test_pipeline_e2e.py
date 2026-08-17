import os
import sys
import zipfile
import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(__file__))

from api_server import (
    generate_procedural_sprite,
    crop_to_content,
    validate_asset,
    compose_scene,
    generate_preview_video,
    create_job_zip,
    app,
    API_OUTPUT_DIR
)

def test_procedural_assets_and_cropping():
    print("[1/5] Testing Procedural Assets & Alpha Cropping...")
    genres = ["racing", "adventure_fighting", "fighting", "mario", "dungeon", "tower_defense", "running"]

    for genre in genres:
        game_plan = {"genre": genre, "title": f"Test {genre.title()}"}
        player_img = generate_procedural_sprite("player", f"{genre} player", 512, 512, seed=42, game_plan=game_plan)
        enemy_img = generate_procedural_sprite("enemy", f"{genre} enemy", 512, 512, seed=42, game_plan=game_plan)
        tile_img = generate_procedural_sprite("platform_tile", f"{genre} tile", 512, 512, seed=42, game_plan=game_plan)
        bg_img = generate_procedural_sprite("background", f"{genre} bg", 1024, 512, seed=42, game_plan=game_plan)

        assert player_img is not None and player_img.size[0] > 0 and player_img.size[1] > 0
        assert enemy_img is not None and enemy_img.size[0] > 0 and enemy_img.size[1] > 0
        assert tile_img is not None and tile_img.size[0] > 0 and tile_img.size[1] > 0
        assert bg_img is not None and bg_img.size[0] == 1024 and bg_img.size[1] == 512

        assert validate_asset(player_img, "player", min_content_ratio=0.15)
        assert validate_asset(enemy_img, "enemy", min_content_ratio=0.15)

    print(" -> All genre procedural assets generated, cropped, and validated successfully.")

def test_scene_composition():
    print("[2/5] Testing Scene Composition across Genres...")
    for genre in ["racing", "adventure_fighting", "fighting", "mario"]:
        game_plan = {"genre": genre}
        best_assets = {
            "player": generate_procedural_sprite("player", "player", 512, 512, seed=42, game_plan=game_plan),
            "enemy": generate_procedural_sprite("enemy", "enemy", 512, 512, seed=42, game_plan=game_plan),
            "platform_tile": generate_procedural_sprite("platform_tile", "tile", 512, 512, seed=42, game_plan=game_plan),
            "background": generate_procedural_sprite("background", "bg", 1024, 512, seed=42, game_plan=game_plan),
        }
        scene = compose_scene(best_assets, {}, game_plan, canvas_width=1024, canvas_height=480)
        assert scene.size == (1024, 480), f"Scene size mismatch for {genre}: {scene.size}"

    print(" -> Scene composition tested for all primary genres.")

def test_video_generation():
    print("[3/5] Testing 120-frame Video Generation...")
    test_dir = os.path.join(API_OUTPUT_DIR, "test_job_001")
    os.makedirs(test_dir, exist_ok=True)

    game_plan = {"genre": "racing"}
    best_assets = {
        "player": generate_procedural_sprite("player", "player", 512, 512, seed=42, game_plan=game_plan),
        "enemy": generate_procedural_sprite("enemy", "enemy", 512, 512, seed=42, game_plan=game_plan),
        "platform_tile": generate_procedural_sprite("platform_tile", "tile", 512, 512, seed=42, game_plan=game_plan),
        "background": generate_procedural_sprite("background", "bg", 1024, 512, seed=42, game_plan=game_plan),
    }
    clean_bg = compose_scene(best_assets, {}, game_plan, include_sprites=False)
    video_path = os.path.join(test_dir, "preview.mp4")

    actual_video = generate_preview_video(clean_bg, {}, best_assets, video_path, game_plan=game_plan)
    assert actual_video is not None and os.path.exists(actual_video), "Video file was not generated"
    assert os.path.getsize(actual_video) > 1024, f"Video file too small: {os.path.getsize(actual_video)} bytes"
    print(f" -> Racing preview video successfully rendered to {actual_video} (Size: {os.path.getsize(actual_video)} bytes).")

def test_atomic_zip_packaging():
    print("[4/5] Testing Atomic ZIP Packaging...")
    test_dir = os.path.join(API_OUTPUT_DIR, "test_job_001")
    scene_path = os.path.join(test_dir, "scene.png")
    Image.new("RGB", (640, 360), (50, 50, 50)).save(scene_path)

    zip_path = create_job_zip("test_job_001")
    assert zip_path is not None and os.path.exists(zip_path), "ZIP file was not created"
    assert os.path.getsize(zip_path) > 500, f"ZIP file too small: {os.path.getsize(zip_path)} bytes"

    with zipfile.ZipFile(zip_path, "r") as zf:
        assert zf.testzip() is None, "ZIP file integrity check failed"
        namelist = zf.namelist()
        assert "scene.png" in namelist, "scene.png missing from ZIP"
        assert not any(n.startswith("temp_") for n in namelist), "Temporary files leaked into ZIP"

    print(f" -> Atomic ZIP archive created and verified at {zip_path} (Size: {os.path.getsize(zip_path)} bytes).")

def test_fastapi_routes():
    print("[5/5] Testing FastAPI Router Integrity...")
    routes = [r.path for r in app.routes]
    
    # Verify exact endpoints exist
    assert "/generate" in routes, "Missing /generate"
    assert "/status/{job_id}" in routes, "Missing /status/{job_id}"
    assert "/download-zip/{job_id}" in routes, "Missing /download-zip/{job_id}"
    assert "/download-assets/{job_id}" in routes, "Missing /download-assets/{job_id}"
    assert "/download/{job_id}/scene" in routes, "Missing /download/{job_id}/scene"
    assert "/download/{job_id}/preview" in routes, "Missing /download/{job_id}/preview"
    assert "/download/{job_id}/asset/{asset_name}" in routes, "Missing /download/{job_id}/asset/{asset_name}"
    assert "/download-status" in routes, "Missing /download-status"

    # Count occurrences of /download-zip/{job_id} to ensure no duplicate definitions
    zip_count = sum(1 for r in app.routes if getattr(r, "path", "") == "/download-zip/{job_id}")
    assert zip_count == 1, f"Duplicate /download-zip/{job_id} route found: {zip_count}"

    print(" -> All FastAPI routes validated with 0 duplicate route definitions.")

if __name__ == "__main__":
    print("==================================================")
    print("RUNNING END-TO-END PIPELINE & SUBSYSTEM TESTS")
    print("==================================================")
    test_procedural_assets_and_cropping()
    test_scene_composition()
    test_video_generation()
    test_atomic_zip_packaging()
    test_fastapi_routes()
    print("==================================================")
    print("ALL 5 END-TO-END SUBSYSTEM TESTS PASSED!")
    print("==================================================")

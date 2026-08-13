import sys
import os
import json

# Setup import path
sys.path.append(os.path.dirname(__file__))

from api_server import resolve_genre

def run_tests():
    print("==================================================")
    print("Running Automated Genre Resolution Tests")
    print("==================================================")
    
    # TEST 1: Racing
    user_desc = "Create a racing game with sleek supercars"
    caption = "A sketch of a racing car on a highway road track"
    od = "car road track"
    layout = {"platforms": [], "enemies": [[10, 8]]}
    vision_info = {"visual_genre_evidence": ["car", "road", "track"], "objects": []}
    res = resolve_genre(user_desc, caption, od, layout, vision_info)
    assert res["genre"] == "racing", f"Test 1 failed: {res}"
    print("[PASS] Test 1: Racing resolved correctly.")
    
    # TEST 2: Fighting
    user_desc = "Create a fighting game with dual fighters"
    caption = "Two martial art combat fighters facing each other"
    od = "fighter opponent"
    layout = {"platforms": [[x, 10] for x in range(24)], "enemies": [[18, 9]]}
    vision_info = {"visual_genre_evidence": ["fighter", "opponent", "arena"], "objects": []}
    res = resolve_genre(user_desc, caption, od, layout, vision_info)
    assert res["genre"] == "fighting", f"Test 2 failed: {res}"
    print("[PASS] Test 2: Fighting resolved correctly.")

    # TEST 3: Adventure
    user_desc = "Create an adventure game"
    caption = "A hero exploring a deep forest with a hidden treasure chest"
    od = "character tree chest"
    layout = {"platforms": [[x, 10] for x in range(24)], "enemies": []}
    vision_info = {"visual_genre_evidence": ["character", "tree", "chest", "forest"], "objects": []}
    res = resolve_genre(user_desc, caption, od, layout, vision_info)
    assert res["genre"] == "adventure", f"Test 3 failed: {res}"
    print("[PASS] Test 3: Adventure resolved correctly.")

    # TEST 4: Dungeon
    user_desc = "Create a dungeon game"
    caption = "A top down view of connected rooms, corridors, and doors"
    od = "rooms corridors doors enemies"
    layout = {"platforms": [], "enemies": [[12, 6]]}
    vision_info = {"visual_genre_evidence": ["rooms", "corridors", "doors", "enemies", "dungeon"], "objects": []}
    res = resolve_genre(user_desc, caption, od, layout, vision_info)
    assert res["genre"] == "dungeon", f"Test 4 failed: {res}"
    print("[PASS] Test 4: Dungeon resolved correctly.")

    # TEST 5: Strategy
    user_desc = "Create a strategy game"
    caption = "Top down tactical map showing bases, resource mines, and armies"
    od = "base resources units"
    layout = {"platforms": [], "enemies": []}
    vision_info = {"visual_genre_evidence": ["base", "resources", "units", "strategy"], "objects": []}
    res = resolve_genre(user_desc, caption, od, layout, vision_info)
    assert res["genre"] == "strategy", f"Test 5 failed: {res}"
    print("[PASS] Test 5: Strategy resolved correctly.")

    # TEST 6: Mario-style (Platformer)
    user_desc = "Create a Mario-style game"
    caption = "A classic level with floating brick platforms and coin blocks"
    od = "platforms character enemies"
    layout = {"platforms": [[2, 8], [3, 8]], "enemies": [[5, 7]]}
    vision_info = {"visual_genre_evidence": ["platform", "character", "enemies"], "objects": []}
    res = resolve_genre(user_desc, caption, od, layout, vision_info)
    assert res["genre"] == "platformer", f"Test 6 failed: {res}"
    print("[PASS] Test 6: Platformer resolved correctly.")

    # TEST 7: Tower Defense
    user_desc = "Create a tower defense game"
    caption = "An winding path with defense towers, cannons, and base shield"
    od = "path towers base waves"
    layout = {"platforms": [], "enemies": []}
    vision_info = {"visual_genre_evidence": ["path", "towers", "base", "waves", "defense"], "objects": []}
    res = resolve_genre(user_desc, caption, od, layout, vision_info)
    assert res["genre"] == "tower_defense", f"Test 7 failed: {res}"
    print("[PASS] Test 7: Tower Defense resolved correctly.")

    # TEST 8: Running
    user_desc = "Create a running game"
    caption = "Side scrolling infinite runner with lanes and spikes"
    od = "runner lanes obstacles"
    layout = {"platforms": [], "enemies": [[8, 6]]}
    vision_info = {"visual_genre_evidence": ["runner", "lanes", "obstacles"], "objects": []}
    res = resolve_genre(user_desc, caption, od, layout, vision_info)
    assert res["genre"] == "running", f"Test 8 failed: {res}"
    print("[PASS] Test 8: Running resolved correctly.")

    # TEST 9: Contradictory Input (Racing User vs Fighting Sketch)
    user_desc = "Create a racing game"
    caption = "Two martial artist fighters punching each other in a combat arena"
    od = "fighters arena"
    layout = {"platforms": [[x, 10] for x in range(24)], "enemies": [[16, 9]]}
    vision_info = {"visual_genre_evidence": ["fighter", "arena"], "objects": []}
    res = resolve_genre(user_desc, caption, od, layout, vision_info)
    assert res["genre"] == "racing", f"Test 9 failed: Resolved genre should be racing. Res: {res}"
    assert res["visual_conflict"] == True, f"Test 9 failed: Visual conflict flag not set. Res: {res}"
    print("[PASS] Test 9: Contradiction handled properly (user requested genre preserved, visual conflict logged).")

    print("\nALL 9 GENRE RESOLUTION TESTS PASSED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    run_tests()

import sys
import os
import json

# Setup import path
sys.path.append(os.path.dirname(__file__))

from api_server import resolve_genre

def run_tests():
    print("==================================================")
    print("Running Comprehensive Genre Resolution Tests")
    print("==================================================")
    
    # TEST 1: Racing
    user_desc = "Create a racing game with two sports cars on a highway"
    caption = "A sketch of a racing car on a highway road track"
    od = "car road track"
    layout = {"platforms": [], "enemies": [[10, 8]]}
    vision_info = {"visual_genre_evidence": ["car", "road", "track"], "objects": []}
    res = resolve_genre(user_desc, caption, od, layout, vision_info)
    assert res["genre"] == "racing", f"Test 1 failed: {res}"
    print("[PASS] Test 1: Racing resolved correctly.")
    
    # TEST 2: Fighting
    user_desc = "Create a fighting game with dual martial arts fighters"
    caption = "Two combat fighters facing each other"
    od = "fighter opponent"
    layout = {"platforms": [[x, 10] for x in range(24)], "enemies": [[18, 9]]}
    vision_info = {"visual_genre_evidence": ["fighter", "opponent", "arena"], "objects": []}
    res = resolve_genre(user_desc, caption, od, layout, vision_info)
    assert res["genre"] == "fighting", f"Test 2 failed: {res}"
    print("[PASS] Test 2: Fighting resolved correctly.")

    # TEST 3: Adventure Fighting (MUST NOT be reduced to fighting)
    user_desc = "make an adventure game with sword fighting in a dungeon"
    caption = "A hero exploring a dungeon and fighting enemies"
    od = "character sword enemy"
    layout = {"platforms": [[x, 10] for x in range(24)], "enemies": [[15, 8]]}
    vision_info = {"visual_genre_evidence": ["character", "sword", "enemy", "dungeon"], "objects": []}
    res = resolve_genre(user_desc, caption, od, layout, vision_info)
    assert res["genre"] == "adventure_fighting", f"Test 3 failed: {res}"
    print("[PASS] Test 3: Adventure Fighting hybrid resolved correctly as 'adventure_fighting'.")

    # TEST 4: Pure Adventure
    user_desc = "Create an exploration adventure game"
    caption = "A hero exploring a deep forest with a hidden treasure chest"
    od = "character tree chest"
    layout = {"platforms": [[x, 10] for x in range(24)], "enemies": []}
    vision_info = {"visual_genre_evidence": ["character", "tree", "chest", "forest"], "objects": []}
    res = resolve_genre(user_desc, caption, od, layout, vision_info)
    assert res["genre"] == "adventure", f"Test 4 failed: {res}"
    print("[PASS] Test 4: Adventure resolved correctly.")

    # TEST 5: Dungeon
    user_desc = "Create a dungeon crawler game"
    caption = "A top down view of connected dungeon rooms, corridors, and doors"
    od = "rooms corridors doors enemies"
    layout = {"platforms": [], "enemies": [[12, 6]]}
    vision_info = {"visual_genre_evidence": ["rooms", "corridors", "doors", "enemies", "dungeon"], "objects": []}
    res = resolve_genre(user_desc, caption, od, layout, vision_info)
    assert res["genre"] == "dungeon", f"Test 5 failed: {res}"
    print("[PASS] Test 5: Dungeon resolved correctly.")

    # TEST 6: Strategy
    user_desc = "Create a strategy base building game"
    caption = "Top down tactical map showing bases, resource mines, and armies"
    od = "base resources units"
    layout = {"platforms": [], "enemies": []}
    vision_info = {"visual_genre_evidence": ["base", "resources", "units", "strategy"], "objects": []}
    res = resolve_genre(user_desc, caption, od, layout, vision_info)
    assert res["genre"] == "strategy", f"Test 6 failed: {res}"
    print("[PASS] Test 6: Strategy resolved correctly.")

    # TEST 7: Mario-style (Platformer)
    user_desc = "Create a Mario-style platformer game"
    caption = "A classic level with floating brick platforms and coin blocks"
    od = "platforms character enemies"
    layout = {"platforms": [[2, 8], [3, 8]], "enemies": [[5, 7]]}
    vision_info = {"visual_genre_evidence": ["platform", "character", "enemies"], "objects": []}
    res = resolve_genre(user_desc, caption, od, layout, vision_info)
    assert res["genre"] == "mario", f"Test 7 failed: {res}"
    print("[PASS] Test 7: Mario resolved correctly.")

    # TEST 8: Tower Defense
    user_desc = "Create a tower defense game"
    caption = "A winding path with defense towers, cannons, and base shield"
    od = "path towers base waves"
    layout = {"platforms": [], "enemies": []}
    vision_info = {"visual_genre_evidence": ["path", "towers", "base", "waves", "defense"], "objects": []}
    res = resolve_genre(user_desc, caption, od, layout, vision_info)
    assert res["genre"] == "tower_defense", f"Test 8 failed: {res}"
    print("[PASS] Test 8: Tower Defense resolved correctly.")

    # TEST 9: Running
    user_desc = "Create an endless runner game"
    caption = "Side scrolling infinite runner with lanes and spikes"
    od = "runner lanes obstacles"
    layout = {"platforms": [], "enemies": [[8, 6]]}
    vision_info = {"visual_genre_evidence": ["runner", "lanes", "obstacles"], "objects": []}
    res = resolve_genre(user_desc, caption, od, layout, vision_info)
    assert res["genre"] == "running", f"Test 9 failed: {res}"
    print("[PASS] Test 9: Running resolved correctly.")

    # TEST 10: Contradictory Input (Racing User vs Fighting Sketch)
    user_desc = "Create a racing game"
    caption = "Two martial artist fighters punching each other in a combat arena"
    od = "fighters arena"
    layout = {"platforms": [[x, 10] for x in range(24)], "enemies": [[16, 9]]}
    vision_info = {"visual_genre_evidence": ["fighter", "arena"], "objects": []}
    res = resolve_genre(user_desc, caption, od, layout, vision_info)
    assert res["genre"] == "racing", f"Test 10 failed: Resolved genre should be racing. Res: {res}"
    assert res["visual_conflict"] == True, f"Test 10 failed: Visual conflict flag not set. Res: {res}"
    print("[PASS] Test 10: Contradiction handled properly (user requested genre preserved, visual conflict logged).")

    # TEST 11: Adventure User vs Vehicle in Sketch (User text wins)
    user_desc = "make an adventure game"
    caption = "A landscape sketch with a vehicle on a road"
    od = "vehicle road tree"
    layout = {"platforms": [[x, 10] for x in range(24)], "enemies": []}
    vision_info = {"visual_genre_evidence": ["vehicle", "road", "tree"], "objects": []}
    res = resolve_genre(user_desc, caption, od, layout, vision_info)
    assert res["genre"] == "adventure", f"Test 11 failed: User requested adventure should not become car/racing. Res: {res}"
    print("[PASS] Test 11: Text priority preserved (adventure game did not become car game despite sketch).")

    print("\nALL 11 COMPREHENSIVE GENRE RESOLUTION TESTS PASSED!")
    print("==================================================")

if __name__ == "__main__":
    run_tests()

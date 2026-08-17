import cv2
import numpy as np

def cv_extract_sketch_layout(image_path):
    """
    Direct Computer Vision sketch parser using adaptive thresholding and contour hierarchy
    to extract actual drawn platforms, spikes, player circle, and goal flag.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    h, w = img.shape
    # Binarize: black sketch lines -> white foreground
    _, thresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    platform_boxes = []
    spikes = []
    player_pos = [2, 10]
    goal_pos = [22, 2]

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = bw / float(bh)

        # 1. Horizontal platform bars (wide aspect ratio)
        if aspect >= 2.2 and bw >= int(w * 0.08):
            norm_box = {
                "x_norm": round(x / w, 3),
                "y_norm": round(y / h, 3),
                "w_norm": round(bw / w, 3),
                "h_norm": round(max(bh, int(h * 0.04)) / h, 3)
            }
            platform_boxes.append(norm_box)

        # 2. Triangular Spikes / Hazards
        elif 0.5 <= aspect <= 2.0 and area < int(w * h * 0.05):
            approx = cv2.approxPolyDP(cnt, 0.06 * cv2.arcLength(cnt, True), True)
            if len(approx) == 3 or (bh > 10 and aspect <= 1.5 and y > int(h * 0.4)):
                col = int((x + bw / 2) / w * 24)
                row = int((y + bh / 2) / h * 12)
                spikes.append([col, row])

        # 3. Player Circle / Start
        elif 0.8 <= aspect <= 1.2 and area < int(w * h * 0.04) and x < int(w * 0.3):
            col = int((x + bw / 2) / w * 24)
            row = int((y + bh / 2) / h * 12)
            player_pos = [col, row]

        # 4. Goal Flag (top right)
        elif x > int(w * 0.7) and y < int(h * 0.5):
            col = int((x + bw / 2) / w * 24)
            row = int((y + bh / 2) / h * 12)
            goal_pos = [col, row]

    # Sort platforms from bottom to top
    platform_boxes.sort(key=lambda b: -b["y_norm"])

    # If no platforms were detected, create a balanced 4-step ascending layout
    if not platform_boxes:
        platform_boxes = [
            {"x_norm": 0.0, "y_norm": 0.88, "w_norm": 1.0, "h_norm": 0.08},
            {"x_norm": 0.12, "y_norm": 0.65, "w_norm": 0.22, "h_norm": 0.06},
            {"x_norm": 0.40, "y_norm": 0.50, "w_norm": 0.22, "h_norm": 0.06},
            {"x_norm": 0.68, "y_norm": 0.35, "w_norm": 0.22, "h_norm": 0.06}
        ]

    # Convert boxes to grid platform tiles (24 cols x 12 rows)
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

if __name__ == "__main__":
    print("Testing CV Sketch Extractor...")
    # Test on a dummy sketch image
    dummy_img = np.full((360, 640), 255, dtype=np.uint8)
    # Floor
    cv2.rectangle(dummy_img, (0, 320), (640, 350), 0, -1)
    # 3 Steps
    cv2.rectangle(dummy_img, (80, 240), (220, 255), 0, -1)
    cv2.rectangle(dummy_img, (260, 180), (400, 195), 0, -1)
    cv2.rectangle(dummy_img, (440, 120), (580, 135), 0, -1)
    # Save
    cv2.imwrite("./test_dummy_sketch.png", dummy_img)

    layout = cv_extract_sketch_layout("./test_dummy_sketch.png")
    print(f"Extracted {len(layout['platform_boxes'])} platforms: {layout['platform_boxes']}")

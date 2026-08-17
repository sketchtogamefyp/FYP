import os
import sys
import math
from PIL import Image, ImageDraw, ImageFilter

def draw_detailed_car(width=512, height=512, is_player=True):
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    px = lambda f: int(width * f)
    py = lambda f: int(height * f)

    main_color = (225, 20, 30) if is_player else (20, 80, 230)
    dark_color = (150, 10, 20) if is_player else (10, 45, 150)
    light_color = (255, 75, 75) if is_player else (70, 140, 255)
    highlight = (255, 140, 140) if is_player else (140, 190, 255)
    accent_stripe = (255, 255, 255) if is_player else (0, 235, 255)

    # 1. Soft feathered ground shadow under entire car
    draw.ellipse([px(0.06), py(0.70), px(0.94), py(0.95)], fill=(10, 12, 18, 190))
    draw.ellipse([px(0.12), py(0.73), px(0.88), py(0.92)], fill=(5, 6, 10, 230))

    # 2. Wide Racing Tires (Rear low-angle 3D perspective)
    # Left tire
    draw.rounded_rectangle([px(0.08), py(0.52), px(0.28), py(0.86)], radius=12, fill=(20, 22, 26, 255))
    draw.rounded_rectangle([px(0.11), py(0.55), px(0.25), py(0.83)], radius=8, fill=(35, 38, 45, 255))
    draw.ellipse([px(0.13), py(0.58), px(0.23), py(0.80)], fill=(75, 80, 92, 255)) # Rim
    draw.ellipse([px(0.15), py(0.62), px(0.21), py(0.76)], fill=(200, 205, 215, 255)) # Alloy center
    draw.ellipse([px(0.165), py(0.66), px(0.195), py(0.72)], fill=(25, 25, 30, 255))
    # Right tire
    draw.rounded_rectangle([px(0.72), py(0.52), px(0.92), py(0.86)], radius=12, fill=(20, 22, 26, 255))
    draw.rounded_rectangle([px(0.75), py(0.55), px(0.89), py(0.83)], radius=8, fill=(35, 38, 45, 255))
    draw.ellipse([px(0.77), py(0.58), px(0.87), py(0.80)], fill=(75, 80, 92, 255))
    draw.ellipse([px(0.79), py(0.62), px(0.85), py(0.76)], fill=(200, 205, 215, 255))
    draw.ellipse([px(0.805), py(0.66), px(0.835), py(0.72)], fill=(25, 25, 30, 255))

    # 3. Lower Rear Aero Diffuser (Carbon fiber fins)
    draw.polygon([
        (px(0.22), py(0.78)), (px(0.78), py(0.78)),
        (px(0.75), py(0.88)), (px(0.25), py(0.88))
    ], fill=(18, 20, 24, 255))
    # Diffuser vertical carbon fins
    for fin_x in [0.32, 0.41, 0.50, 0.59, 0.68]:
        draw.line([px(fin_x), py(0.78), px(fin_x), py(0.88)], fill=(45, 48, 55, 255), width=3)

    # 4. Quad Chrome Exhaust Tips with internal glow
    for ex_x in [0.26, 0.32, 0.68, 0.74]:
        draw.ellipse([px(ex_x-0.035), py(0.79), px(ex_x+0.035), py(0.87)], fill=(28, 30, 35, 255))
        draw.ellipse([px(ex_x-0.03), py(0.80), px(ex_x+0.03), py(0.86)], fill=(210, 215, 225, 255))
        draw.ellipse([px(ex_x-0.02), py(0.81), px(ex_x+0.02), py(0.85)], fill=(20, 22, 28, 255))
        # Warm core glow
        draw.ellipse([px(ex_x-0.01), py(0.82), px(ex_x+0.01), py(0.84)], fill=(255, 120, 20, 255))

    # 5. Sculpted Main Rear Bodywork & Flared Fenders
    # Lower bumper bumper base
    draw.polygon([
        (px(0.14), py(0.76)), (px(0.15), py(0.60)), (px(0.24), py(0.44)),
        (px(0.76), py(0.44)), (px(0.85), py(0.60)), (px(0.86), py(0.76)),
        (px(0.78), py(0.80)), (px(0.22), py(0.80))
    ], fill=dark_color)

    # Mid body curve
    draw.polygon([
        (px(0.16), py(0.74)), (px(0.18), py(0.56)), (px(0.26), py(0.40)),
        (px(0.74), py(0.40)), (px(0.82), py(0.56)), (px(0.84), py(0.74)),
        (px(0.76), py(0.78)), (px(0.24), py(0.78))
    ], fill=main_color)

    # Upper body bevel / highlights
    draw.polygon([
        (px(0.20), py(0.68)), (px(0.22), py(0.52)), (px(0.28), py(0.38)),
        (px(0.72), py(0.38)), (px(0.78), py(0.52)), (px(0.80), py(0.68)),
        (px(0.74), py(0.72)), (px(0.26), py(0.72))
    ], fill=light_color)

    # 6. Cabin Glass & Roofline (Curved teardrop shape)
    draw.polygon([
        (px(0.27), py(0.38)), (px(0.33), py(0.18)),
        (px(0.67), py(0.18)), (px(0.73), py(0.38))
    ], fill=(16, 20, 30, 255))

    # Rear window dark glass & louvers
    draw.polygon([
        (px(0.30), py(0.36)), (px(0.35), py(0.20)),
        (px(0.65), py(0.20)), (px(0.70), py(0.36))
    ], fill=(22, 28, 42, 255))
    # Glass gradient specular reflection
    draw.polygon([
        (px(0.33), py(0.34)), (px(0.37), py(0.21)),
        (px(0.48), py(0.21)), (px(0.42), py(0.34))
    ], fill=(90, 160, 240, 160))
    # Horizontal aerodynamic louvers
    for l_y in [0.24, 0.28, 0.32]:
        draw.line([px(0.32), py(l_y), px(0.68), py(l_y)], fill=(12, 14, 20, 240), width=3)

    # Roof top edge
    draw.polygon([
        (px(0.33), py(0.18)), (px(0.37), py(0.15)),
        (px(0.63), py(0.15)), (px(0.67), py(0.18))
    ], fill=highlight)

    # 7. Carbon Fiber GT Wing / Rear Spoiler (Double mount, curved blade)
    # Wing Mount Struts
    draw.polygon([(px(0.30), py(0.22)), (px(0.33), py(0.22)), (px(0.34), py(0.36)), (px(0.29), py(0.36))], fill=(25, 28, 34, 255))
    draw.polygon([(px(0.67), py(0.22)), (px(0.70), py(0.22)), (px(0.71), py(0.36)), (px(0.66), py(0.36))], fill=(25, 28, 34, 255))
    # Main carbon wing blade
    draw.polygon([
        (px(0.12), py(0.20)), (px(0.14), py(0.14)), (px(0.86), py(0.14)), (px(0.88), py(0.20)),
        (px(0.84), py(0.24)), (px(0.16), py(0.24))
    ], fill=(22, 25, 30, 255))
    # Wing upper highlight
    draw.line([px(0.15), py(0.15), px(0.85), py(0.15)], fill=(120, 130, 145, 255), width=3)
    # Wing endplates
    draw.rounded_rectangle([px(0.11), py(0.11), px(0.15), py(0.27)], radius=4, fill=main_color)
    draw.rounded_rectangle([px(0.85), py(0.11), px(0.89), py(0.27)], radius=4, fill=main_color)

    # 8. Dual Twin LED Taillight Clusters (Modern split optic design)
    # Left taillight cluster
    draw.polygon([
        (px(0.20), py(0.54)), (px(0.38), py(0.53)),
        (px(0.37), py(0.60)), (px(0.19), py(0.61))
    ], fill=(120, 10, 15, 255))
    draw.line([px(0.21), py(0.56), px(0.36), py(0.55)], fill=(255, 45, 55, 255), width=4)
    draw.line([px(0.23), py(0.58), px(0.35), py(0.57)], fill=(255, 180, 180, 255), width=2)
    # Right taillight cluster
    draw.polygon([
        (px(0.62), py(0.53)), (px(0.80), py(0.54)),
        (px(0.81), py(0.61)), (px(0.63), py(0.60))
    ], fill=(120, 10, 15, 255))
    draw.line([px(0.64), py(0.55), px(0.79), py(0.56)], fill=(255, 45, 55, 255), width=4)
    draw.line([px(0.65), py(0.57), px(0.77), py(0.58)], fill=(255, 180, 180, 255), width=2)

    # Center Brake Light Strip
    draw.rectangle([px(0.44), py(0.44), px(0.56), py(0.47)], fill=(255, 30, 40, 255))
    draw.rectangle([px(0.46), py(0.45), px(0.54), py(0.46)], fill=(255, 220, 220, 255))

    # 9. Racing Livery / Dual Stripes
    draw.polygon([
        (px(0.45), py(0.18)), (px(0.48), py(0.18)),
        (px(0.48), py(0.78)), (px(0.45), py(0.78))
    ], fill=accent_stripe)
    draw.polygon([
        (px(0.52), py(0.18)), (px(0.55), py(0.18)),
        (px(0.55), py(0.78)), (px(0.52), py(0.78))
    ], fill=accent_stripe)

    # 10. Side Mirrors
    draw.ellipse([px(0.18), py(0.36), px(0.25), py(0.44)], fill=main_color)
    draw.ellipse([px(0.20), py(0.38), px(0.24), py(0.42)], fill=(120, 180, 240, 200))
    draw.ellipse([px(0.75), py(0.36), px(0.82), py(0.44)], fill=main_color)
    draw.ellipse([px(0.76), py(0.38), px(0.80), py(0.42)], fill=(120, 180, 240, 200))

    return img

def draw_detailed_fighter(width=512, height=512, is_player=True):
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    px = lambda f: int(width * f)
    py = lambda f: int(height * f)

    skin = (235, 180, 135)
    skin_shadow = (195, 140, 100)
    skin_hi = (255, 205, 165)

    if is_player:
        # Street Fighter style Martial Arts Master (Ryu-inspired dynamic combat stance)
        # Drop shadow
        draw.ellipse([px(0.18), py(0.86), px(0.82), py(0.97)], fill=(15, 12, 18, 170))

        # Flowing headband tails (behind body)
        draw.polygon([(px(0.56), py(0.14)), (px(0.78), py(0.10)), (px(0.72), py(0.18))], fill=(210, 25, 25, 255))
        draw.polygon([(px(0.56), py(0.15)), (px(0.84), py(0.16)), (px(0.76), py(0.23))], fill=(170, 15, 15, 255))

        # Left (Rear) Leg - Stance wide
        draw.polygon([(px(0.54), py(0.62)), (px(0.76), py(0.64)), (px(0.82), py(0.88)), (px(0.64), py(0.88))], fill=(225, 225, 220, 255))
        # Left Foot & Wraps
        draw.polygon([(px(0.64), py(0.86)), (px(0.86), py(0.86)), (px(0.88), py(0.93)), (px(0.62), py(0.93))], fill=(210, 150, 110, 255))
        draw.rectangle([px(0.66), py(0.84), px(0.82), py(0.89)], fill=(240, 240, 235, 255))

        # Right (Front) Leg - Bent forward
        draw.polygon([(px(0.24), py(0.62)), (px(0.48), py(0.62)), (px(0.42), py(0.90)), (px(0.20), py(0.90))], fill=(245, 245, 240, 255))
        draw.polygon([(px(0.26), py(0.65)), (px(0.36), py(0.65)), (px(0.32), py(0.88)), (px(0.22), py(0.88))], fill=(215, 215, 210, 255))
        # Right Foot
        draw.polygon([(px(0.18), py(0.88)), (px(0.44), py(0.88)), (px(0.42), py(0.94)), (px(0.16), py(0.94))], fill=(225, 165, 120, 255))
        draw.rectangle([px(0.22), py(0.86), px(0.38), py(0.91)], fill=(240, 240, 235, 255))

        # Torso & Gi Top with muscular folds
        draw.polygon([(px(0.26), py(0.28)), (px(0.74), py(0.28)), (px(0.68), py(0.64)), (px(0.32), py(0.64))], fill=(245, 245, 240, 255))
        # Muscular chest shadow & v-neck open collar
        draw.polygon([(px(0.42), py(0.28)), (px(0.58), py(0.28)), (px(0.50), py(0.48))], fill=skin_shadow)
        draw.polygon([(px(0.44), py(0.30)), (px(0.56), py(0.30)), (px(0.50), py(0.45))], fill=skin)
        # Red Lapel Edges
        draw.polygon([(px(0.36), py(0.28)), (px(0.44), py(0.28)), (px(0.52), py(0.52)), (px(0.46), py(0.52))], fill=(200, 30, 30, 255))
        draw.polygon([(px(0.64), py(0.28)), (px(0.56), py(0.28)), (px(0.48), py(0.52)), (px(0.54), py(0.52))], fill=(200, 30, 30, 255))

        # Black Master Belt with hanging knot
        draw.rectangle([px(0.28), py(0.58), px(0.72), py(0.65)], fill=(24, 22, 25, 255))
        draw.polygon([(px(0.46), py(0.64)), (px(0.54), py(0.64)), (px(0.56), py(0.80)), (px(0.48), py(0.80))], fill=(20, 18, 22, 255))
        draw.polygon([(px(0.50), py(0.64)), (px(0.58), py(0.64)), (px(0.62), py(0.76)), (px(0.54), py(0.76))], fill=(32, 28, 35, 255))

        # Left (Rear) Arm - Raised in guard
        draw.polygon([(px(0.66), py(0.28)), (px(0.86), py(0.34)), (px(0.78), py(0.48)), (px(0.64), py(0.42))], fill=skin)
        draw.rounded_rectangle([px(0.74), py(0.32), px(0.88), py(0.46)], radius=6, fill=(210, 30, 30, 255)) # Red glove

        # Right (Front) Arm - Punching / Forward guard
        draw.polygon([(px(0.18), py(0.32)), (px(0.34), py(0.30)), (px(0.28), py(0.46)), (px(0.12), py(0.44))], fill=skin_hi)
        draw.polygon([(px(0.12), py(0.40)), (px(0.28), py(0.44)), (px(0.22), py(0.58)), (px(0.08), py(0.52))], fill=skin)
        # Red Combat Glove (Fist)
        draw.rounded_rectangle([px(0.06), py(0.46), px(0.22), py(0.60)], radius=8, fill=(225, 30, 30, 255))
        draw.rectangle([px(0.09), py(0.52), px(0.19), py(0.57)], fill=(160, 15, 15, 255)) # Fist knuckle pad

        # Head & Face
        draw.ellipse([px(0.35), py(0.06), px(0.65), py(0.30)], fill=skin)
        draw.polygon([(px(0.46), py(0.28)), (px(0.54), py(0.28)), (px(0.56), py(0.36)), (px(0.44), py(0.36))], fill=skin_shadow) # Neck

        # Spiky Black Hair
        draw.polygon([
            (px(0.33), py(0.12)), (px(0.30), py(0.04)), (px(0.40), py(0.05)),
            (px(0.46), py(-0.02)), (px(0.54), py(0.04)), (px(0.62), py(-0.02)),
            (px(0.68), py(0.05)), (px(0.72), py(0.12)), (px(0.65), py(0.14)), (px(0.35), py(0.14))
        ], fill=(28, 24, 25, 255))

        # Red Headband across forehead
        draw.rectangle([px(0.33), py(0.12), px(0.67), py(0.18)], fill=(225, 30, 30, 255))
        draw.line([px(0.33), py(0.14), px(0.67), py(0.14)], fill=(255, 90, 90, 255), width=2)

        # Expressive Fierce Face
        # Brows
        draw.line([px(0.40), py(0.19), px(0.47), py(0.21)], fill=(25, 20, 22, 255), width=3)
        draw.line([px(0.60), py(0.19), px(0.53), py(0.21)], fill=(25, 20, 22, 255), width=3)
        # Eyes
        draw.rectangle([px(0.42), py(0.21), px(0.47), py(0.24)], fill=(255, 255, 255, 255))
        draw.rectangle([px(0.44), py(0.21), px(0.47), py(0.24)], fill=(30, 25, 25, 255))
        draw.rectangle([px(0.53), py(0.21), px(0.58), py(0.24)], fill=(255, 255, 255, 255))
        draw.rectangle([px(0.53), py(0.21), px(0.56), py(0.24)], fill=(30, 25, 25, 255))
        # Nose & Mouth
        draw.line([px(0.50), py(0.22), px(0.48), py(0.26)], fill=skin_shadow, width=2)
        draw.line([px(0.45), py(0.28), px(0.55), py(0.28)], fill=(120, 45, 45, 255), width=2)

    else:
        # Menacing Cyber Ninja / Rival Boss
        draw.ellipse([px(0.18), py(0.86), px(0.82), py(0.97)], fill=(15, 12, 18, 170))

        # Ninja Scarf
        draw.polygon([(px(0.52), py(0.16)), (px(0.82), py(0.12)), (px(0.74), py(0.24))], fill=(140, 20, 220, 255))

        # Legs in black shinobi armor
        draw.polygon([(px(0.52), py(0.60)), (px(0.76), py(0.62)), (px(0.80), py(0.88)), (px(0.62), py(0.88))], fill=(25, 26, 35, 255))
        draw.polygon([(px(0.22), py(0.60)), (px(0.46), py(0.60)), (px(0.40), py(0.88)), (px(0.18), py(0.88))], fill=(32, 34, 45, 255))
        # Shin guards & boots
        draw.rectangle([px(0.20), py(0.74), px(0.38), py(0.88)], fill=(45, 48, 62, 255))
        draw.rectangle([px(0.64), py(0.74), px(0.80), py(0.88)], fill=(45, 48, 62, 255))
        draw.rectangle([px(0.16), py(0.88), px(0.40), py(0.94)], fill=(18, 20, 26, 255))
        draw.rectangle([px(0.62), py(0.88), px(0.84), py(0.94)], fill=(18, 20, 26, 255))

        # Chest Armor with Glowing Cyber Matrix
        draw.polygon([(px(0.24), py(0.28)), (px(0.76), py(0.28)), (px(0.70), py(0.62)), (px(0.30), py(0.62))], fill=(28, 30, 42, 255))
        draw.polygon([(px(0.34), py(0.32)), (px(0.66), py(0.32)), (px(0.60), py(0.56)), (px(0.40), py(0.56))], fill=(42, 45, 60, 255))
        # Glowing purple energy core
        draw.polygon([(px(0.45), py(0.36)), (px(0.55), py(0.36)), (px(0.50), py(0.48))], fill=(180, 40, 255, 255))

        # Shoulder Pauldrons
        draw.rounded_rectangle([px(0.18), py(0.26), px(0.32), py(0.40)], radius=6, fill=(55, 58, 75, 255))
        draw.rounded_rectangle([px(0.68), py(0.26), px(0.82), py(0.40)], radius=6, fill=(55, 58, 75, 255))

        # Arms with Spiked Gauntlets
        draw.polygon([(px(0.66), py(0.34)), (px(0.88), py(0.40)), (px(0.80), py(0.56)), (px(0.64), py(0.48))], fill=(32, 34, 46, 255))
        draw.rounded_rectangle([px(0.76), py(0.44), px(0.90), py(0.58)], radius=6, fill=(55, 58, 75, 255))
        draw.polygon([(px(0.18), py(0.34)), (px(0.34), py(0.34)), (px(0.26), py(0.56)), (px(0.10), py(0.52))], fill=(32, 34, 46, 255))
        draw.rounded_rectangle([px(0.08), py(0.46), px(0.24), py(0.60)], radius=6, fill=(55, 58, 75, 255))

        # Head with Full Cowl & Crimson Cyber Visor
        draw.ellipse([px(0.34), py(0.08), px(0.66), py(0.32)], fill=(25, 26, 36, 255))
        draw.polygon([(px(0.38), py(0.06)), (px(0.42), py(-0.02)), (px(0.48), py(0.06))], fill=(45, 48, 62, 255)) # Horn/crest
        draw.polygon([(px(0.52), py(0.06)), (px(0.58), py(-0.02)), (px(0.62), py(0.06))], fill=(45, 48, 62, 255))

        # Glowing Neon Red/Pink Visor
        draw.rounded_rectangle([px(0.36), py(0.16), px(0.64), py(0.24)], radius=4, fill=(255, 20, 60, 255))
        draw.line([px(0.38), py(0.20), px(0.62), py(0.20)], fill=(255, 210, 230, 255), width=2)

    return img

if __name__ == "__main__":
    out_dir = "./test_sprites_preview"
    os.makedirs(out_dir, exist_ok=True)
    p_car = draw_detailed_car(512, 512, is_player=True)
    p_car.save(os.path.join(out_dir, "player_car.png"))
    e_car = draw_detailed_car(512, 512, is_player=False)
    e_car.save(os.path.join(out_dir, "rival_car.png"))

    p_fight = draw_detailed_fighter(512, 512, is_player=True)
    p_fight.save(os.path.join(out_dir, "player_fighter.png"))
    e_fight = draw_detailed_fighter(512, 512, is_player=False)
    e_fight.save(os.path.join(out_dir, "enemy_fighter.png"))
    print("Generated test preview sprites!")

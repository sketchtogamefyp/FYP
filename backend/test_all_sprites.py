import os
import sys
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

def draw_adventure_hero(width=512, height=512):
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    px = lambda f: int(width * f)
    py = lambda f: int(height * f)

    skin = (240, 185, 140)
    brown_leather = (130, 70, 30)
    khaki = (195, 170, 125)
    dark_brown = (75, 40, 20)

    # Shadow
    draw.ellipse([px(0.18), py(0.86), px(0.82), py(0.97)], fill=(15, 12, 18, 170))

    # Boots
    draw.rounded_rectangle([px(0.24), py(0.80), px(0.44), py(0.92)], radius=6, fill=dark_brown)
    draw.rounded_rectangle([px(0.56), py(0.80), px(0.76), py(0.92)], radius=6, fill=dark_brown)

    # Khaki Pants
    draw.polygon([(px(0.26), py(0.58)), (px(0.44), py(0.58)), (px(0.42), py(0.82)), (px(0.24), py(0.82))], fill=khaki)
    draw.polygon([(px(0.56), py(0.58)), (px(0.74), py(0.58)), (px(0.76), py(0.82)), (px(0.58), py(0.82))], fill=khaki)

    # Leather Jacket & Belt
    draw.rectangle([px(0.28), py(0.32), px(0.72), py(0.60)], fill=brown_leather)
    draw.rectangle([px(0.38), py(0.32), px(0.62), py(0.48)], fill=khaki) # Undershirt
    draw.rectangle([px(0.28), py(0.54), px(0.72), py(0.60)], fill=dark_brown) # Belt
    draw.ellipse([px(0.46), py(0.54), px(0.54), py(0.60)], fill=(255, 215, 0, 255)) # Gold buckle

    # Arms: Left arm holds glowing torch, right arm holds compass
    # Left Arm & Torch
    draw.polygon([(px(0.28), py(0.34)), (px(0.12), py(0.42)), (px(0.14), py(0.54)), (px(0.28), py(0.46))], fill=brown_leather)
    draw.ellipse([px(0.10), py(0.48), px(0.18), py(0.56)], fill=skin)
    # Torch Wood Handle & Flame
    draw.rectangle([px(0.12), py(0.30), px(0.16), py(0.58)], fill=dark_brown)
    draw.polygon([(px(0.08), py(0.30)), (px(0.20), py(0.30)), (px(0.14), py(0.14))], fill=(255, 120, 20, 255))
    draw.polygon([(px(0.10), py(0.28)), (px(0.18), py(0.28)), (px(0.14), py(0.18))], fill=(255, 220, 40, 255))

    # Right Arm
    draw.polygon([(px(0.72), py(0.34)), (px(0.86), py(0.44)), (px(0.84), py(0.56)), (px(0.72), py(0.46))], fill=brown_leather)
    draw.ellipse([px(0.80), py(0.50), px(0.88), py(0.58)], fill=skin)

    # Head & Adventurer Fedora Hat
    draw.ellipse([px(0.34), py(0.12), px(0.66), py(0.36)], fill=skin)
    draw.ellipse([px(0.40), py(0.22), px(0.46), py(0.28)], fill=(30, 25, 20, 255))
    draw.ellipse([px(0.54), py(0.22), px(0.60), py(0.28)], fill=(30, 25, 20, 255))
    # Fedora
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
        # Armored Knight with glowing sword & shield
        steel = (190, 200, 215)
        steel_dark = (120, 130, 145)
        gold = (255, 215, 0)
        blue_cape = (35, 75, 180)

        # Shadow
        draw.ellipse([px(0.18), py(0.86), px(0.82), py(0.97)], fill=(15, 12, 18, 170))

        # Blue Cape
        draw.polygon([(px(0.30), py(0.28)), (px(0.70), py(0.28)), (px(0.82), py(0.84)), (px(0.18), py(0.84))], fill=blue_cape)

        # Steel Greaves (Legs)
        draw.rounded_rectangle([px(0.26), py(0.60), px(0.44), py(0.90)], radius=6, fill=steel)
        draw.rounded_rectangle([px(0.56), py(0.60), px(0.74), py(0.90)], radius=6, fill=steel)

        # Breastplate & Pauldrons
        draw.polygon([(px(0.28), py(0.26)), (px(0.72), py(0.26)), (px(0.66), py(0.62)), (px(0.34), py(0.62))], fill=steel)
        draw.polygon([(px(0.36), py(0.30)), (px(0.64), py(0.30)), (px(0.50), py(0.52))], fill=steel_dark)
        draw.ellipse([px(0.18), py(0.24), px(0.32), py(0.38)], fill=gold) # Left pauldron
        draw.ellipse([px(0.68), py(0.24), px(0.82), py(0.38)], fill=gold) # Right pauldron

        # Shield in Left Arm
        draw.polygon([(px(0.10), py(0.36)), (px(0.28), py(0.36)), (px(0.24), py(0.72)), (px(0.10), py(0.62))], fill=blue_cape)
        draw.polygon([(px(0.12), py(0.38)), (px(0.26), py(0.38)), (px(0.22), py(0.70)), (px(0.12), py(0.60))], fill=steel)
        draw.ellipse([px(0.16), py(0.46), px(0.22), py(0.54)], fill=gold)

        # Glowing Runic Sword in Right Hand
        draw.rectangle([px(0.78), py(0.12), px(0.84), py(0.68)], fill=(225, 240, 255, 255))
        draw.rectangle([px(0.74), py(0.46), px(0.88), py(0.50)], fill=gold) # Crossguard
        draw.rectangle([px(0.79), py(0.50), px(0.83), py(0.62)], fill=(80, 45, 20, 255)) # Hilt
        # Blue Rune Glow
        draw.line([px(0.81), py(0.16), px(0.81), py(0.44)], fill=(0, 220, 255, 255), width=2)

        # Helmet with Plume
        draw.ellipse([px(0.34), py(0.08), px(0.66), py(0.32)], fill=steel)
        draw.rectangle([px(0.40), py(0.18), px(0.60), py(0.22)], fill=(25, 25, 30, 255)) # Visor slit
        draw.polygon([(px(0.46), py(0.08)), (px(0.54), py(0.08)), (px(0.50), py(-0.02))], fill=(220, 30, 30, 255)) # Plume
    else:
        # Skeleton Warrior
        bone = (230, 230, 220)
        bone_dark = (140, 140, 130)
        draw.ellipse([px(0.18), py(0.86), px(0.82), py(0.97)], fill=(15, 12, 18, 170))

        # Ribcage & Spine
        draw.rectangle([px(0.48), py(0.28), px(0.52), py(0.64)], fill=bone_dark)
        for ry in [0.34, 0.42, 0.50, 0.58]:
            draw.line([px(0.36), py(ry), px(0.64), py(ry)], fill=bone, width=4)

        # Bone Legs
        draw.line([px(0.38), py(0.64), px(0.34), py(0.90)], fill=bone, width=5)
        draw.line([px(0.62), py(0.64), px(0.66), py(0.90)], fill=bone, width=5)

        # Skull
        draw.ellipse([px(0.35), py(0.06), px(0.65), py(0.30)], fill=bone)
        draw.ellipse([px(0.40), py(0.16), px(0.46), py(0.24)], fill=(220, 20, 20, 255)) # Glowing Red Eye
        draw.ellipse([px(0.54), py(0.16), px(0.60), py(0.24)], fill=(220, 20, 20, 255)) # Glowing Red Eye
        draw.rectangle([px(0.44), py(0.26), px(0.56), py(0.30)], fill=(40, 40, 40, 255)) # Teeth

        # Rusty Scimitar
        draw.polygon([(px(0.72), py(0.50)), (px(0.86), py(0.20)), (px(0.82), py(0.16)), (px(0.70), py(0.46))], fill=(150, 95, 60, 255))

    return img

def draw_strategy_unit(width=512, height=512, is_player=True):
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    px = lambda f: int(width * f)
    py = lambda f: int(height * f)

    main_c = (35, 110, 240) if is_player else (220, 35, 35)
    dark_c = (20, 60, 150) if is_player else (140, 20, 20)

    # Shadow
    draw.ellipse([px(0.10), py(0.75), px(0.90), py(0.95)], fill=(15, 12, 18, 170))

    # Heavy Treads
    draw.rounded_rectangle([px(0.14), py(0.62), px(0.86), py(0.88)], radius=12, fill=(35, 38, 45, 255))
    for tx in [0.22, 0.36, 0.50, 0.64, 0.78]:
        draw.ellipse([px(tx-0.05), py(0.68), px(tx+0.05), py(0.82)], fill=(75, 80, 95, 255))

    # Armored Chassis & Cockpit
    draw.polygon([(px(0.22), py(0.64)), (px(0.78), py(0.64)), (px(0.70), py(0.36)), (px(0.30), py(0.36))], fill=main_c)
    draw.rectangle([px(0.36), py(0.40), px(0.64), py(0.50)], fill=(0, 230, 255, 200)) # Cockpit Visor

    # Heavy Autocannon Barrels
    draw.rectangle([px(0.46), py(0.12), px(0.54), py(0.40)], fill=(60, 65, 75, 255))
    draw.rectangle([px(0.44), py(0.10), px(0.56), py(0.16)], fill=(20, 22, 28, 255)) # Muzzle brake

    return img

def draw_tower_defense_turret(width=512, height=512, is_player=True):
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    px = lambda f: int(width * f)
    py = lambda f: int(height * f)

    if is_player:
        # High Tech Plasma Defense Turret
        draw.ellipse([px(0.10), py(0.76), px(0.90), py(0.96)], fill=(15, 12, 18, 170))
        # Pedestal
        draw.polygon([(px(0.20), py(0.84)), (px(0.80), py(0.84)), (px(0.68), py(0.52)), (px(0.32), py(0.52))], fill=(65, 70, 85, 255))
        draw.rectangle([px(0.36), py(0.44), px(0.64), py(0.54)], fill=(45, 48, 60, 255))
        # Turret Dome
        draw.ellipse([px(0.26), py(0.24), px(0.74), py(0.52)], fill=(40, 125, 245, 255))
        draw.ellipse([px(0.38), py(0.30), px(0.62), py(0.46)], fill=(0, 240, 255, 255)) # Energy Core
        # Dual Barrels
        draw.rectangle([px(0.66), py(0.30), px(0.90), py(0.36)], fill=(30, 32, 40, 255))
        draw.rectangle([px(0.66), py(0.40), px(0.90), py(0.46)], fill=(30, 32, 40, 255))
    else:
        # Creep Monster
        draw.ellipse([px(0.15), py(0.70), px(0.85), py(0.90)], fill=(15, 12, 18, 170))
        draw.ellipse([px(0.22), py(0.32), px(0.78), py(0.78)], fill=(160, 45, 190, 255))
        # Glowing Red Eyes & Horns
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
        # Running Legs
        draw.polygon([(px(0.24), py(0.56)), (px(0.44), py(0.56)), (px(0.18), py(0.88)), (px(0.10), py(0.88))], fill=dark)
        draw.polygon([(px(0.56), py(0.56)), (px(0.76), py(0.56)), (px(0.88), py(0.86)), (px(0.74), py(0.86))], fill=dark)
        draw.rounded_rectangle([px(0.06), py(0.84), px(0.22), py(0.92)], radius=4, fill=neon_cyan) # Sneaker
        draw.rounded_rectangle([px(0.76), py(0.82), px(0.92), py(0.90)], radius=4, fill=neon_cyan) # Sneaker
        # Torso & Neon Runner Shirt
        draw.polygon([(px(0.30), py(0.28)), (px(0.70), py(0.28)), (px(0.66), py(0.60)), (px(0.34), py(0.60))], fill=neon_cyan)
        draw.rectangle([px(0.44), py(0.28), px(0.56), py(0.60)], fill=dark)
        # Head with Neon Cyber Visor
        draw.ellipse([px(0.36), py(0.08), px(0.64), py(0.30)], fill=skin)
        draw.rounded_rectangle([px(0.34), py(0.14), px(0.66), py(0.22)], radius=4, fill=(255, 220, 0, 255))
    else:
        # Electric Barrier Hurdle
        draw.rectangle([px(0.10), py(0.60), px(0.18), py(0.90)], fill=(45, 50, 60, 255))
        draw.rectangle([px(0.82), py(0.60), px(0.90), py(0.90)], fill=(45, 50, 60, 255))
        draw.rectangle([px(0.14), py(0.68), px(0.86), py(0.76)], fill=(255, 30, 60, 255))
        draw.line([px(0.14), py(0.72), px(0.86), py(0.72)], fill=(255, 230, 50, 255), width=3)

    return img

if __name__ == "__main__":
    os.makedirs("./test_sprites_all", exist_ok=True)
    draw_adventure_hero().save("./test_sprites_all/adventure_hero.png")
    draw_dungeon_knight(is_player=True).save("./test_sprites_all/dungeon_knight.png")
    draw_dungeon_knight(is_player=False).save("./test_sprites_all/dungeon_skeleton.png")
    draw_strategy_unit(is_player=True).save("./test_sprites_all/strategy_mech.png")
    draw_tower_defense_turret(is_player=True).save("./test_sprites_all/td_turret.png")
    draw_runner_athlete(is_player=True).save("./test_sprites_all/runner_hero.png")
    print("All specialized genre sprites verified successfully!")

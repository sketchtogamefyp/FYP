import os
import sys
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

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

    # Ground Shadow
    draw.ellipse([px(0.18), py(0.86), px(0.82), py(0.97)], fill=(15, 12, 18, 170))

    # Brown Work Boots
    # Left Boot
    draw.rounded_rectangle([px(0.52), py(0.82), px(0.78), py(0.94)], radius=8, fill=brown)
    draw.rectangle([px(0.52), py(0.90), px(0.78), py(0.94)], fill=(70, 35, 15, 255))
    # Right Boot (stepping forward)
    draw.rounded_rectangle([px(0.20), py(0.82), px(0.48), py(0.94)], radius=8, fill=brown)
    draw.rectangle([px(0.20), py(0.90), px(0.48), py(0.94)], fill=(70, 35, 15, 255))

    # Blue Overalls & Legs
    draw.polygon([(px(0.24), py(0.58)), (px(0.46), py(0.58)), (px(0.44), py(0.86)), (px(0.22), py(0.86))], fill=blue)
    draw.polygon([(px(0.54), py(0.58)), (px(0.76), py(0.58)), (px(0.78), py(0.86)), (px(0.56), py(0.86))], fill=blue)
    draw.rectangle([px(0.28), py(0.44), px(0.72), py(0.68)], fill=blue)
    draw.rectangle([px(0.28), py(0.58), px(0.72), py(0.68)], fill=blue_dark)

    # Overalls Straps & Yellow Brass Buttons
    draw.polygon([(px(0.32), py(0.32)), (px(0.40), py(0.32)), (px(0.40), py(0.52)), (px(0.32), py(0.52))], fill=blue)
    draw.polygon([(px(0.60), py(0.32)), (px(0.68), py(0.32)), (px(0.68), py(0.52)), (px(0.60), py(0.52))], fill=blue)
    draw.ellipse([px(0.33), py(0.44), px(0.39), py(0.50)], fill=yellow)
    draw.ellipse([px(0.61), py(0.44), px(0.67), py(0.50)], fill=yellow)

    # Red Shirt (Torso & Arms)
    draw.polygon([(px(0.34), py(0.30)), (px(0.66), py(0.30)), (px(0.66), py(0.46)), (px(0.34), py(0.46))], fill=red)
    # Left Arm (Raised victory / run)
    draw.polygon([(px(0.64), py(0.32)), (px(0.84), py(0.24)), (px(0.88), py(0.36)), (px(0.68), py(0.44))], fill=red)
    draw.rounded_rectangle([px(0.78), py(0.18), px(0.92), py(0.34)], radius=8, fill=white) # White glove fist
    draw.line([px(0.80), py(0.24), px(0.90), py(0.24)], fill=(200, 200, 210, 255), width=2)
    # Right Arm (Forward action)
    draw.polygon([(px(0.16), py(0.36)), (px(0.36), py(0.32)), (px(0.34), py(0.46)), (px(0.14), py(0.50))], fill=red)
    draw.rounded_rectangle([px(0.08), py(0.44), px(0.22), py(0.58)], radius=8, fill=white) # White glove
    draw.line([px(0.10), py(0.50), px(0.20), py(0.50)], fill=(200, 200, 210, 255), width=2)

    # Head & Face
    draw.ellipse([px(0.30), py(0.10), px(0.70), py(0.40)], fill=skin)
    draw.ellipse([px(0.26), py(0.18), px(0.36), py(0.30)], fill=skin) # Left ear
    draw.ellipse([px(0.64), py(0.18), px(0.74), py(0.30)], fill=skin) # Right ear

    # Large Round Nose
    draw.ellipse([px(0.42), py(0.20), px(0.58), py(0.32)], fill=skin)
    draw.arc([px(0.42), py(0.20), px(0.58), py(0.32)], 0, 180, fill=skin_shadow, width=2)

    # Big Expressive Eyes
    draw.ellipse([px(0.38), py(0.14), px(0.46), py(0.24)], fill=white)
    draw.ellipse([px(0.42), py(0.15), px(0.46), py(0.23)], fill=(20, 60, 180, 255))
    draw.ellipse([px(0.43), py(0.16), px(0.45), py(0.19)], fill=white)
    draw.ellipse([px(0.54), py(0.14), px(0.62), py(0.24)], fill=white)
    draw.ellipse([px(0.54), py(0.15), px(0.58), py(0.23)], fill=(20, 60, 180, 255))
    draw.ellipse([px(0.55), py(0.16), px(0.57), py(0.19)], fill=white)

    # Iconic Bushy Mustache
    draw.polygon([
        (px(0.34), py(0.30)), (px(0.42), py(0.26)), (px(0.50), py(0.28)),
        (px(0.58), py(0.26)), (px(0.66), py(0.30)), (px(0.62), py(0.36)),
        (px(0.50), py(0.34)), (px(0.38), py(0.36))
    ], fill=(45, 25, 15, 255))

    # Red Cap with Visor & Emblem
    draw.polygon([
        (px(0.26), py(0.16)), (px(0.30), py(0.04)), (px(0.50), py(-0.02)),
        (px(0.70), py(0.04)), (px(0.74), py(0.16)), (px(0.80), py(0.18)),
        (px(0.50), py(0.18)), (px(0.20), py(0.18))
    ], fill=red)
    draw.polygon([(px(0.20), py(0.16)), (px(0.80), py(0.16)), (px(0.76), py(0.22)), (px(0.24), py(0.22))], fill=red_dark) # Visor shadow

    # White Crest / "M" Circle
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

    # Shadow
    draw.ellipse([px(0.15), py(0.80), px(0.85), py(0.95)], fill=(15, 12, 18, 160))

    # Black Feet
    draw.ellipse([px(0.18), py(0.72), px(0.48), py(0.90)], fill=black)
    draw.ellipse([px(0.52), py(0.72), px(0.82), py(0.90)], fill=black)

    # Cream Stem / Body
    draw.polygon([(px(0.32), py(0.45)), (px(0.68), py(0.45)), (px(0.62), py(0.78)), (px(0.38), py(0.78))], fill=cream)

    # Brown Mushroom Cap Head
    draw.polygon([
        (px(0.20), py(0.50)), (px(0.14), py(0.36)), (px(0.22), py(0.15)),
        (px(0.50), py(0.06)), (px(0.78), py(0.15)), (px(0.86), py(0.36)),
        (px(0.80), py(0.50)), (px(0.50), py(0.54))
    ], fill=brown)
    draw.polygon([(px(0.18), py(0.48)), (px(0.82), py(0.48)), (px(0.74), py(0.54)), (px(0.26), py(0.54))], fill=brown_dark)

    # Angry Bushy Eyebrows
    draw.polygon([(px(0.30), py(0.22)), (px(0.48), py(0.28)), (px(0.48), py(0.32)), (px(0.30), py(0.26))], fill=black)
    draw.polygon([(px(0.70), py(0.22)), (px(0.52), py(0.28)), (px(0.52), py(0.32)), (px(0.70), py(0.26))], fill=black)

    # Big Eyes
    draw.ellipse([px(0.34), py(0.28), px(0.46), py(0.44)], fill=(255, 255, 255, 255))
    draw.ellipse([px(0.38), py(0.30), px(0.44), py(0.42)], fill=black)
    draw.ellipse([px(0.54), py(0.28), px(0.66), py(0.44)], fill=(255, 255, 255, 255))
    draw.ellipse([px(0.56), py(0.30), px(0.62), py(0.42)], fill=black)

    # Pointy Little Fangs
    draw.polygon([(px(0.36), py(0.60)), (px(0.42), py(0.52)), (px(0.44), py(0.60))], fill=(255, 255, 255, 255))
    draw.polygon([(px(0.56), py(0.60)), (px(0.58), py(0.52)), (px(0.64), py(0.60))], fill=(255, 255, 255, 255))

    return img

if __name__ == "__main__":
    os.makedirs("./test_sprites_preview", exist_ok=True)
    mario = draw_mario_hero()
    mario.save("./test_sprites_preview/mario_hero.png")
    goomba = draw_goomba_enemy()
    goomba.save("./test_sprites_preview/goomba_enemy.png")
    print("Saved Mario and Goomba sprites!")

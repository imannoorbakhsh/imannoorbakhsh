"""Convert assets/avatar.jpg into the two ASCII portraits the card draws.

The subject is cut out of the background with GrabCut, so only the person is
drawn and the sky/wall stay blank. Writes assets/ascii-art.txt (dark card:
light ink on dark) and assets/ascii-art-light.txt (light card: dark ink on
light). Cells are 0.54 as wide as tall, matching build_card.ART_CELL_ASPECT.

Needs opencv-python-headless, numpy and Pillow.

Usage:
    python tools/build_ascii.py
"""
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "assets" / "avatar.jpg"
RAMP = " .:-=+*#%@"          # light -> dense
CROP = (70, 75, 345, 400)    # left, top, right, bottom
COLS = 62
CELL = 0.54
FLOOR = 0.22                 # minimum ink on the subject so dark hair still shows


def cut_out(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    mask = np.full((h, w), cv2.GC_BGD, np.uint8)
    mask[70:455, 60:400] = cv2.GC_PR_FGD
    mask[250:330, 230:320] = cv2.GC_FGD      # beard
    mask[160:260, 90:250] = cv2.GC_FGD       # hair
    mask[340:h, 60:400] = cv2.GC_FGD         # jacket
    bgd, fgd = np.zeros((1, 65)), np.zeros((1, 65))
    cv2.grabCut(img, mask, None, bgd, fgd, 6, cv2.GC_INIT_WITH_MASK)
    m = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8) * 255
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))


def render(invert: bool) -> str:
    img = cv2.imread(str(SRC))
    mask = cut_out(img)
    l, t, r, b = CROP
    img, mask = img[t:b, l:r], mask[t:b, l:r]
    rows = round(COLS * CELL * img.shape[0] / img.shape[1])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(6, 6)).apply(gray)
    gray = cv2.resize(gray, (COLS, rows), interpolation=cv2.INTER_AREA) / 255.0
    gray = np.clip((gray - 0.5) * 1.5 + 0.5, 0, 1)
    mask = cv2.resize(mask, (COLS, rows), interpolation=cv2.INTER_AREA) / 255.0
    out = []
    for y in range(rows):
        line = ""
        for x in range(COLS):
            if mask[y, x] < 0.5:
                line += " "
                continue
            v = 1 - gray[y, x] if invert else gray[y, x]
            d = (0.0 if invert else FLOOR) + (1 - (0.0 if invert else FLOOR)) * v ** (1.6 if invert else 1)
            line += RAMP[min(len(RAMP) - 1, int(d * len(RAMP)))]
        out.append(line.rstrip())
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    (ROOT / "assets" / "ascii-art.txt").write_text(render(invert=False), encoding="utf-8")
    (ROOT / "assets" / "ascii-art-light.txt").write_text(render(invert=True), encoding="utf-8")
    print("wrote both portraits")

# app/compose.py
from io import BytesIO
from typing import Tuple
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .effects import comicify

# Colours (your brand)
FRAME = (247, 186, 54)   # yellow
DARK  = (50, 50, 50)     # bottom bar
WHITE = (240, 240, 240)
SE_L  = (247,186,54)     # S## colour
SE_R  = (240,240,240)    # E## colour

def _paste_cover(dst: Image.Image, src: Image.Image, box: Tuple[int,int,int,int]):
    """Paste src so it covers box fully, preserving aspect (like CSS background-size: cover)."""
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    sw, sh = src.size
    scale = max(bw / sw, bh / sh)
    new = src.resize((int(sw * scale), int(sh * scale)), Image.LANCZOS)
    nx, ny = new.size
    dst.paste(new, (x0 + (bw - nx)//2, y0 + (bh - ny)//2))

def build_thumbnail_png(
    screenshot_bytes: bytes,
    logo_bytes: bytes,
    title: str,
    season: int,
    episode: int,
    size: Tuple[int,int] = (1280, 720),
    font_title_path: str | None = None,   # can pass .ttf later
    font_se_path: str | None = None,
    **style_kwargs,
) -> bytes:
    W, H = size

    # --- Read screenshot -> comic -> convert to PIL
    arr = np.frombuffer(screenshot_bytes, np.uint8)
    im_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if im_bgr is None:
        raise ValueError("could not decode screenshot")
    comic_bgr = comicify(im_bgr, **style_kwargs)
    comic_rgb = cv2.cvtColor(comic_bgr, cv2.COLOR_BGR2RGB)
    comic = Image.fromarray(comic_rgb)

    # --- Base canvas
    im = Image.new("RGB", (W, H), (24,24,24))
    d  = ImageDraw.Draw(im)

    # Outer rounded frame
    d.rectangle((0, 0, W, H), fill=FRAME)

    # Image area (white rounded panel)
    image_padding = 12
    img_box = (image_padding, image_padding, W - image_padding, H - image_padding)
    d.rounded_rectangle(img_box, radius=22, fill=(255,255,255))

    # Rounded mask so the image also has rounded corners
    mask = Image.new("L", (W, H), 0)
    md = ImageDraw.Draw(mask)
    md.rounded_rectangle(img_box, radius=22, fill=255)
    tmp = Image.new("RGB", (W, H))
    _paste_cover(tmp, comic, img_box)
    im.paste(tmp, (0, 0), mask)

    # Bottom bar
    bar_h = 110
    bar_box = (180, H - bar_h, W, H)
    d.rectangle(bar_box)

    # Logo
    logo = Image.open(BytesIO(logo_bytes)).convert("RGBA")
    L = 110
    logo = logo.resize((L, L), Image.LANCZOS)
    im.paste(logo, (30, H - bar_h + (bar_h - L)//2), logo)

    # Fonts (fallback to default if you haven't dropped .ttf files yet)
    if font_title_path:
        title_font = ImageFont.truetype(font_title_path, 64)
    else:
        title_font = ImageFont.load_default()
    if font_se_path:
        se_font = ImageFont.truetype(font_se_path, 56)
    else:
        se_font = ImageFont.load_default()

    # Title (left side of bar)
    tx = bar_box[0] + 24
    ty = bar_box[1] + (bar_h - 64)//2
    d.text((tx, ty), title.upper(), font=title_font, fill=WHITE)

    # S/E (right aligned)
    left = f"S{int(season):02d} "
    right = f"E{int(episode):02d}"
    lx = d.textlength(left, font=se_font)
    rx = d.textlength(right, font=se_font)
    gap = 8
    start_x = bar_box[2] - 24 - (lx + gap + rx)
    y = bar_box[1] + (bar_h - 56)//2
    d.text((start_x, y), left,  font=se_font, fill=SE_L)
    d.text((start_x + lx + gap, y), right, font=se_font, fill=SE_R)

    # Return PNG bytes
    buf = BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return buf.getvalue()

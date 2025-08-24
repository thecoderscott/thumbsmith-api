import cv2
import numpy as np

from io import BytesIO
from typing import Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

from .effects import comicify
from .helpers import load_font

# Colours (your brand)
FRAME = (247, 186, 54)   # yellow
DARK  = (50, 50, 50)     # bottom bar
WHITE = (240, 240, 240)

# Fonts
ASSETS_DIR = Path(__file__).parent / "fonts"
TITLE_FONT = load_font(None, 80)
SE_FONT    = load_font(None, 80)

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
    season: int,
    episode: int,
    title: Optional[str] = None,
    game_logo_bytes: Optional[bytes] = None,
    size: Tuple[int,int] = (1280, 720),
    edge_color_hex: str = FRAME,
    **style_kwargs,
) -> bytes:
    W, H = size
    image_padding = 12
    inner_w = W - image_padding
    inner_h = H - image_padding

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
    d.rectangle((0, 0, W, H), fill=edge_color_hex)

    # Image area (white rounded panel)
    img_box = (image_padding, image_padding, inner_w, inner_h)
    d.rounded_rectangle(img_box, radius=24, fill=edge_color_hex)

    # Rounded mask so the image also has rounded corners
    mask = Image.new("L", (inner_w, inner_h), 0)
    md = ImageDraw.Draw(mask)
    md.rounded_rectangle(img_box, radius=24, fill=255)
    tmp = Image.new("RGB", (inner_w, inner_h))
    _paste_cover(tmp, comic, img_box)
    im.paste(tmp, (0, 0), mask)

    # Bottom bar
    bar_h = 180
    bar_box = (180, inner_h - bar_h, inner_w, inner_h)

    # Channel logo — always fit the bar height, keep aspect
    logo_rgba = Image.open(BytesIO(logo_bytes)).convert("RGBA")

    pad = 16
    L_h = bar_h - 2 * pad
    scale = L_h / logo_rgba.height
    L_w = max(1, int(logo_rgba.width * scale))
    logo_r = logo_rgba.resize((L_w, L_h), Image.LANCZOS)

    x = max(pad, 32)
    y = (inner_h - bar_h) + (bar_h - L_h) // 2
    im.paste(logo_r, (x, y), logo_r)

    # Title
    gx = image_padding + 24
    gy = image_padding + 24

    if game_logo_bytes is not None:
        arr = np.frombuffer(game_logo_bytes, np.uint8)
        raw = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise ValueError("could not decode game logo")
        rgba = cv2.cvtColor(raw, cv2.COLOR_BGRA2RGBA) if raw.shape[2] == 4 else cv2.cvtColor(raw, cv2.COLOR_BGR2RGBA)
        g_logo = Image.fromarray(rgba)

        target_h = 128
        scale = target_h / g_logo.height
        g_logo_w = max(1, int(g_logo.width * scale))
        g_logo_r = g_logo.resize((g_logo_w, target_h), Image.LANCZOS)

        # safety: if very short canvases, keep clear of the bar
        if gy + target_h > H - bar_h - image_padding:
            gy = max(image_padding, (H - bar_h - image_padding) - target_h)

        im.paste(g_logo_r, (gx, gy), g_logo_r)
    elif title:
        d.text((gx, gy), title.upper(), font=TITLE_FONT, fill=WHITE)

    # S/E (right aligned)
    left = f"S{int(season):02d} "
    right = f"E{int(episode):02d}"
    lx = d.textlength(left, font=SE_FONT)
    rx = d.textlength(right, font=SE_FONT)
    gap = 4
    start_x = bar_box[2] - 24 - (lx + gap + rx)
    y = bar_box[1] + (bar_h - 48) // 2

    # tiny shadow
    for dx, dy in ((2, 0), (0, 2), (4, 4)):
        d.text((start_x + dx, y + dy), left, font=SE_FONT, fill=(0, 0, 0))
        d.text((start_x + lx + gap + dx, y + dy), right, font=SE_FONT, fill=(0, 0, 0))

    d.text((start_x, y), left, font=SE_FONT, fill=edge_color_hex)
    d.text((start_x + lx + gap, y), right, font=SE_FONT, fill=WHITE)

    # Return PNG bytes
    buf = BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return buf.getvalue()

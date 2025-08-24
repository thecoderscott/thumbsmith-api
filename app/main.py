import numpy as np
import cv2

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
from typing import Optional

from .effects import comicify
from .compose import build_thumbnail_png
from .enums import StyleEnum, StrengthEnum

app = FastAPI(title="ThumbGen API", version="0.3.0")
ALLOWED = {"image/png", "image/jpeg"}

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/v1/generate", response_class=Response)
async def generate(
    screenshot: UploadFile = File(...),
    logo: UploadFile = File(...),
    season: int = Form(...),
    episode: int = Form(...),
    game_logo: Optional[UploadFile] = File(None),
    title: Optional[str] = Form(None),
    style: StyleEnum = Form(StyleEnum.game),
    strength: StrengthEnum = Form(StrengthEnum.normal),

    edge_color: str = Form("#000000"),
    edge_alpha: float = Form(1.0),
    overlay_color: str | None = Form(None),
    overlay_alpha: float = Form(0.0),
    width: int = Form(1280),
    height: int = Form(720),
    is_test: bool = Form(False)
):
    if screenshot.content_type not in ALLOWED:
        raise HTTPException(400, "screenshot must be PNG or JPEG")
    if logo.content_type not in ALLOWED:
        raise HTTPException(400, "logo must be PNG or JPEG")

    shot_bytes = await screenshot.read()
    logo_bytes = await logo.read()
    game_logo_bytes = await game_logo.read() if game_logo else None

    # Decode original
    arr = np.frombuffer(shot_bytes, np.uint8)
    im_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if im_bgr is None:
        raise HTTPException(400, "could not decode screenshot")

    if is_test:
        # --- Comicify and create side-by-side comparison
        comic_bgr = comicify(
            im_bgr,
            style=style,
            strength=strength,
            edge_color_hex=edge_color,
            edge_alpha=edge_alpha,
            overlay_hex=overlay_color,
            overlay_alpha=overlay_alpha,
        )

        # Resize both to the same height for neat comparison
        h = 400
        scale_orig = h / im_bgr.shape[0]
        w_orig = int(im_bgr.shape[1] * scale_orig)
        orig_resized = cv2.resize(im_bgr, (w_orig, h))

        scale_comic = h / comic_bgr.shape[0]
        w_comic = int(comic_bgr.shape[1] * scale_comic)
        comic_resized = cv2.resize(comic_bgr, (w_comic, h))

        side_by_side = np.hstack((orig_resized, comic_resized))

        ok, png = cv2.imencode(".png", side_by_side)
        if not ok:
            raise HTTPException(500, "encode failed")
        return Response(content=png.tobytes(), media_type="image/png")

    # --- Normal mode: return full thumbnail
    try:
        png_bytes = build_thumbnail_png(
            screenshot_bytes=shot_bytes,
            logo_bytes=logo_bytes,
            title=title,
            season=season,
            episode=episode,
            game_logo_bytes=game_logo_bytes,
            size=(max(640, min(3840, width)), max(360, min(2160, height))),
            style=style,
            strength=strength,
            edge_color_hex=edge_color,
            edge_alpha=edge_alpha,
            overlay_hex=overlay_color,
            overlay_alpha=overlay_alpha,
        )
    except Exception as e:
        raise HTTPException(400, f"processing error: {e}")

    return Response(content=png_bytes, media_type="image/png")

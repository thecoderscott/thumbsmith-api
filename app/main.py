import numpy as np
import cv2

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

from .compose import build_thumbnail_png
from .enums import StyleEnum, StrengthEnum

app = FastAPI(title="ThumbGen API", version="0.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://coderscott.dev",
        "https://api.coderscott.dev",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)
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
    game_logo: Optional[bytes] = File(None),
    title: Optional[str] = Form(default=None),
    style: StyleEnum = Form(StyleEnum.game),
    strength: StrengthEnum = Form(StrengthEnum.normal),

    edge_color: str = Form("#000000"),
    edge_alpha: float = Form(1.0),
    overlay_color: str | None = Form(None),
    overlay_alpha: float = Form(0.1),
    width: int = Form(1280),
    height: int = Form(720),
):
    if screenshot.content_type not in ALLOWED:
        raise HTTPException(400, "screenshot must be PNG or JPEG")
    if logo.content_type not in ALLOWED:
        raise HTTPException(400, "logo must be PNG or JPEG")

    shot_bytes = await screenshot.read()
    logo_bytes = await logo.read()
    game_logo_bytes = game_logo if game_logo else None

    # Sanitise inputs
    edge_color = edge_color or None
    edge_alpha = edge_alpha or None
    overlay_color = overlay_color or None
    overlay_alpha = overlay_alpha or None
    width = width or 1280
    height = height or 720

    # Decode original
    arr = np.frombuffer(shot_bytes, np.uint8)
    im_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if im_bgr is None:
        raise HTTPException(400, "could not decode screenshot")

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

from pathlib import Path
from functools import lru_cache
from PIL import ImageFont

ASSETS_DIR = Path(__file__).resolve().parents[1] / "fonts"
DEFAULT_TITLE_FONT = ASSETS_DIR / "ChakraPetch-Regular.ttf"
DEFAULT_SE_FONT    = ASSETS_DIR / "ChakraPetch-SemiBold.ttf"

@lru_cache(maxsize=64)
def _truetype_cached(path_str: str, size: int) -> ImageFont.FreeTypeFont:
    """Hashable + cacheable wrapper around ImageFont.truetype."""
    return ImageFont.truetype(path_str, size)

def load_font(path: str | Path | None, size: int) -> ImageFont.FreeTypeFont:
    """
    Public helper. Normalizes inputs and uses a cached truetype loader.
    Falls back to bundled defaults, then Pillow's bitmap font.
    """
    # try explicit path first
    if path:
        try:
            p = Path(path)
            if p.exists():
                return _truetype_cached(str(p), size)
        except Exception:
            pass

    # fallback to bundled defaults (pick title vs SE based on size)
    try:
        default = DEFAULT_SE_FONT if size <= 80 else DEFAULT_TITLE_FONT
        return _truetype_cached(str(default), size)
    except Exception:
        return ImageFont.load_default()

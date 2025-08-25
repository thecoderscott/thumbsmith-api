import cv2, numpy as np

from .enums import StyleEnum, StrengthEnum

def _clamp01(x: float) -> float:
    try: x = float(x)
    except Exception: x = 0.0
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x

def _hex_to_bgr(hexstr: str) -> tuple[int, int, int]:
    s = hexstr.strip().lstrip("#")
    if len(s) == 3:
        s = "".join(ch*2 for ch in s)
    if len(s) != 6:
        raise ValueError("overlay/edge color must be #RGB or #RRGGBB")
    r, g, b = int(s[0:2],16), int(s[2:4],16), int(s[4:6],16)
    return b, g, r

def _gamma(bgr: np.ndarray, g: float) -> np.ndarray:
    if g == 1.0: return bgr
    lut = np.array([((i/255.0)**(1.0/g))*255 for i in range(256)], np.uint8)
    return cv2.LUT(bgr, lut)

def comicify(
    im_bgr: np.ndarray,
    style: StyleEnum = StyleEnum.game,
    strength: StrengthEnum = StrengthEnum.normal,
    edge_color_hex: str = "#000000",
    edge_alpha: float = 1.0,
    overlay_hex: str | None = None,
    overlay_alpha: float = 0.0,
):
    edge_alpha = _clamp01(edge_alpha)
    overlay_alpha = _clamp01(overlay_alpha)

    if style == StyleEnum.photo:
        return im_bgr.copy()

    """
    Cel-shaded comic look with configurable edge colour/opacity and optional overlay.
    style: "game" | "photo" | "avatar"
    strength: 1..3
    """
    if style == StyleEnum.game:
        params = {
            StrengthEnum.weak: dict(k_colors=14, band_levels=8, edge_th=(80, 160), min_area=20, thicken=1,
                                    sat_gain=1.06, gamma=1.03),
            StrengthEnum.normal: dict(k_colors=10, band_levels=6, edge_th=(60, 120), min_area=30, thicken=2,
                                      sat_gain=1.12, gamma=1.06),
            StrengthEnum.strong: dict(k_colors=7, band_levels=4, edge_th=(45, 90), min_area=50, thicken=3,
                                      sat_gain=1.18, gamma=1.10),
        }[strength]
    elif style == StyleEnum.avatar:
        params = dict(k_colors=8, band_levels=5, edge_th=(80, 160), min_area=10, thicken=2, sat_gain=1.15, gamma=1.05)
    else:
        # Unknown style? Do nothing.
        return im_bgr.copy()

    k_colors = params["k_colors"]
    band_levels = params["band_levels"]
    edge_th = params["edge_th"]
    min_area = params["min_area"]
    thicken = params["thicken"]
    sat_gain = params["sat_gain"]
    gamma = params["gamma"]

    # ---- 1) smooth + LAB chroma quantize + L banding (cel shade) ----
    sm = cv2.bilateralFilter(im_bgr, d=9, sigmaColor=90, sigmaSpace=90)
    lab = cv2.cvtColor(sm, cv2.COLOR_BGR2LAB)
    L  = lab[:,:,0]
    ab = lab[:,:,1:3].reshape(-1,2).astype(np.float32)
    K  = max(2, k_colors)
    criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(ab, K, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    abq = centers[labels.flatten()].reshape(lab[:,:,1:3].shape).astype(np.uint8)

    N = band_levels
    step = max(1, 256//N)
    Lband = ((L // step) * step).astype(np.uint8)

    lab_q = np.dstack([Lband, abq])
    flat_bgr = cv2.cvtColor(lab_q, cv2.COLOR_LAB2BGR)

    # ---- 2) edges on original L, cleaned ----
    edges = cv2.Canny(L, *edge_th)
    k3 = np.ones((3,3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, k3, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k3, iterations=1)
    n, lab_cc, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
    clean = np.zeros_like(edges)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[lab_cc==i] = 255
    if thicken>0:
        clean = cv2.dilate(clean, np.ones((2,2), np.uint8), iterations=thicken)

    # ---- 3) pop (sat + gamma) ----
    hsv = cv2.cvtColor(flat_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[...,1] = np.clip(hsv[...,1]*sat_gain, 0, 255)
    flat_bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    out = _gamma(flat_bgr, gamma)

    # ---- 4) draw edges with chosen colour/opacity ----
    mask = clean > 0
    if edge_alpha <= 0:
        edge_alpha = 0.0
    if edge_alpha >= 1:
        out[mask] = 0 if edge_color_hex.lower() == "#000000" else np.array(_hex_to_bgr(edge_color_hex), np.uint8)
    else:
        edge_col = np.array(_hex_to_bgr(edge_color_hex), dtype=np.float32)
        src = out.astype(np.float32)
        src[mask] = (1.0 - edge_alpha) * src[mask] + edge_alpha * edge_col
        out = np.clip(src, 0, 255).astype(np.uint8)

    # ---- 5) optional overlay tint ----
    if overlay_hex is not None and overlay_alpha is not None:
        try:
            a = float(overlay_alpha)
        except Exception:
            a = 0.0
        a = max(0.0, min(1.0, a))
        if a > 0.0:
            ov = np.full_like(out, _hex_to_bgr(overlay_hex), dtype=np.uint8)
            # cross‑fade: (1-a)*image + a*overlay
            out = cv2.addWeighted(out, 1.0 - a, ov, a, 0.0)

    return out

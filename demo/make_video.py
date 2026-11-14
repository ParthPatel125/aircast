"""Render the AirCast demo film as a .mov, frames and music generated in code.

Concept: the whole story lives in the sky. Content sits among drifting clouds,
and scene changes are weather: a cloud bank wipes the screen, fog dissolves,
lightning opens the storm scene, rain clears into a warm sunset.

Usage:
  python demo/make_video.py --draft     # 640x360 10fps, quick look, no audio
  python demo/make_video.py             # 1920x1080 24fps + music -> AirCast.mov
  python demo/make_video.py --stills 1 8 20 ...   # dump PNG stills at seconds
"""
import argparse
import json
import math
import os
import subprocess
import sys
import wave

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------- timeline --
FINAL_W, FINAL_H, FINAL_FPS = 1920, 1080, 24
DUR = 110.0
SCENES = {  # name: (start, end) seconds; 2s gaps are weather transitions
    "s1": (0.0, 13.0),    # night smoke, the 134 day
    "s2": (15.0, 26.0),   # dawn, the decision
    "s3": (28.0, 46.0),   # day, how it works (map)
    "s4": (48.0, 64.0),   # terminal demo
    "s5": (66.0, 84.0),   # results
    "s6": (86.0, 99.0),   # storm, where it fails
    "s7": (101.0, 110.0), # sunset close
}
TRANS = [  # (start, end, kind, from, to)
    (13.0, 15.0, "cloudwipe", "s1", "s2"),
    (26.0, 28.0, "fog", "s2", "s3"),
    (46.0, 48.0, "cloudwipe", "s3", "s4"),
    (64.0, 66.0, "fog", "s4", "s5"),
    (84.0, 86.0, "lightning", "s5", "s6"),
    (99.0, 101.0, "sunbreak", "s6", "s7"),
]

# ------------------------------------------------------------------- utils --
def ease(t):  # smoothstep
    t = min(1.0, max(0.0, t))
    return t * t * (3 - 2 * t)

def ease_out(t):
    t = min(1.0, max(0.0, t))
    return 1 - (1 - t) ** 3

def lerp(a, b, t):
    return a + (b - a) * t

def mixc(c1, c2, t):
    return tuple(int(lerp(a, b, t)) for a, b in zip(c1, c2))

def fade_in(t, t0, d=0.7):
    return ease((t - t0) / d)

# ------------------------------------------------------------------ assets --
class Assets:
    def __init__(self, W, H):
        self.W, self.H = W, H
        s = H / 1080.0
        self.s = s
        F = "/System/Library/Fonts/"
        def fnt(file, idx, px):
            return ImageFont.truetype(F + file, max(10, int(px * s)), index=idx)
        self.f_mega  = fnt("Avenir Next.ttc", 8, 200)   # Heavy
        self.f_h1    = fnt("Avenir Next.ttc", 8, 86)
        self.f_h2    = fnt("Avenir Next.ttc", 2, 56)    # Demi Bold
        self.f_body  = fnt("Avenir Next.ttc", 5, 36)    # Medium
        self.f_small = fnt("Avenir Next.ttc", 5, 27)
        self.f_tag   = fnt("Avenir Next.ttc", 2, 26)
        self.f_mono  = fnt("Menlo.ttc", 0, 27)
        self.f_monob = fnt("Menlo.ttc", 1, 27)
        rng = np.random.default_rng(7)
        self.tiles = [self._noise_tile(rng, int(W * 2.4), int(H * 1.25), o)
                      for o in (3, 4, 5)]
        self.glow = self._radial(int(600 * s))

    def _noise_tile(self, rng, w, h, base_oct):
        acc = np.zeros((h, w), np.float32)
        amp, tot = 1.0, 0.0
        for o in range(5):
            gw, gh = 2 ** (base_oct + o), 2 ** (base_oct + o - 1)
            g = rng.random((gh, gw)).astype(np.float32)
            layer = np.asarray(Image.fromarray((g * 255).astype(np.uint8), "L")
                               .resize((w, h), Image.BICUBIC), np.float32) / 255.0
            acc += amp * layer
            tot += amp
            amp *= 0.55
        acc /= tot
        acc = (acc - acc.min()) / (acc.max() - acc.min() + 1e-6)
        return acc

    def _radial(self, size):
        y, x = np.ogrid[:size, :size]
        c = size / 2
        d = np.sqrt((x - c) ** 2 + (y - c) ** 2) / c
        a = np.clip(1 - d, 0, 1) ** 2
        return a.astype(np.float32)

# ------------------------------------------------------------- sky + cloud --
SKY = {  # name: (top rgb, bottom rgb, cloud tint, cloud coverage 0..1, cloud alpha)
    "s1": ((16, 8, 14),   (52, 18, 20),   (120, 58, 48),  0.62, 0.75),
    "s2": ((38, 48, 82),  (166, 122, 100),(226, 198, 182),0.50, 0.65),
    "s3": ((40, 78, 128), (148, 192, 228),(245, 248, 252),0.42, 0.75),
    "s4": ((34, 64, 106), (120, 162, 200),(235, 240, 248),0.45, 0.70),
    "s5": ((46, 92, 150), (160, 204, 236),(248, 250, 253),0.38, 0.70),
    "s6": ((14, 18, 28),  (42, 50, 64),   (88, 98, 116),  0.72, 0.85),
    "s7": ((40, 30, 62),  (198, 108, 72), (250, 190, 150),0.48, 0.70),
}

def sky_image(A, name, t):
    W, H = A.W, A.H
    top, bot, tint, cov, calpha = SKY[name]
    col = np.linspace(0, 1, H, dtype=np.float32)[:, None]
    img = np.empty((H, W, 3), np.float32)
    for i in range(3):
        img[:, :, i] = top[i] + (bot[i] - top[i]) * col
    # three parallax cloud layers
    for li, (tile, speed, soft) in enumerate(zip(A.tiles, (6, 14, 26), (0.30, 0.22, 0.16))):
        th, tw = tile.shape
        off = int((t * speed * A.s) % (tw - W))
        win = tile[(th - H) // 2:(th - H) // 2 + H, off:off + W]
        a = np.clip((win - (1 - cov)) / soft, 0, 1) ** 1.5 * calpha * (0.45 + 0.27 * li)
        for i in range(3):
            img[:, :, i] = img[:, :, i] * (1 - a) + tint[i] * a
    return img

def paste_glow(img, A, cx, cy, radius, color, strength):
    H, W, _ = img.shape
    g = A.glow
    size = int(radius * 2)
    if size < 8:
        return
    g = np.asarray(Image.fromarray((g * 255).astype(np.uint8), "L")
                   .resize((size, size), Image.BILINEAR), np.float32) / 255.0
    x0, y0 = int(cx - size / 2), int(cy - size / 2)
    x1, y1 = x0 + size, y0 + size
    gx0, gy0 = max(0, -x0), max(0, -y0)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1), min(H, y1)
    if x1 <= x0 or y1 <= y0:
        return
    a = g[gy0:gy0 + y1 - y0, gx0:gx0 + x1 - x0] * strength
    for i in range(3):
        img[y0:y1, x0:x1, i] = img[y0:y1, x0:x1, i] * (1 - a) + color[i] * a

# ----------------------------------------------------------- draw helpers ---
INK = (240, 243, 248)
DIM = (196, 205, 218)
ACCENT = (108, 176, 255)
AMBER = (242, 180, 60)
RED = (244, 108, 100)
GREEN = (118, 212, 160)
GRAY = (165, 173, 186)

def shadow_text(d, xy, text, font, fill, anchor="la", alpha=255, shadow=200):
    if alpha <= 0:
        return
    x, y = xy
    off = max(1, int(font.size * 0.07))
    sh = (8, 10, 16, int(shadow * alpha / 255))
    d.text((x + off, y + off), text, font=font, fill=sh, anchor=anchor)
    d.text((x, y + off), text, font=font, fill=sh, anchor=anchor)
    d.text((x, y), text, font=font, fill=fill + (int(alpha),), anchor=anchor)

def rrect(d, box, r, fill, outline=None, width=2):
    d.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=width)

def panel(d, box, r, alpha=200):
    rrect(d, box, r, fill=(10, 14, 22, alpha), outline=(255, 255, 255, 36), width=2)

def reveal_lines(d, A, lines, x, y0, gap, t, t0, stagger=0.45):
    for i, (text, font, fill) in enumerate(lines):
        a = fade_in(t, t0 + i * stagger)
        dy = (1 - ease_out((t - (t0 + i * stagger)) / 0.8)) * 22 * A.s
        shadow_text(d, (x, y0 + dy), text, font, fill, alpha=int(255 * a))
        y0 += gap

# ------------------------------------------------------------------ scenes --
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "video_data.json")) as fh:
    DATA = json.load(fh)

def chart_panel(A, pw, ph, series, ymax, frac, who=15.0, xlabels=(), ylabels=()):
    """Render a chart into an RGBA panel; each series drawn up to frac."""
    S = 2  # supersample
    img = Image.new("RGBA", (pw * S, ph * S), (10, 14, 22, 205))
    d = ImageDraw.Draw(img)
    L, R, T, B = int(pw * S * 0.10), int(pw * S * 0.965), int(ph * S * 0.10), int(ph * S * 0.84)
    f_ax = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", int(A.f_small.size * S * 0.62), index=0)
    for yv in ylabels:
        yy = B - (B - T) * yv / ymax
        d.line([(L, yy), (R, yy)], fill=(255, 255, 255, 26), width=S)
        d.text((L - 8 * S, yy), str(int(yv)), font=f_ax, fill=(170, 178, 192, 220), anchor="rm")
    yy = B - (B - T) * who / ymax
    for xseg in range(L, R, 7 * S):
        d.line([(xseg, yy), (min(R, xseg + 4 * S), yy)], fill=AMBER + (210,), width=S)
    d.text((R, yy - 6 * S), "WHO 15", font=f_ax, fill=AMBER + (235,), anchor="rd")
    lx = L
    for frac_i, (vals, color, wpx, name) in enumerate(series):
        n = len(vals)
        k = max(2, int(n * min(1.0, frac - frac_i * 0.12)))
        pts = [(L + (R - L) * i / (n - 1), B - (B - T) * min(vals[i], ymax) / ymax)
               for i in range(k)]
        if len(pts) >= 2:
            d.line(pts, fill=color + (255,), width=int(wpx * S), joint="curve")
        # legend row inside the panel, top left
        d.rounded_rectangle([lx, T - 6 * S, lx + 14 * S, T + 4 * S], radius=3 * S,
                            fill=color + (255,))
        d.text((lx + 20 * S, T - 1 * S), name, font=f_ax,
               fill=(225, 230, 240, 255), anchor="lm")
        lx += 20 * S + d.textlength(name, font=f_ax) + 26 * S
    for xi, lab in xlabels:
        d.text((L + (R - L) * xi, B + 6 * S), lab, font=f_ax,
               fill=(170, 178, 192, 220), anchor="ma")
    rrect(d, (S, S, pw * S - S, ph * S - S), 14 * S, fill=None,
          outline=(255, 255, 255, 36), width=2 * S)
    return img.resize((pw, ph), Image.LANCZOS)

def scene_s1(A, img, t):
    H, W = A.H, A.W
    paste_glow(img, A, W * 0.72, H * 0.30, 560 * A.s, (190, 70, 50), 0.34 * ease(t / 4))
    paste_glow(img, A, W * 0.25, H * 0.85, 480 * A.s, (210, 140, 40), 0.20 * ease(t / 5))
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    x = int(W * 0.08)
    shadow_text(d, (x, H * 0.16), "JUNE 27, 2023   CHICAGO", A.f_tag, AMBER,
                alpha=int(255 * fade_in(t, 0.4)))
    n = int(9 + (134 - 9) * ease_out((t - 1.0) / 5.0)) if t > 1 else 9
    a_num = fade_in(t, 0.8)
    shadow_text(d, (x - 6 * A.s, H * 0.22), str(n), A.f_mega, RED, alpha=int(255 * a_num))
    numw = d.textlength(str(n), font=A.f_mega)
    shadow_text(d, (x + numw + 18 * A.s, H * 0.40), "µg/m³ of fine-particle pollution",
                A.f_body, DIM, alpha=int(255 * a_num))
    reveal_lines(d, A, [
        ("Overnight, Chicago had some of the worst air in the world.", A.f_h2, INK),
        ("The day before, the monitors read 9. Nobody saw it coming.", A.f_body, DIM),
    ], x, H * 0.64, 86 * A.s, t, 5.6, 1.6)
    return ov

def scene_s2(A, img, t):
    H, W = A.H, A.W
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    x = int(W * 0.08)
    shadow_text(d, (x, H * 0.14), "THE DECISION", A.f_tag, ACCENT, alpha=int(255 * fade_in(t, 0.3)))
    reveal_lines(d, A, [
        ("Should tomorrow's practice", A.f_h1, INK),
        ("be outside?", A.f_h1, INK),
    ], x, H * 0.20, 110 * A.s, t, 0.6, 0.35)
    # nine cloud puffs, the ninth burning
    r = 30 * A.s
    y = H * 0.56
    for i in range(9):
        a = fade_in(t, 2.2 + i * 0.22, 0.4)
        cx = x + r + i * (r * 2.65)
        col = RED if i == 8 else (225, 230, 240)
        for dx, dy, rr in ((-0.7, 0.15, 0.72), (0.0, -0.18, 1.0), (0.75, 0.2, 0.66)):
            d.ellipse([cx + dx * r - rr * r, y + dy * r - rr * r,
                       cx + dx * r + rr * r, y + dy * r + rr * r],
                      fill=col + (int(235 * a),))
    reveal_lines(d, A, [
        ("A school athletics director makes this call every day.", A.f_body, INK),
        ("In Chicago, 1 day in 9 is over the WHO air quality guideline.", A.f_body, DIM),
        ("Apps say what the air is like now. The call is about tomorrow.", A.f_body, DIM),
    ], x, H * 0.68, 60 * A.s, t, 4.6, 0.9)
    return ov

def scene_s3(A, img, t):
    H, W = A.H, A.W
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    x = int(W * 0.07)
    shadow_text(d, (x, H * 0.10), "MEET AIRCAST", A.f_tag, ACCENT, alpha=int(255 * fade_in(t, 0.3)))
    reveal_lines(d, A, [
        ("It predicts tomorrow's PM2.5", A.f_h2, INK),
        ("as the EPA monitors will measure it.", A.f_h2, INK),
    ], x, H * 0.155, 72 * A.s, t, 0.6, 0.4)
    # map panel, cities as cloud puffs
    pw, ph = int(W * 0.44), int(H * 0.52)
    px, py = int(W * 0.07), int(H * 0.38)
    a_map = fade_in(t, 2.0)
    if a_map > 0:
        pnl = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
        pd = ImageDraw.Draw(pnl)
        panel(pd, (0, 0, pw - 1, ph - 1), 18 * A.s, alpha=150)
        cities = {"Thunder Bay": (0.66, 0.12), "Minneapolis": (0.16, 0.50),
                  "Green Bay": (0.80, 0.55), "Madison": (0.63, 0.72)}
        chi = (0.86, 0.86)
        fx = lambda u: u * (pw * 0.9) + pw * 0.05
        fy = lambda v: v * (ph * 0.82) + ph * 0.06
        for name, (u, v) in cities.items():
            x1, y1, x2, y2 = fx(u), fy(v), fx(chi[0]), fy(chi[1])
            for seg in range(14):
                s0, s1 = seg / 14, (seg + 0.55) / 14
                pd.line([(lerp(x1, x2, s0), lerp(y1, y2, s0)),
                         (lerp(x1, x2, s1), lerp(y1, y2, s1))],
                        fill=(255, 255, 255, 56), width=max(1, int(2 * A.s)))
            # drifting smoke puff
            k = ((t * 0.16) + hash(name) % 97 / 97) % 1.0
            pd.ellipse([lerp(x1, x2, k) - 7 * A.s, lerp(y1, y2, k) - 7 * A.s,
                        lerp(x1, x2, k) + 7 * A.s, lerp(y1, y2, k) + 7 * A.s],
                       fill=AMBER + (190,))
            pd.ellipse([x1 - 9 * A.s, y1 - 9 * A.s, x1 + 9 * A.s, y1 + 9 * A.s],
                       fill=(225, 230, 240, 235))
            pd.text((x1, y1 - 14 * A.s), name, font=A.f_small,
                    fill=(222, 228, 238, 255), anchor="mb")
        pd.ellipse([fx(chi[0]) - 12 * A.s, fy(chi[1]) - 12 * A.s,
                    fx(chi[0]) + 12 * A.s, fy(chi[1]) + 12 * A.s], fill=ACCENT + (255,))
        pd.text((fx(chi[0]) - 18 * A.s, fy(chi[1])), "CHICAGO", font=A.f_tag,
                fill=INK + (255,), anchor="rm")
        pnl.putalpha(pnl.getchannel("A").point(lambda v: int(v * a_map)))
        ov.alpha_composite(pnl, (px, py))
    chips = [
        ("25 public signals: monitors, weather, season", 3.2, INK),
        ("Plus air quality right now in four cities upwind", 4.2, INK),
        ("Minneapolis air today is Chicago air tomorrow", 5.2, ACCENT),
        ("Ridge regression, tested only on the future", 6.4, INK),
    ]
    cy = H * 0.42
    for text, t0, color in chips:
        a = fade_in(t, t0)
        if a > 0:
            cw = d.textlength(text, font=A.f_small) + 40 * A.s
            box = (W * 0.54, cy, min(W * 0.965, W * 0.54 + cw), cy + 58 * A.s)
            rrect(d, box, 13 * A.s, fill=(10, 14, 22, int(190 * a)),
                  outline=(255, 255, 255, int(40 * a)), width=2)
            shadow_text(d, (W * 0.54 + 20 * A.s, cy + 13 * A.s), text, A.f_small,
                        color, alpha=int(255 * a))
        cy += 84 * A.s
    return ov

TERM_LINES = [
    ("AirCast. Next-day PM2.5 forecast for Chicago", DIM, 2.8),
    ("------------------------------------------------", DIM, 3.1),
    ("Model trained on 1,194 days (2022-08-06 to 2025-11-11)", DIM, 3.5),
    ("Today (2026-11-14) so far: 3.3 µg/m³ (24 hours of data)", DIM, 4.3),
    ("Upwind now: Minneapolis 13, Green Bay 2, Madison 3, Thunder Bay 1", DIM, 5.3),
    ("", DIM, 5.5),
    ("Forecast for tomorrow (2026-11-15): 8.4 µg/m³  (likely range 3 to 12)", ACCENT, 6.6),
    ("Alert level: ALL CLEAR. Outdoor activity looks fine.", GREEN, 8.0),
]

def scene_s4(A, img, t):
    H, W = A.H, A.W
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    x = int(W * 0.08)
    shadow_text(d, (x, H * 0.10), "LIVE DEMO   REAL INPUT, REAL RESULT", A.f_tag,
                ACCENT, alpha=int(255 * fade_in(t, 0.3)))
    pw, ph = int(W * 0.84), int(H * 0.60)
    px, py = int(W * 0.08), int(H * 0.17)
    a_p = fade_in(t, 0.5)
    panel(d, (px, py, px + pw, py + ph), 18 * A.s, alpha=int(215 * a_p))
    for i in range(3):
        d.ellipse([px + (22 + i * 30) * A.s, py + 18 * A.s,
                   px + (38 + i * 30) * A.s, py + 34 * A.s],
                  fill=(60, 70, 88, int(255 * a_p)))
    cmd = "python src/predict_tomorrow.py"
    nchars = int(max(0, (t - 1.0)) / 0.055)
    ty = py + 64 * A.s
    shadow_text(d, (px + 30 * A.s, ty), "$ " + cmd[:nchars] +
                ("▌" if (nchars < len(cmd) and int(t * 2) % 2 == 0) else ""),
                A.f_monob, INK, alpha=int(255 * a_p), shadow=0)
    ty += 56 * A.s
    for text, color, t0 in TERM_LINES:
        if t >= t0:
            shadow_text(d, (px + 30 * A.s, ty), text, A.f_mono, color, shadow=0)
        ty += 44 * A.s
    shadow_text(d, (x, py + ph + 34 * A.s),
                "Live conditions for Chicago and the four upwind cities. No API key. A few seconds.",
                A.f_small, DIM, alpha=int(255 * fade_in(t, 9.0)))
    return ov

def scene_s5(A, img, t):
    H, W = A.H, A.W
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    x = int(W * 0.07)
    shadow_text(d, (x, H * 0.09), "DOES IT BEAT THE SIMPLE GUESS?", A.f_tag, ACCENT,
                alpha=int(255 * fade_in(t, 0.3)))
    shadow_text(d, (x, H * 0.135), "315 days of 2025 the model never saw", A.f_h2, INK,
                alpha=int(255 * fade_in(t, 0.5)))
    # bars
    bx, by, bw = x, H * 0.70, 120 * A.s
    bh_max = H * 0.30
    for i, (val, label, color, t0) in enumerate([
            (3.46, "persistence\ntomorrow = today", GRAY, 1.2),
            (3.11, "AirCast\nerror, µg/m³", ACCENT, 1.8)]):
        g = ease_out((t - t0) / 1.1)
        h = bh_max * (val / 3.8) * g
        if g > 0:
            d.rounded_rectangle([bx, by - h, bx + bw, by], radius=8 * A.s,
                                fill=color + (235,))
            shadow_text(d, (bx + bw / 2, by - h - 14 * A.s), f"{val:.2f}", A.f_h2,
                        INK, anchor="ms")
            shadow_text(d, (bx + bw / 2, by + 16 * A.s), label, A.f_small, INK, anchor="ma")
        bx += bw + 110 * A.s
    # stat chips
    stats = [("10%", "better at 1 day", 3.2), ("25%", "better at 3 days", 3.7),
             ("88%", "bad days caught", 4.2), ("77%", "warnings correct", 4.7)]
    cy = H * 0.84
    cx = x
    for v, k, t0 in stats:
        a = fade_in(t, t0)
        if a > 0:
            box = (cx, cy, cx + 205 * A.s, cy + 92 * A.s)
            rrect(d, box, 14 * A.s, fill=(10, 14, 22, int(190 * a)),
                  outline=(255, 255, 255, int(40 * a)), width=2)
            shadow_text(d, (cx + 18 * A.s, cy + 6 * A.s), v, A.f_h2, ACCENT, alpha=int(255 * a))
            shadow_text(d, (cx + 18 * A.s, cy + 56 * A.s), k, A.f_small, DIM, alpha=int(255 * a))
        cx += 225 * A.s
    # 2025 chart
    pw, ph = int(W * 0.46), int(H * 0.56)
    frac = ease((t - 2.2) / 4.5)
    if frac > 0:
        pnl = chart_panel(A, pw, ph, [
            (DATA["test"]["actual"], INK, 2, "actual"),
            (DATA["test"]["forecast"], ACCENT, 3, "forecast"),
        ], 70, frac, xlabels=[(0.02, "Jan"), (0.35, "May"), (0.67, "Aug"), (0.95, "Nov")],
            ylabels=(35, 70))
        ov.alpha_composite(pnl, (int(W * 0.50), int(H * 0.22)))
        shadow_text(d, (W * 0.50, H * 0.22 + ph + 14 * A.s),
                    "2025, held out. Trained on 2022 to 2024. Never shuffled.",
                    A.f_small, INK)
    return ov

def scene_s6(A, img, t):
    H, W = A.H, A.W
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    x = int(W * 0.07)
    shadow_text(d, (x, H * 0.09), "WHERE IT FAILS. WE SAY SO.", A.f_tag, RED,
                alpha=int(255 * fade_in(t, 0.3)))
    shadow_text(d, (x, H * 0.135), "June 2023, replayed with a model trained only on the past",
                A.f_h2, INK, alpha=int(255 * fade_in(t, 0.5)))
    pw, ph = int(W * 0.52), int(H * 0.58)
    frac = ease((t - 1.2) / 4.0)
    if frac > 0:
        pnl = chart_panel(A, pw, ph, [
            (DATA["case"]["actual"], INK, 3, "actual"),
            (DATA["case"]["persistence"], GRAY, 2, "baseline"),
            (DATA["case"]["forecast"], ACCENT, 3, "AirCast"),
        ], 140, frac, xlabels=[(0.02, "Jun 18"), (0.45, "Jun 27"), (0.97, "Jul 8")],
            ylabels=(70, 140))
        ov.alpha_composite(pnl, (x, int(H * 0.24)))
    a_c = fade_in(t, 6.0)
    if a_c > 0:
        box = (x, H * 0.845, x + pw * 0.56, H * 0.845 + 118 * A.s)
        rrect(d, box, 14 * A.s, fill=(60, 16, 18, int(225 * a_c)),
              outline=RED + (int(220 * a_c),), width=3)
        shadow_text(d, (box[0] + 20 * A.s, box[1] + 10 * A.s),
                    "Reality: 134.  Our forecast: 17.", A.f_body, INK, alpha=int(255 * a_c))
        shadow_text(d, (box[0] + 20 * A.s, box[1] + 62 * A.s),
                    "Right decision, wrong magnitude.", A.f_small, DIM, alpha=int(255 * a_c))
    reveal_lines(d, A, [
        ("The first smoke day is invisible", A.f_body, INK),
        ("from the ground. Nothing local", A.f_body, INK),
        ("saw it coming, ours or anyone's.", A.f_body, INK),
        ("Extreme events need satellites.", A.f_body, AMBER),
        ("We measured that gap", A.f_body, DIM),
        ("instead of hiding it.", A.f_body, DIM),
    ], int(W * 0.66), H * 0.28, 56 * A.s, t, 3.2, 0.75)
    return ov

def scene_s7(A, img, t):
    H, W = A.H, A.W
    paste_glow(img, A, W * 0.5, H * 0.95, 900 * A.s, (250, 160, 90), 0.30)
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    a1 = fade_in(t, 0.5)
    shadow_text(d, (W / 2, H * 0.30), "AirCast", A.f_mega, INK, anchor="mm",
                alpha=int(255 * a1))
    _centered_reveal(d, A, [
        ("Honest forecasts for the days that can be forecast,", A.f_body, DIM),
        ("and honesty about the days that can't.", A.f_body, DIM),
    ], W / 2, H * 0.52, 54 * A.s, t, 1.8, 0.8)
    a2 = fade_in(t, 4.0)
    if a2 > 0:
        box_w = 760 * A.s
        rrect(d, (W / 2 - box_w / 2, H * 0.70, W / 2 + box_w / 2, H * 0.70 + 70 * A.s),
              999, fill=(10, 14, 22, int(190 * a2)),
              outline=(255, 255, 255, int(45 * a2)), width=2)
        shadow_text(d, (W / 2, H * 0.70 + 16 * A.s), "github.com/ParthPatel125/aircast",
                    A.f_mono, DIM, anchor="ma", alpha=int(255 * a2))
    shadow_text(d, (W / 2, H * 0.84), "Parth Patel and Anay Goyal   Emerson High School   AI + Climate",
                A.f_small, DIM, anchor="ma", alpha=int(255 * fade_in(t, 5.0)))
    return ov

SCENE_FN = {"s1": scene_s1, "s2": scene_s2, "s3": scene_s3, "s4": scene_s4,
            "s5": scene_s5, "s6": scene_s6, "s7": scene_s7}

# fix s7 reveal_lines to center: wrap draw with centered anchor
def _centered_reveal(d, A, lines, cx, y0, gap, t, t0, stagger):
    for i, (text, font, fill) in enumerate(lines):
        a = fade_in(t, t0 + i * stagger)
        shadow_text(d, (cx, y0), text, font, fill, anchor="ma", alpha=int(255 * a))
        y0 += gap

def render_scene(A, name, t_abs):
    t0, t1 = SCENES[name]
    t = max(0.0, t_abs - t0)
    img = sky_image(A, name, t_abs)
    if name == "s6":
        img = add_rain(A, img, t_abs)
    ov = SCENE_FN[name](A, img, t)
    base = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB").convert("RGBA")
    base.alpha_composite(ov)
    return np.asarray(base.convert("RGB"), np.float32)

def add_rain(A, img, t_abs, strength=1.0):
    H, W, _ = img.shape
    rng = np.random.default_rng(int(t_abs * A_FPS[0]) % 100000)
    n = int(260 * strength * (W / 1920))
    ov = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(ov)
    xs = rng.random(n) * W
    ys = rng.random(n) * H
    ln = (14 + rng.random(n) * 18) * A.s
    for x, y, l in zip(xs, ys, ln):
        d.line([(x, y), (x - l * 0.25, y + l)], fill=120, width=max(1, int(A.s * 2)))
    a = np.asarray(ov, np.float32) / 255.0 * 0.6 * strength
    for i in range(3):
        img[:, :, i] = img[:, :, i] * (1 - a) + 205 * a
    return img

A_FPS = [24]  # mutable fps holder for rain seed

# -------------------------------------------------------------- transitions --
def frame_at(A, t):
    for name, (t0, t1) in SCENES.items():
        if t0 <= t < t1:
            return render_scene(A, name, t)
    for (t0, t1, kind, sa, sb) in TRANS:
        if t0 <= t < t1:
            k = (t - t0) / (t1 - t0)
            fa = render_scene(A, sa, min(t, SCENES[sa][1] - 0.01))
            fb = render_scene(A, sb, SCENES[sb][0])
            return transition(A, fa, fb, kind, k, t)
    return render_scene(A, "s7", min(t, DUR - 0.01))

def transition(A, fa, fb, kind, k, t_abs):
    H, W, _ = fa.shape
    if kind == "cloudwipe":
        xc = (k * 1.6 - 0.3) * W
        band = A.tiles[2]
        th, tw = band.shape
        win = band[(th - H) // 2:(th - H) // 2 + H, :W]
        cols = np.arange(W)[None, :]
        edge = np.clip((cols - xc) / (0.22 * W), -1, 1)
        base_mix = np.clip(0.5 - edge * 0.5 + (win - 0.5) * 0.9, 0, 1) ** 2
        m = base_mix[:, :, None]
        out = fa * (1 - m) + fb * m
        dense = np.clip((win - 0.25) / 0.3, 0, 1) * np.clip(1 - np.abs(edge) * 1.6, 0, 1)
        dm = dense[:, :, None] * 0.9
        cloud_col = np.array([225, 228, 236], np.float32)
        return out * (1 - dm) + cloud_col * dm
    if kind == "fog":
        noise = A.tiles[1]
        th, tw = noise.shape
        win = noise[(th - H) // 2:(th - H) // 2 + H, :W]
        thr = 1 - k * 1.2
        m = np.clip((win - thr) / 0.18, 0, 1)[:, :, None]
        fogw = np.sin(k * math.pi) ** 2
        out = fa * (1 - m) + fb * m
        fog_col = np.array([210, 214, 224], np.float32)
        return out * (1 - 0.45 * fogw) + fog_col * 0.45 * fogw
    if kind == "lightning":
        out = fa * (1 - k) + fb * k
        flash = 0.0
        for fs, fd in ((0.18, 0.10), (0.42, 0.16)):
            if fs <= k < fs + fd:
                flash = 1 - (k - fs) / fd
        if flash > 0:
            out = out * (1 - 0.85 * flash) + 255 * 0.85 * flash
        return out
    if kind == "sunbreak":
        out = fa * (1 - k) + fb * k
        paste_glow(out, A, W * 0.5, H * 0.5, (600 + 900 * k) * A.s,
                   (255, 220, 170), 0.5 * math.sin(k * math.pi))
        return out
    return fa * (1 - k) + fb * k

# -------------------------------------------------------------------- audio --
SR = 44100

def synth_audio(path):
    import scipy.signal as sig
    n = int(DUR * SR)
    L = np.zeros(n, np.float32)
    Rr = np.zeros(n, np.float32)
    tt = np.arange(n) / SR

    def note(f):
        return f

    def add_pad(t0, dur, freqs, amp=0.10, attack=1.2, release=1.8, bright=0.4):
        i0, i1 = int(t0 * SR), min(n, int((t0 + dur) * SR))
        seg_t = tt[:i1 - i0]
        env = np.minimum(1, seg_t / attack) * np.minimum(1, (seg_t[-1] - seg_t) / release)
        w = np.zeros(i1 - i0, np.float32)
        for f in freqs:
            for h, ha in ((1, 1.0), (2, 0.5 * bright), (3, 0.25 * bright), (4, 0.12 * bright)):
                ph = np.random.default_rng(int(f * h)).random() * 6.28
                w += ha * np.sin(2 * np.pi * f * h * seg_t + ph).astype(np.float32)
            w += 0.3 * np.sin(2 * np.pi * (f * 1.003) * seg_t).astype(np.float32)
        w = w / (len(freqs) * 2.2) * env * amp
        L[i0:i1] += w; Rr[i0:i1] += w * 0.92

    def add_pluck(t0, f, amp=0.10, dur=1.6, pan=0.0):
        i0, i1 = int(t0 * SR), min(n, int((t0 + dur) * SR))
        seg_t = tt[:i1 - i0]
        env = np.exp(-seg_t * 3.2)
        w = (np.sin(2 * np.pi * f * seg_t) + 0.35 * np.sin(2 * np.pi * 2 * f * seg_t)) * env * amp
        L[i0:i1] += w * (1 - max(0, pan)); Rr[i0:i1] += w * (1 + min(0, pan))

    def add_kick(t0, amp=0.20):
        i0 = int(t0 * SR); i1 = min(n, i0 + int(0.22 * SR))
        seg_t = tt[:i1 - i0]
        f = 92 * np.exp(-seg_t * 16) + 42
        w = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-seg_t * 11) * amp
        L[i0:i1] += w; Rr[i0:i1] += w

    def add_noise(t0, dur, amp, lo, hi, env_shape="swell", seed=0):
        i0, i1 = int(t0 * SR), min(n, int((t0 + dur) * SR))
        m = i1 - i0
        rng = np.random.default_rng(seed)
        w = rng.standard_normal(m).astype(np.float32)
        b, a = sig.butter(2, [lo / (SR / 2), hi / (SR / 2)], "bandpass")
        w = sig.lfilter(b, a, w).astype(np.float32)
        seg_t = np.linspace(0, 1, m)
        env = np.sin(seg_t * np.pi) ** 2 if env_shape == "swell" else \
              np.minimum(1, seg_t * 8) * np.minimum(1, (1 - seg_t) * 4)
        w *= env * amp
        L[i0:i1] += w; Rr[i0:i1] += np.roll(w, 220)

    # chords (Hz): Am(110,165,220,262) F(87,131,175,220) C(131,165,196,262) G(98,147,196,247)
    Am = [110, 165, 220, 262]; F = [87.3, 130.8, 174.6, 220]
    C = [130.8, 164.8, 196, 261.6]; G = [98, 146.8, 196, 246.9]
    Dm = [73.4, 110, 146.8, 175]
    # S1 dark
    add_pad(0.0, 14.5, [55, 110, 164.8], amp=0.14, attack=3.5, bright=0.15)
    add_noise(0.2, 13, 0.05, 80, 400, seed=1)
    # S2 dawn
    add_pad(15.0, 6.0, F, amp=0.11, bright=0.3)
    add_pad(21.0, 6.5, C, amp=0.11, bright=0.35)
    # S3 + S4 gentle loop
    tpos = 28.0
    for i, ch in enumerate([Am, F, C, G, Am, F, C, G]):
        add_pad(tpos, 5.0, ch, amp=0.10, bright=0.4)
        tpos += 4.5
    for i, (t0, f) in enumerate([(29.5, 440), (33.9, 523.3), (38.5, 659.3), (43.0, 392),
                                 (47.5, 440), (52.0, 523.3), (56.5, 440), (61.0, 659.3)]):
        add_pluck(t0, f, amp=0.055, pan=(-0.4 if i % 2 else 0.4))
    # S5 pulse
    tpos = 66.0
    for ch in [C, G, Am, F]:
        add_pad(tpos, 5.0, ch, amp=0.12, bright=0.55)
        tpos += 4.5
    b = 66.0
    while b < 84.0:
        add_kick(b, 0.16)
        add_noise(b + 0.335, 0.05, 0.03, 5000, 11000, "swell", seed=int(b * 10))
        b += 0.667
    # S6 storm
    add_pad(85.5, 14.5, Dm, amp=0.13, attack=2.0, bright=0.2)
    add_pad(85.5, 14.5, [36.7], amp=0.10, attack=1.0, bright=0.1)
    add_noise(85.0, 14.5, 0.10, 900, 9000, "gate", seed=5)      # rain
    add_noise(84.2, 2.6, 0.22, 30, 160, "swell", seed=6)        # thunder
    add_noise(92.0, 2.2, 0.12, 30, 180, "swell", seed=7)
    # S7 resolve
    add_pad(100.5, 4.0, F, amp=0.11, bright=0.4)
    add_pad(104.0, 6.0, C, amp=0.11, attack=1.5, release=4.0, bright=0.45)
    add_pluck(102.0, 523.3, 0.05); add_pluck(104.5, 659.3, 0.05); add_pluck(106.5, 783.99, 0.045)
    # transition whooshes
    for (t0, t1, kind, _, _b) in TRANS:
        add_noise(t0 - 0.3, (t1 - t0) + 0.8, 0.07, 300, 2600, seed=int(t0))
    # master: gentle fade in/out, normalize
    master_env = np.minimum(1, tt / 1.5) * np.minimum(1, np.maximum(0, (DUR - tt) / 3.0))
    L *= master_env; Rr *= master_env
    peak = max(np.abs(L).max(), np.abs(Rr).max(), 1e-6)
    L = L / peak * 0.84; Rr = Rr / peak * 0.84
    data = np.empty(n * 2, np.float32)
    data[0::2] = L; data[1::2] = Rr
    pcm = (np.clip(data, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(SR)
        wf.writeframes(pcm.tobytes())
    print(f"audio: {path} ({DUR:.0f}s)")

# ------------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", action="store_true")
    ap.add_argument("--stills", nargs="*", type=float)
    ap.add_argument("--out", default=os.path.join(ROOT, "demo", "AirCast.mov"))
    args = ap.parse_args()

    if args.draft:
        W, H, FPS = 640, 360, 10
    else:
        W, H, FPS = FINAL_W, FINAL_H, FINAL_FPS
    A_FPS[0] = FPS
    A = Assets(W, H)

    if args.stills:
        outdir = os.path.join(ROOT, "demo", "stills")
        os.makedirs(outdir, exist_ok=True)
        for t in args.stills:
            arr = frame_at(A, t)
            Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(
                os.path.join(outdir, f"t{t:05.1f}.png"))
            print("still", t)
        return

    audio_path = os.path.join(ROOT, "demo", "score.wav")
    if not args.draft:
        synth_audio(audio_path)

    nframes = int(DUR * FPS)
    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-"]
    if not args.draft:
        cmd += ["-i", audio_path]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p"]
    if not args.draft:
        cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest"]
    cmd += [args.out if not args.draft else os.path.join(ROOT, "demo", "draft.mov")]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stderr=subprocess.DEVNULL)
    for i in range(nframes):
        t = i / FPS
        arr = frame_at(A, t)
        # end fade to black
        if t > DUR - 1.5:
            arr = arr * max(0.0, (DUR - t) / 1.5)
        proc.stdin.write(np.clip(arr, 0, 255).astype(np.uint8).tobytes())
        if i % (FPS * 10) == 0:
            print(f"  frame {i}/{nframes} (t={t:.0f}s)", flush=True)
    proc.stdin.close()
    proc.wait()
    print("wrote", cmd[-1])

if __name__ == "__main__":
    main()

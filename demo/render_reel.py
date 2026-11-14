"""Render demo/reel.html into AirCast.mov via Playwright.

  python demo/render_reel.py --beats      # one still per beat -> contact sheets
  python demo/render_reel.py              # full 60fps render, 3 subframes, motion blur, audio
"""
import argparse
import base64
import json
import os
import subprocess
import sys

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO = os.path.join(ROOT, "demo")
BPM, FPS, SUB = 120, 60, 3
DUR = 144 * 60 / BPM  # 72.0 s


def inject_data():
    with open(os.path.join(DEMO, "video_data.json")) as f:
        d = json.load(f)
    html = open(os.path.join(DEMO, "reel.html")).read()
    html = html.replace("__TEST__", json.dumps(d["test"]))
    html = html.replace("__CASE__", json.dumps(d["case"]))
    out = os.path.join(DEMO, "_reel_built.html")
    open(out, "w").write(html)
    return out


def open_page(p):
    b = p.chromium.launch(args=["--force-device-scale-factor=1",
                                "--disable-lcd-text", "--hide-scrollbars"])
    pg = b.new_page(viewport={"width": 1920, "height": 1080})
    pg.goto("file://" + inject_data())
    pg.wait_for_load_state("networkidle")
    pg.evaluate("document.fonts.ready")
    return b, pg


def beats():
    from playwright.sync_api import sync_playwright
    outdir = os.path.join(DEMO, "beatsheet")
    os.makedirs(outdir, exist_ok=True)
    with sync_playwright() as p:
        b, pg = open_page(p)
        frames = []
        for beat in range(0, 144, 1):
            t = beat * 60 / BPM + 0.24  # just after the beat lands
            pg.evaluate(f"seek({t})")
            path = os.path.join(outdir, f"b{beat:03d}.png")
            pg.screenshot(path=path)
            frames.append(path)
        b.close()
    # contact sheets: 16 per sheet (4x4), downscaled
    for s in range(0, 144, 16):
        sheet = Image.new("RGB", (4 * 480, 4 * 270), "#222")
        for i, f in enumerate(frames[s:s + 16]):
            im = Image.open(f).resize((480, 270))
            sheet.paste(im, ((i % 4) * 480, (i // 4) * 270))
        sheet.save(os.path.join(outdir, f"sheet_{s:03d}.png"))
    print("beat sheets written to demo/beatsheet/")


def full():
    from playwright.sync_api import sync_playwright
    audio_path = os.path.join(DEMO, "reel_score.wav")
    out_path = os.path.join(DEMO, "AirCast.mov")
    nframes = int(DUR * FPS)
    with sync_playwright() as p:
        b, pg = open_page(p)
        print("rendering soundtrack...")
        wav_b64 = pg.evaluate("renderAudio()")
        open(audio_path, "wb").write(base64.b64decode(wav_b64))
        cmd = ["ffmpeg", "-y",
               "-f", "image2pipe", "-framerate", str(FPS * SUB), "-c:v", "mjpeg", "-i", "-",
               "-i", audio_path,
               "-filter_complex",
               f"[0:v]tmix=frames={SUB},select='not(mod(n\\,{SUB}))',setpts=N/{FPS}/TB[v]",
               "-map", "[v]", "-map", "1:a",
               "-r", str(FPS), "-c:v", "libx264", "-preset", "medium", "-crf", "17",
               "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest",
               out_path]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        shutter = 1 / (FPS * SUB)  # 180 deg-ish shutter via consecutive subframes
        for i in range(nframes):
            for k in range(SUB):
                t = i / FPS + k * shutter
                pg.evaluate(f"seek({t})")
                shot = pg.screenshot(type="jpeg", quality=92)
                proc.stdin.write(shot)
            if i % (FPS * 5) == 0:
                print(f"  frame {i}/{nframes} (t={i / FPS:.0f}s)", flush=True)
        proc.stdin.close()
        proc.wait()
        b.close()
    print("wrote", out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--beats", action="store_true")
    a = ap.parse_args()
    beats() if a.beats else full()

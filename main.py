# ============================================================
# VAULTS OF HISTORY - AI Video Generator v3
# Two-call GPT: Story Beats → Render Decisions
# OpenCV + Pillow renderer with fade in/out animation
# Proper timing via timestamp_hint + fuzzy anchor matching
# ============================================================

import subprocess
import os
import json
import random
import requests
import traceback
import glob
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Vaults of History v3")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

current_job = {"status": "idle", "progress": 0, "output": None, "error": None, "started_at": None}

OUTPUT_WIDTH  = 1920
OUTPUT_HEIGHT = 1080
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

if not OPENAI_API_KEY:
    print("⚠  WARNING: OPENAI_API_KEY not set.")

MUSIC_MAP = {
    "space":    "bg_musics/space_ambient.mp3",
    "death":    "bg_musics/dark_ambient.mp3",
    "ancient":  "bg_musics/ancient_ambient.mp3",
    "religion": "bg_musics/sacred_ambient.mp3",
    "human":    "bg_musics/human_ambient.mp3",
    "default":  "bg_musics/vaults_ambient.mp3",
}

FONTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")

FONT_BLACK_CANDIDATES = [
    os.path.join(FONTS_DIR, "Anton-Regular.ttf"),
    os.path.join(FONTS_DIR, "Montserrat-Black.ttf"),
    os.path.join(FONTS_DIR, "Montserrat-ExtraBold.ttf"),
    "/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]
FONT_BOLD_CANDIDATES = [
    os.path.join(FONTS_DIR, "Montserrat-ExtraBold.ttf"),
    os.path.join(FONTS_DIR, "Montserrat-Bold.ttf"),
    "/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]
FONT_REGULAR_CANDIDATES = [
    os.path.join(FONTS_DIR, "Montserrat-Bold.ttf"),
    "/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]

def find_font(candidates):
    for path in candidates:
        if os.path.exists(path) and os.path.getsize(path) > 1000:
            print(f"  ✓ Font: {path}")
            return path
    return None

FONT_BLACK   = find_font(FONT_BLACK_CANDIDATES)
FONT_BOLD    = find_font(FONT_BOLD_CANDIDATES)
FONT_REGULAR = find_font(FONT_REGULAR_CANDIDATES)

def get_primary_font_path(bold: bool = True) -> str:
    """Return best available font: Black > ExtraBold > Bold > anything."""
    if FONT_BLACK:   return FONT_BLACK
    if FONT_BOLD:    return FONT_BOLD
    if FONT_REGULAR: return FONT_REGULAR
    return None

@app.on_event("startup")
async def startup_event():
    print("🚀 Vaults of History v3 starting...")
    # Audit broll folders so missing clips are immediately visible
    broll_dirs = ['space_vids','ancient_ruins_vids','cosmic_vids',
                  'dark_sky_vids','temple_vids']
    print("📁 Broll folder audit:")
    for d in broll_dirs:
        if os.path.exists(d):
            files = [f for f in os.listdir(d) if f.lower().endswith(('.mp4','.mov','.avi'))]
            status = f"✅ {len(files)} clips" if files else "❌ EMPTY -- add Seedance clips here"
        else:
            status = "❌ MISSING -- folder doesn't exist"
        print(f"  {d}: {status}")


# ============================================================
# GPT CALL 1 -- STORY BEAT ANALYZER
# Sends Whisper segments (with timestamps) to GPT so it can
# produce accurate timing without word-level lookup
# ============================================================
def analyze_story_beats(transcript_text: str, whisper_segments: list,
                        topic_hint: str, total_duration: float) -> dict:
    if not OPENAI_API_KEY:
        raise Exception("OPENAI_API_KEY not set.")

    print(f"  🎭 Call 1: Story beats ({len(transcript_text)} chars, {total_duration:.1f}s)...")
    client = OpenAI(api_key=OPENAI_API_KEY)

    # Build timed transcript — each Whisper segment on its own line with [start - end]
    timed_lines = []
    for seg in whisper_segments:
        s = float(seg.get('start', 0))
        e = float(seg.get('end', 0))
        t = seg.get('text', '').strip()
        if t:
            timed_lines.append(f"[{s:.2f}s - {e:.2f}s] {t}")
    timed_transcript = "\n".join(timed_lines)

    system_prompt = f"""You are the story producer for VAULTS OF HISTORY -- a viral mind-bending facts channel.
Style: @sackfeels on TikTok. Dramatic. Eerie. Cinematic.
Total audio duration: {total_duration:.1f} seconds.

You will receive a transcript with EXACT timestamps from Whisper speech recognition.
Each line is formatted as: [start - end] spoken words

YOUR JOB: Segment the transcript into story beats for cinematic video editing.

RULES:
- Use the Whisper timestamps directly -- they are accurate. Copy start_time and end_time from the brackets.
- Beat text MUST be copied VERBATIM from the transcript. Exact words, exact spelling. No paraphrasing.
- Keep beats 2-10 words -- natural spoken phrases or short clauses.
- A single Whisper segment can become 1-3 beats if it contains multiple natural phrases.
- Cover the ENTIRE transcript -- every word must appear in some beat.
- "pause" beats only for clear silence gaps (>0.5s) between segments.

beat_type: "hook"|"buildup"|"reveal"|"shock"|"pause"|"resolution"|"outro"

Return ONLY valid JSON:
{{
  "topic": "space|ancient|religion|human|death|default",
  "music_mood": "eerie|dark|mysterious|sacred|cosmic|haunting",
  "beats": [
    {{
      "beat_type": "hook|buildup|reveal|shock|pause|resolution|outro",
      "text": "verbatim words from transcript",
      "start_time": 0.0,
      "end_time": 2.5,
      "intensity": 8,
      "broll_category": "space|ancient|cosmic|sky|temple|any",
      "clip_duration": 4.0
    }}
  ]
}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Topic hint: {topic_hint}\n\nTimed transcript:\n{timed_transcript}\n\nSegment every line into beats. Use the timestamps shown. Copy text verbatim."}
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
            timeout=90,
        )
        result = json.loads(response.choices[0].message.content)
        beats  = result.get('beats', [])
        print(f"  ✅ {len(beats)} beats, topic={result.get('topic')}")
        return result
    except Exception as e:
        print(f"  ❌ Story beats failed: {e}")
        raise


# ============================================================
# GPT CALL 2 -- RENDER DECISION GENERATOR
# ============================================================
def generate_render_decisions(beats: list, topic: str) -> list:
    if not OPENAI_API_KEY:
        raise Exception("OPENAI_API_KEY not set.")

    print(f"  🎨 Call 2: Scene compositions for {len(beats)} beats...")
    client = OpenAI(api_key=OPENAI_API_KEY)

    system_prompt = f"""You are an elite short-form video editor. You compose every frame like a motion designer -- choosing position, size, color, animation, and timing for each visual element. You are not picking from preset templates. You are designing each scene.

Channel: VAULTS OF HISTORY -- mind-bending facts about space, ancient civilizations, and human consciousness. Audience is impossible to impress. The aesthetic is CINEMATIC and DARK -- like a documentary trailer, not a social media post.

=== FONT BEHAVIOR ===
The renderer uses Anton (ultra-condensed, cinematic) as the primary font. This font is TALL and NARROW. Design accordingly:
- Single impact words: size 180-280px, dead center or dramatically off-center
- Sentence words: size 90-140px per word, spread across the canvas
- Never stack more than 3-4 words at the same size in a column -- vary sizes
- Words should feel MASSIVE and COMMANDING, not informational

=== YOUR RENDERING ENGINE ===
Python OpenCV + Pillow on a {OUTPUT_WIDTH}x{OUTPUT_HEIGHT} canvas.

For each beat, you output a SCENE -- a list of ELEMENTS placed and animated however you want.

=== ELEMENT TYPES ===

TEXT element:
{{
  "type": "text",
  "content": "WORD",              // the text string
  "x": 0.5,                       // 0.0-1.0 horizontal position (anchor)
  "y": 0.4,                       // 0.0-1.0 vertical position (anchor)
  "anchor": "center",             // "center" | "left" | "right" -- how x,y aligns the text
  "size": 120,                    // pixels
  "color": "#FFFFFF",             // hex
  "weight": "black",              // "regular" | "bold" | "black" (use black for impact)
  "outline": 4,                   // pixels of black outline (3-6 typical)
  "anim": "fade_in",              // see ANIMATIONS below
  "start_offset": 0.0,            // seconds after beat starts when element appears
  "duration": null,               // seconds visible (null = until next beat)
  "anim_duration": 0.15,          // how long entrance animation takes
  "effect": "none"                // see EFFECTS below
}}

LINE element (for fractions, dividers, underlines):
{{
  "type": "line",
  "x1": 0.3, "y1": 0.5, "x2": 0.7, "y2": 0.5,
  "thickness": 8,
  "color": "#FFFFFF",
  "anim": "draw_horizontal",      // "draw_horizontal" draws left-to-right, "fade_in" fades, "none" appears instantly
  "start_offset": 0.2,
  "duration": null,
  "anim_duration": 0.3
}}

RECT element (for boxes, backgrounds, highlight bars):
{{
  "type": "rect",
  "x": 0.4, "y": 0.5, "w": 0.2, "h": 0.1,   // x,y is top-left corner. w,h are width/height
  "color": "#FBC02D",
  "filled": true,                   // true=filled, false=outline only
  "thickness": 4,                   // only used if filled=false
  "anim": "fade_in",
  "start_offset": 0.0,
  "duration": null
}}

CIRCLE element:
{{
  "type": "circle",
  "x": 0.5, "y": 0.5, "radius": 0.05,
  "color": "#FFFFFF",
  "filled": false,
  "thickness": 4,
  "anim": "fade_in",
  "start_offset": 0.0,
  "duration": null
}}

=== ANIMATIONS ===
- "none": appears instantly
- "fade_in": opacity 0→100% over anim_duration
- "slide_in_left": slides in from off-screen left
- "slide_in_right": slides in from off-screen right
- "slide_in_top": slides in from off-screen top
- "slide_in_bottom": slides in from off-screen bottom
- "scale_in": starts at 1.3x scale and snaps to 1.0x (punch effect)
- "snap": appears instantly with a 1-frame white flash
- "draw_horizontal": (lines only) draws progressively left-to-right

=== EFFECTS (applied during display, not just entrance) ===
- "none": static
- "flicker": rapid on/off blinking for first 0.3s (for shock words)
- "shake": position jitters slightly (for impact)
- "glow": adds soft colored glow halo around element

=== HOW TO COMPOSE SCENES ===

For a SPOKEN SENTENCE (3+ words): create one TEXT element per word. Words build across the frame -- horizontal flow, diagonal cascade, or scattered cluster. Keep words in the CENTER ZONE (x: 0.2-0.8, y: 0.2-0.8). Don't push words to extreme corners unless it's a dramatic 1-word-per-corner composition with 4+ words.

For a SHORT BEAT (1-2 words): compose as a CLUSTER, not a scatter. Both words should be near each other -- stacked vertically or side by side in the center region. NEVER put 2 words at opposite extreme corners (x:0.1 and x:0.8) with empty space between them. That looks broken.

For a SINGLE IMPACT WORD: one giant TEXT element, centered or slightly off-center. scale_in animation. Size 180-260px.

For a NUMBER OR FRACTION (like "1 in 1 billion"): compose it visually -- TEXT for "1", LINE for the divider, TEXT for "1 BILLION" below. Make it look like a math equation.

For COMPARISONS or BEFORE/AFTER: stack two TEXT elements vertically, use different colors, maybe a LINE separator.

For EMPHASIS: combine a RECT highlight bar behind a TEXT word. Or a CIRCLE around a key word.

VARY the composition across beats -- no two beats should look identical. But each beat must feel INTENTIONAL and complete, not like words floating randomly in space.

=== HARD RULES ===
1. Output exactly {len(beats)} scenes, one per beat, in order.
2. Every "content" in TEXT elements must use words VERBATIM from the beat text. Don't paraphrase. Don't add words not in the beat.
3. For pause beats: output {{"elements": []}} (empty scene).
4. start_offset values must fit within the beat duration.
5. x, y values are 0.0-1.0. NEVER use percentages or pixels.
6. For beats with 1-2 words: keep all elements within x: 0.2-0.75, y: 0.25-0.75. No extreme-corner scatters.
7. Never repeat the same word twice in one scene's elements -- each word appears exactly once.

=== COLOR DISCIPLINE ===
White (#FFFFFF) is your dominant color. Yellow (#FBC02D) for ONE key word per scene maximum. Other colors (red, blue, purple) only for very specific moments.

Return ONLY valid JSON:
{{
  "scenes": [
    {{
      "beat_index": <int>,
      "beat_type": "<hook|shock|reveal|buildup|pause|resolution|outro>",
      "elements": [
        // list of element objects as specified above
      ]
    }}
    // ... exactly {len(beats)} scenes
  ]
}}"""

    # Batch to stay safely under GPT-4o's 16384 output token limit.
    # 6 beats per batch ~ 5-7k tokens output, well within limit even with long scenes.
    BATCH_SIZE = 6
    all_scenes = []
    batches = [beats[i:i+BATCH_SIZE] for i in range(0, len(beats), BATCH_SIZE)]

    for batch_idx, batch in enumerate(batches):
        start_beat = batch_idx * BATCH_SIZE
        print(f"  🎨 Batch {batch_idx+1}/{len(batches)}: beats {start_beat}-{start_beat+len(batch)-1}...")
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Topic: {topic}\n\nBeats ({len(batch)} total -- output exactly {len(batch)} scenes):\n{json.dumps(batch, indent=2)}\n\nCompose each scene like a real motion designer. Vary your layouts. Use the full canvas. Make numbers visual. Make impact words massive. Use white as your base, yellow for one key emphasis per scene."}
                ],
                response_format={"type": "json_object"},
                temperature=0.85,
                max_tokens=8000,
                timeout=120,
            )
            result = json.loads(response.choices[0].message.content)
            batch_scenes = result.get('scenes', [])
            if len(batch_scenes) != len(batch):
                print(f"  ⚠️  Batch {batch_idx+1}: expected {len(batch)} scenes, got {len(batch_scenes)} -- padding with empty scenes")
                while len(batch_scenes) < len(batch):
                    batch_scenes.append({"beat_index": start_beat + len(batch_scenes), "elements": []})
            all_scenes.extend(batch_scenes)
            print(f"  ✅ Batch {batch_idx+1} done: {len(batch_scenes)} scenes")
        except Exception as e:
            print(f"  ❌ Batch {batch_idx+1} failed: {e}")
            raise

    print(f"  ✅ {len(all_scenes)} total scenes composed")
    return all_scenes


# ============================================================
# VALIDATION PASS - validates scene compositions
# ============================================================
def validate_decisions(scenes: list, beats: list) -> list:
    print(f"  🔍 Validating {len(scenes)} scenes...")
    fixed = 0

    for scene_pos, scene in enumerate(scenes):
        if not isinstance(scene, dict):
            scenes[scene_pos] = {"beat_index": scene_pos, "elements": []}
            continue

        # Use enumeration position -- ignore GPT's beat_index
        scene["beat_index"] = scene_pos
        beat = beats[scene_pos] if scene_pos < len(beats) else {}
        beat_text = beat.get("text", "").strip().lower()
        beat_words = set()
        for w in beat_text.split():
            beat_words.add(w.strip('.,!?;:\'"()[]- '))

        elements = scene.get("elements", [])
        if not isinstance(elements, list):
            scene["elements"] = []
            continue

        cleaned = []
        for el in elements:
            if not isinstance(el, dict):
                continue
            etype = el.get("type", "text")

            # Validate TEXT content against beat text
            if etype == "text":
                content = (el.get("content") or "").strip()
                if not content:
                    continue
                # Strip punctuation, lowercase for comparison
                check_words = [w.strip('.,!?;:\'"()[]- ').lower()
                               for w in content.split()
                               if len(w.strip('.,!?;:\'"()[]- ')) > 2]
                if check_words and beat_words:
                    matches = sum(1 for w in check_words if w in beat_words)
                    if matches == 0 and len(check_words) > 0:
                        # Pure hallucination -- skip element
                        print(f"  ⚠ Scene {scene_pos}: dropped hallucinated text '{content[:30]}'")
                        fixed += 1
                        continue

                # Default text properties
                el.setdefault("x", 0.5)
                el.setdefault("y", 0.5)
                el.setdefault("anchor", "center")
                el.setdefault("size", 90)
                el.setdefault("color", "#FFFFFF")
                el.setdefault("weight", "black")
                el.setdefault("outline", 4)
                el.setdefault("anim", "fade_in")
                el.setdefault("start_offset", 0.0)
                el.setdefault("duration", None)
                el.setdefault("anim_duration", 0.15)
                el.setdefault("effect", "none")

            elif etype == "line":
                el.setdefault("x1", 0.3)
                el.setdefault("y1", 0.5)
                el.setdefault("x2", 0.7)
                el.setdefault("y2", 0.5)
                el.setdefault("thickness", 6)
                el.setdefault("color", "#FFFFFF")
                el.setdefault("anim", "draw_horizontal")
                el.setdefault("start_offset", 0.0)
                el.setdefault("duration", None)
                el.setdefault("anim_duration", 0.3)

            elif etype == "rect":
                el.setdefault("x", 0.4)
                el.setdefault("y", 0.4)
                el.setdefault("w", 0.2)
                el.setdefault("h", 0.1)
                el.setdefault("color", "#FFFFFF")
                el.setdefault("filled", True)
                el.setdefault("thickness", 3)
                el.setdefault("anim", "fade_in")
                el.setdefault("start_offset", 0.0)
                el.setdefault("duration", None)
                el.setdefault("anim_duration", 0.2)

            elif etype == "circle":
                el.setdefault("x", 0.5)
                el.setdefault("y", 0.5)
                el.setdefault("radius", 0.05)
                el.setdefault("color", "#FFFFFF")
                el.setdefault("filled", False)
                el.setdefault("thickness", 4)
                el.setdefault("anim", "fade_in")
                el.setdefault("start_offset", 0.0)
                el.setdefault("duration", None)
                el.setdefault("anim_duration", 0.2)

            else:
                # Unknown type, skip
                continue

            # Clamp coordinates to safe range
            for k in ("x", "y", "x1", "y1", "x2", "y2", "w", "h", "radius"):
                if k in el and isinstance(el[k], (int, float)):
                    el[k] = max(0.0, min(1.0, float(el[k])))

            cleaned.append(el)

        scene["elements"] = cleaned

    print(f"  ✅ Validated {len(scenes)} scenes, fixed {fixed} issues")
    return scenes



# ============================================================
# SCENE-BASED RENDERER v5
# GPT outputs scene compositions (lists of elements).
# This renderer executes any combination of text/line/rect/circle
# with per-element animation and timing.
# ============================================================
def render_text_overlay_opencv(video_path: str, scenes: list, beats: list,
                               whisper_segments: list, output_path: str):
    print(f"🎨 Scene renderer v5: {len(scenes)} scenes...")

    try:
        import cv2
        from PIL import Image, ImageDraw, ImageFont
        print(f"  ✓ OpenCV {cv2.__version__} + Pillow ready")
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        subprocess.run(['ffmpeg', '-y', '-i', video_path, '-c', 'copy', output_path],
                       check=True, capture_output=True)
        return

    if not os.path.exists(video_path):
        raise Exception(f"Video not found: {video_path}")

    # ── basic helpers ──────────────────────────────────────────────
    def load_pil_font(path, size, weight="black"):
        # weight selects between Black / ExtraBold / Bold
        try:
            if weight == "regular":
                p = FONT_BOLD or path
            elif weight == "black":
                p = FONT_BLACK or FONT_BOLD or path
            else:
                p = FONT_BOLD or FONT_BLACK or path
            if p and os.path.exists(p):
                return ImageFont.truetype(p, size)
        except: pass
        return ImageFont.load_default()

    def hex_to_rgb(hex_str):
        h = (hex_str or "#FFFFFF").lstrip('#')
        try: return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
        except: return (255, 255, 255)

    def apply_vignette(frame):
        import numpy as np
        rows, cols = frame.shape[:2]
        X = cv2.getGaussianKernel(cols, cols * 0.6)
        Y = cv2.getGaussianKernel(rows, rows * 0.6)
        mask = (Y * X.T) / (Y * X.T).max()
        out = frame.copy().astype(np.float32)
        for i in range(3): out[:,:,i] *= mask
        return np.clip(out, 0, 255).astype(np.uint8)

    def apply_warm_grade(frame):
        import numpy as np
        out = frame.copy().astype(np.float32)
        out[:,:,2] = np.clip(out[:,:,2] * 1.04, 0, 255)
        return out.astype(np.uint8)

    def to_pil(frame):
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def to_frame(pil_img):
        import numpy as np
        return cv2.cvtColor(np.array(pil_img.convert('RGB')), cv2.COLOR_RGB2BGR)

    def composite_layer(frame, layer):
        pil = to_pil(frame).convert('RGBA')
        merged = Image.alpha_composite(pil, layer)
        return to_frame(merged)

    # ── video metadata ─────────────────────────────────────────────
    def _ffprobe_dur(path):
        try:
            r = subprocess.run(['ffprobe','-v','error','-show_entries','format=duration',
                                '-of','default=noprint_wrappers=1:nokey=1', path],
                               capture_output=True, text=True)
            return float(r.stdout.strip())
        except: return 0.0

    TARGET_FPS = 30.0
    vid_dur = _ffprobe_dur(video_path)

    # Transcode to CFR for predictable frame timing
    cfr_video = output_path.replace(".mp4", "_cfr_tmp.mp4")
    subprocess.run(['ffmpeg','-y','-i',video_path,
                    '-vf',f'fps={TARGET_FPS:.0f}',
                    '-c:v','libx264','-preset','ultrafast','-crf','18',
                    '-an', cfr_video],
                   capture_output=True, check=True)

    # ── Whisper word time index for word-by-word timing ───────────
    whisper_word_times = {}
    for seg in whisper_segments:
        seg_start = float(seg.get('start', 0))
        seg_end = float(seg.get('end', 0))
        seg_dur = max(seg_end - seg_start, 0.01)
        word_entries = seg.get('words', [])
        if word_entries:
            for we in word_entries:
                wc = we.get('word', '').upper().strip('.,!?;:\'"()[]- ')
                if not wc: continue
                ws_t = float(we.get('start', seg_start))
                we_t = float(we.get('end', ws_t + 0.2))
                whisper_word_times.setdefault(wc, []).append((ws_t, we_t))

    def find_word_time(word: str, hint_start: float):
        wc = word.upper().strip('.,!?;:\'"()[]- ')
        candidates = whisper_word_times.get(wc, [])
        if not candidates:
            for k, v in whisper_word_times.items():
                if wc in k or k in wc:
                    candidates = v; break
        if not candidates:
            return None
        best = min(candidates, key=lambda x: abs(x[0] - hint_start))
        return best[0]

    def clamp(v, lo, hi): return max(lo, min(v, hi))

    # ── Build flat timeline: each element gets absolute start/end times ─────
    timeline = []
    for scene_pos, scene in enumerate(scenes):
        beat = beats[scene_pos] if scene_pos < len(beats) else {}
        beat_start = clamp(float(beat.get("start_time", 0.0)), 0, vid_dur - 0.1)
        beat_end = clamp(float(beat.get("end_time", beat_start + 2.0)),
                         beat_start + 0.1, vid_dur)

        # Cap beat_end at next beat's start so elements never bleed into next beat
        if scene_pos + 1 < len(beats):
            next_start = float(beats[scene_pos + 1].get("start_time", beat_end))
            beat_end = min(beat_end, next_start)

        elements = scene.get("elements", [])
        for el in elements:
            etype = el.get("type", "text")

            # Resolve start time: prefer Whisper word match for text elements
            start_offset = float(el.get("start_offset", 0.0))
            anim_start = beat_start + start_offset

            # If text matches a word in the beat, use exact Whisper timestamp
            if etype == "text":
                content = (el.get("content") or "").strip()
                if content and " " not in content:
                    # single word - try Whisper timestamp
                    ws = find_word_time(content, beat_start + start_offset)
                    if ws is not None and beat_start <= ws <= beat_end + 1.0:
                        anim_start = ws

            duration = el.get("duration")
            if duration is None or duration <= 0:
                anim_end = beat_end
            else:
                anim_end = anim_start + float(duration)

            # Never let element extend past beat_end (which is now capped at next beat start)
            anim_end = clamp(anim_end, anim_start + 0.1, beat_end)
            anim_start = clamp(anim_start, 0, beat_end - 0.05)

            timeline.append({
                "el": el,
                "start": anim_start,
                "end": anim_end,
                "anim_duration": float(el.get("anim_duration", 0.15)),
            })

    timeline.sort(key=lambda x: x["start"])
    print(f"  📊 Timeline: {len(timeline)} elements")

    # ── Frame-by-frame render ─────────────────────────────────────
    cap = cv2.VideoCapture(cfr_video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_vid = TARGET_FPS

    temp_video = output_path.replace(".mp4", "_noaudio_tmp.mp4")
    out = cv2.VideoWriter(temp_video, cv2.VideoWriter_fourcc(*'mp4v'),
                          fps_vid, (fw, fh))

    print(f"  🎬 {total_frames} frames @ {fps_vid:.0f}fps...")
    frame_idx = 0
    prev_pct = -1

    # ── Element drawing functions ──────────────────────────────────
    def get_anim_progress(el_t, start, end, anim_dur):
        """Return (entrance_progress, exit_progress) both 0..1.
        entrance_progress: 0=not started, 1=fully appeared
        exit_progress: 1=visible, 0=fully gone (only at end)
        """
        if anim_dur <= 0:
            anim_dur = 0.001
        entrance = clamp((el_t - start) / anim_dur, 0.0, 1.0)
        # No exit fade by default - just snap out at end
        return entrance

    def draw_text_element(layer, el, el_t, anim_t):
        """Draw a TEXT element with animation."""
        draw = ImageDraw.Draw(layer)
        content = el.get("content", "")
        x_pct = float(el.get("x", 0.5))
        y_pct = float(el.get("y", 0.5))
        size = max(20, min(int(el.get("size", 90)), 400))
        color = hex_to_rgb(el.get("color", "#FFFFFF"))
        weight = el.get("weight", "black")
        outline = max(0, min(int(el.get("outline", 4)), 12))
        anim = el.get("anim", "fade_in")
        anchor = el.get("anchor", "center")
        effect = el.get("effect", "none")

        font = load_pil_font(get_primary_font_path(), size, weight)
        try:
            bbox = font.getbbox(content)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
        except:
            tw = size * len(content) * 0.55
            th = size

        # Compute base position
        target_x = int(OUTPUT_WIDTH * x_pct)
        target_y = int(OUTPUT_HEIGHT * y_pct)
        if anchor == "center":
            base_x = target_x - tw // 2
            base_y = target_y - th // 2
        elif anchor == "left":
            base_x = target_x
            base_y = target_y - th // 2
        elif anchor == "right":
            base_x = target_x - tw
            base_y = target_y - th // 2
        else:
            base_x = target_x - tw // 2
            base_y = target_y - th // 2

        # Clamp to screen with padding so text never goes off-edge
        pad = 30
        max_x = OUTPUT_WIDTH - tw - pad
        max_y = OUTPUT_HEIGHT - th - pad
        base_x = max(pad, min(base_x, max_x))
        base_y = max(pad, min(base_y, max_y))

        # Apply entrance animation
        draw_x, draw_y = base_x, base_y
        alpha = 1.0
        scale = 1.0

        if anim == "fade_in":
            alpha = anim_t
        elif anim == "slide_in_left":
            slide_dist = int(OUTPUT_WIDTH * 0.3)
            draw_x = base_x - int(slide_dist * (1.0 - anim_t))
            alpha = anim_t
        elif anim == "slide_in_right":
            slide_dist = int(OUTPUT_WIDTH * 0.3)
            draw_x = base_x + int(slide_dist * (1.0 - anim_t))
            alpha = anim_t
        elif anim == "slide_in_top":
            slide_dist = int(OUTPUT_HEIGHT * 0.2)
            draw_y = base_y - int(slide_dist * (1.0 - anim_t))
            alpha = anim_t
        elif anim == "slide_in_bottom":
            slide_dist = int(OUTPUT_HEIGHT * 0.2)
            draw_y = base_y + int(slide_dist * (1.0 - anim_t))
            alpha = anim_t
        elif anim == "scale_in":
            scale = 1.3 - 0.3 * anim_t
            alpha = anim_t
        elif anim == "snap":
            alpha = 1.0
        elif anim == "none":
            alpha = 1.0

        # Effects (applied during display)
        if effect == "flicker":
            # Blink during first 0.3s of element life
            if el_t - 0 < 0.3:
                frame_no = int(el_t * 30)
                if frame_no % 2 == 1:
                    return
        elif effect == "shake":
            import random as _r
            draw_x += _r.randint(-3, 3)
            draw_y += _r.randint(-3, 3)

        # Re-render text at scaled size if scale != 1.0
        render_font = font
        if abs(scale - 1.0) > 0.02:
            new_size = max(20, int(size * scale))
            render_font = load_pil_font(get_primary_font_path(), new_size, weight)
            try:
                bbox = render_font.getbbox(content)
                tw2 = bbox[2] - bbox[0]
                th2 = bbox[3] - bbox[1]
                draw_x = base_x + (tw - tw2) // 2
                draw_y = base_y + (th - th2) // 2
            except: pass

        # Clamp draw position AFTER animation offsets (slide/shake can push text off-screen)
        draw_x = max(pad, min(draw_x, OUTPUT_WIDTH - tw - pad))
        draw_y = max(pad, min(draw_y, OUTPUT_HEIGHT - th - pad))

        a_int = max(0, min(int(255 * alpha), 255))
        if a_int < 5:
            return

        # Draw outline (multiple offsets for thick outline)
        if outline > 0:
            for ox in range(-outline, outline + 1):
                for oy in range(-outline, outline + 1):
                    if ox * ox + oy * oy <= outline * outline:
                        if ox == 0 and oy == 0:
                            continue
                        draw.text((draw_x + ox, draw_y + oy), content,
                                  font=render_font, fill=(0, 0, 0, a_int))
        # Draw fill
        draw.text((draw_x, draw_y), content, font=render_font,
                  fill=(color[0], color[1], color[2], a_int))

    def draw_line_element(layer, el, el_t, anim_t):
        """Draw a LINE element with animation."""
        draw = ImageDraw.Draw(layer)
        x1 = int(OUTPUT_WIDTH * float(el.get("x1", 0.3)))
        y1 = int(OUTPUT_HEIGHT * float(el.get("y1", 0.5)))
        x2 = int(OUTPUT_WIDTH * float(el.get("x2", 0.7)))
        y2 = int(OUTPUT_HEIGHT * float(el.get("y2", 0.5)))
        thickness = max(1, int(el.get("thickness", 6)))
        color = hex_to_rgb(el.get("color", "#FFFFFF"))
        anim = el.get("anim", "draw_horizontal")

        alpha = 1.0
        end_x, end_y = x2, y2

        if anim == "fade_in":
            alpha = anim_t
        elif anim == "draw_horizontal":
            end_x = x1 + int((x2 - x1) * anim_t)
            end_y = y1 + int((y2 - y1) * anim_t)
            alpha = 1.0
        elif anim == "none":
            alpha = 1.0

        a_int = max(0, min(int(255 * alpha), 255))
        if a_int < 5:
            return

        draw.line([(x1, y1), (end_x, end_y)],
                  fill=(color[0], color[1], color[2], a_int),
                  width=thickness)

    def draw_rect_element(layer, el, el_t, anim_t):
        """Draw a RECT element."""
        draw = ImageDraw.Draw(layer)
        x = int(OUTPUT_WIDTH * float(el.get("x", 0.4)))
        y = int(OUTPUT_HEIGHT * float(el.get("y", 0.4)))
        w = int(OUTPUT_WIDTH * float(el.get("w", 0.2)))
        h = int(OUTPUT_HEIGHT * float(el.get("h", 0.1)))
        color = hex_to_rgb(el.get("color", "#FFFFFF"))
        filled = bool(el.get("filled", True))
        thickness = max(1, int(el.get("thickness", 3)))
        anim = el.get("anim", "fade_in")

        alpha = 1.0
        if anim == "fade_in":
            alpha = anim_t
        elif anim == "scale_in":
            scale = anim_t
            cx, cy = x + w // 2, y + h // 2
            w = int(w * scale); h = int(h * scale)
            x = cx - w // 2; y = cy - h // 2
            alpha = anim_t

        a_int = max(0, min(int(255 * alpha), 255))
        if a_int < 5:
            return

        rgba = (color[0], color[1], color[2], a_int)
        if filled:
            draw.rectangle([x, y, x + w, y + h], fill=rgba)
        else:
            draw.rectangle([x, y, x + w, y + h], outline=rgba, width=thickness)

    def draw_circle_element(layer, el, el_t, anim_t):
        """Draw a CIRCLE element."""
        draw = ImageDraw.Draw(layer)
        cx = int(OUTPUT_WIDTH * float(el.get("x", 0.5)))
        cy = int(OUTPUT_HEIGHT * float(el.get("y", 0.5)))
        r = int(min(OUTPUT_WIDTH, OUTPUT_HEIGHT) * float(el.get("radius", 0.05)))
        color = hex_to_rgb(el.get("color", "#FFFFFF"))
        filled = bool(el.get("filled", False))
        thickness = max(1, int(el.get("thickness", 4)))
        anim = el.get("anim", "fade_in")

        alpha = 1.0
        if anim == "fade_in":
            alpha = anim_t
        elif anim == "scale_in":
            r = int(r * anim_t)
            alpha = anim_t

        a_int = max(0, min(int(255 * alpha), 255))
        if a_int < 5 or r <= 0:
            return

        rgba = (color[0], color[1], color[2], a_int)
        bbox = [cx - r, cy - r, cx + r, cy + r]
        if filled:
            draw.ellipse(bbox, fill=rgba)
        else:
            draw.ellipse(bbox, outline=rgba, width=thickness)

    # ── Main frame loop ──────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret: break

        t = frame_idx / fps_vid
        frame = apply_vignette(frame)
        frame = apply_warm_grade(frame)

        # Find all elements active at time t
        raw_active = [item for item in timeline
                      if item["start"] <= t < item["end"]]

        # Deduplicate: if two elements share the same content + near-same x/y position,
        # keep only the one whose start time is closest to t (most recently activated).
        # This prevents "WE WE" doubles when GPT repeats a word across adjacent beats.
        seen_keys = {}
        for item in sorted(raw_active, key=lambda x: x["start"], reverse=True):
            el = item["el"]
            if el.get("type") == "text":
                content_key = (el.get("content", "").upper().strip(),
                               round(float(el.get("x", 0.5)), 1),
                               round(float(el.get("y", 0.5)), 1))
                if content_key not in seen_keys:
                    seen_keys[content_key] = item
            else:
                seen_keys[id(item["el"])] = item
        active = list(seen_keys.values())

        if active:
            # Slight darkening overlay when text is on screen
            import numpy as np
            frame = cv2.addWeighted(frame, 0.82, np.zeros_like(frame), 0.18, 0)

            # Build composite layer
            layer = Image.new('RGBA', (OUTPUT_WIDTH, OUTPUT_HEIGHT), (0, 0, 0, 0))

            for item in active:
                el = item["el"]
                el_t = t - item["start"]   # time since element started
                anim_t = get_anim_progress(el_t, 0, item["end"] - item["start"],
                                            item["anim_duration"])
                etype = el.get("type", "text")
                try:
                    if etype == "text":
                        draw_text_element(layer, el, el_t, anim_t)
                    elif etype == "line":
                        draw_line_element(layer, el, el_t, anim_t)
                    elif etype == "rect":
                        draw_rect_element(layer, el, el_t, anim_t)
                    elif etype == "circle":
                        draw_circle_element(layer, el, el_t, anim_t)
                except Exception as e:
                    print(f"  ⚠ element render error: {e}")

            frame = composite_layer(frame, layer)

        out.write(frame)
        frame_idx += 1
        pct = int(frame_idx / max(total_frames, 1) * 20)
        if pct != prev_pct:
            print(f"  [{'█' * pct}{'░' * (20 - pct)}] {frame_idx}/{total_frames}",
                  end='\r')
            prev_pct = pct

    cap.release(); out.release()
    if os.path.exists(cfr_video): os.remove(cfr_video)
    print(f"\n  ✓ Frames done")

    result = subprocess.run([
        'ffmpeg', '-y', '-i', temp_video, '-i', video_path,
        '-map', '0:v', '-map', '1:a',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
        '-c:a', 'copy', output_path
    ], capture_output=True)

    if os.path.exists(temp_video): os.remove(temp_video)
    if result.returncode != 0:
        raise Exception(f"Audio merge failed: {result.stderr.decode()[-200:]}")

    print(f"  ✅ Render complete: {output_path}")



# ============================================================
# MAIN GENERATOR
# ============================================================
class VaultsGenerator:
    def __init__(self, audio_path: str, output_path: str = "output.mp4", niche_config: dict = None):
        self.audio_path  = audio_path
        self.output_path = output_path

        if niche_config:
            self.broll_dirs  = niche_config.get('broll_dirs', {})
            self.keyword_map = niche_config.get('keyword_map', {})
        else:
            self.broll_dirs = {
                'space':   'space_vids',
                'ancient': 'ancient_ruins_vids',
                'cosmic':  'cosmic_vids',
                'sky':     'dark_sky_vids',
                'temple':  'temple_vids',
            }
            self.keyword_map = {
                'space':   ['universe', 'galaxy', 'black hole', 'star', 'planet', 'cosmos'],
                'ancient': ['ancient', 'civilization', 'pyramid', 'ruins', 'lost', 'forgotten'],
                'cosmic':  ['time', 'reality', 'dimension', 'quantum', 'existence', 'consciousness'],
                'sky':     ['sky', 'atmosphere', 'above', 'beyond', 'vast', 'endless'],
                'temple':  ['religion', 'god', 'sacred', 'ritual', 'belief', 'worship'],
            }

    def get_audio_duration(self) -> float:
        cmd    = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                  '-of', 'default=noprint_wrappers=1:nokey=1', self.audio_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"ffprobe failed: {result.stderr}")
        return float(result.stdout.strip())

    def get_video_info(self, filepath: str):
        cmd    = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                  '-show_entries', 'stream=width,height', '-of', 'json', filepath]
        result = subprocess.run(cmd, capture_output=True, text=True)
        try:
            data = json.loads(result.stdout)
            w    = data['streams'][0]['width']
            h    = data['streams'][0]['height']
            return w, h, w / h
        except:
            return None, None, None

    def get_all_files_from_dir(self, directory: str) -> list:
        if not os.path.exists(directory):
            return []
        files = [os.path.join(directory, f) for f in os.listdir(directory)
                 if f.lower().endswith(('.mp4', '.mov', '.avi'))]
        if not files:
            print(f"  ⚠ Folder exists but is EMPTY: {directory}")
        return files

    def transcribe_with_whisper(self, model: str = "base") -> dict | None:
        cache_file = f"{os.path.splitext(self.audio_path)[0]}_transcription.json"
        if os.path.exists(cache_file):
            print(f"  ✅ Cached transcription")
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        try:
            import whisper
            if not hasattr(whisper, 'load_model'):
                raise ImportError("Wrong whisper. Run: pip install openai-whisper")
            print(f"  🎤 Transcribing ({model})...")
            wm     = whisper.load_model(model)
            result = wm.transcribe(self.audio_path, word_timestamps=True, language="en")
            with open(cache_file, 'w') as f:
                json.dump(result, f, indent=2)
            return result
        except Exception as e:
            print(f"  ❌ Whisper error: {e}")
            return None

    def match_broll_categories(self, full_text: str) -> list:
        text   = full_text.lower()
        scores = {cat: sum(text.count(k) for k in kws)
                  for cat, kws in self.keyword_map.items()}
        sorted_cats = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top = [self.broll_dirs[c] for c, s in sorted_cats if s > 0 and c in self.broll_dirs]

        # FIX 2: Filter to only folders that actually exist and have clips
        valid_top = []
        for folder in top:
            files = self.get_all_files_from_dir(folder)
            if files:
                valid_top.append(folder)
            else:
                print(f"  ⚠ Skipping empty/missing broll folder: {folder}")

        if not valid_top:
            # Fallback: scan ALL configured broll dirs and use any that have clips
            print(f"  ⚠ No keyword-matched folders had clips -- scanning all broll dirs...")
            for folder in self.broll_dirs.values():
                files = self.get_all_files_from_dir(folder)
                if files:
                    valid_top.append(folder)
                    print(f"  ✓ Found clips in: {folder} ({len(files)} files)")

        if not valid_top:
            raise Exception(
                "No broll clips found in ANY configured folder.\n"
                f"Configured dirs: {list(self.broll_dirs.values())}\n"
                "Add your Seedance space/ancient/cosmic clips to these folders."
            )

        return valid_top

    def create_segment_plan(self, duration: float, beats: list, top_categories: list) -> list:
        segments = []
        # Build complete pool of all folders that have clips
        all_folders = []
        for folder in self.broll_dirs.values():
            if self.get_all_files_from_dir(folder):
                all_folders.append(folder)
        if not all_folders:
            raise Exception("No broll clips found in any folder.")

        # Per-folder clip pools with used tracking
        folder_pools = {}
        for folder in all_folders:
            folder_pools[folder] = list(self.get_all_files_from_dir(folder))

        # Beat category → preferred folder (best-effort, falls back to rotation)
        broll_cat_to_folder = {
            'space':   self.broll_dirs.get('space',   'space_vids'),
            'ancient': self.broll_dirs.get('ancient', 'ancient_ruins_vids'),
            'cosmic':  self.broll_dirs.get('cosmic',  'cosmic_vids'),
            'sky':     self.broll_dirs.get('sky',     'dark_sky_vids'),
            'temple':  self.broll_dirs.get('temple',  'temple_vids'),
        }

        # Per-folder shuffle indices so we cycle without repeating
        folder_idx = {f: 0 for f in all_folders}
        for f in all_folders:
            random.shuffle(folder_pools[f])

        base_dur   = 4.0
        n_segs     = max(int(duration / base_dur), 1)
        folder_rot = 0

        for i in range(n_segs):
            seg_dur = float(beats[i].get('clip_duration', base_dur)) if i < len(beats) else base_dur
            # Pure round-robin across all folders -- guaranteed variety
            target_folder = all_folders[folder_rot % len(all_folders)]
            folder_rot += 1

            # Pick next clip from folder, cycling
            pool = folder_pools[target_folder]
            idx  = folder_idx[target_folder]
            if idx >= len(pool):
                random.shuffle(pool)
                idx = 0
            chosen = pool[idx]
            folder_idx[target_folder] = idx + 1

            segments.append({
                'type':     'broll',
                'category': target_folder,
                'file':     chosen,
                'duration': seg_dur,
            })
            print(f"    seg {i+1}: {os.path.basename(chosen)} [{os.path.basename(target_folder)}]")

        if not segments:
            raise Exception("No segments created.")

        total = sum(s['duration'] for s in segments)
        if total < duration:
            segments[-1]['duration'] += (duration - total)

        return segments

    def process_segment_to_file(self, segment: dict, output_file: str,
                                fps: int = 30, progress_callback=None) -> str | None:
        dur = segment['duration']
        w, h, aspect = self.get_video_info(segment['file'])

        cmd = ['ffmpeg', '-y', '-progress', 'pipe:1', '-nostats',
               '-i', segment['file'], '-t', str(dur)]

        vf = []
        if aspect and aspect < (OUTPUT_WIDTH / OUTPUT_HEIGHT):
            vf += [f"scale={OUTPUT_WIDTH}:-2:force_original_aspect_ratio=decrease",
                   f"pad={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:(ow-iw)/2:(oh-ih)/2:black"]
        else:
            vf += [f"scale=-2:{OUTPUT_HEIGHT}:force_original_aspect_ratio=increase",
                   f"crop={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}"]

        # fps=30 MUST come first to convert VFR → CFR before any other filter
        # Without this, VFR clips produce frozen frames or wrong duration
        vf = [f"fps=30"] + vf
        vf += ["eq=brightness=0.02:contrast=1.05:saturation=1.1", "format=yuv420p"]
        cmd += ['-vf', ','.join(vf), '-c:v', 'libx264', '-preset', 'ultrafast',
                '-crf', '23', '-pix_fmt', 'yuv420p', '-r', '30', '-an', output_file]

        if progress_callback:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    universal_newlines=True, bufsize=1)
            total_f = int(dur * fps)
            last_f  = 0
            for line in proc.stdout:  # progress goes to stdout via -progress pipe:1
                if line.startswith('frame='):
                    try:
                        cf = int(line.split('=')[1].strip())
                        if cf > last_f:
                            last_f = cf
                            progress_callback(cf, total_f)
                    except:
                        pass
            proc.wait()
            if proc.returncode != 0:
                print(f"\n  ⚠ Segment failed -- skipping")
                return None
        else:
            r = subprocess.run(cmd, capture_output=True)
            if r.returncode != 0:
                print(f"  ⚠ Skipping: {os.path.basename(segment['file'])}")
                return None

        return output_file

    def _add_cta_overlay(self, video_input: str, output_path: str, duration: float):
        end_time = round(max(duration - 4, 1), 3)
        vf = (
            f"drawtext=text='Vaults of History'"
            f":fontcolor=yellow:fontsize=42:font=Arial"
            f":borderw=2:bordercolor=black:shadowx=2:shadowy=2"
            f":x=(w-text_w)/2:y=h*0.91:enable='gt(t\\,{end_time})'"
        )
        result = subprocess.run(
            ['ffmpeg', '-y', '-i', video_input, '-vf', vf, '-c:a', 'copy', output_path],
            capture_output=True
        )
        if result.returncode != 0:
            subprocess.run(['ffmpeg', '-y', '-i', video_input, '-c', 'copy', output_path],
                           check=True, capture_output=True)
        else:
            print(f"  ✨ CTA added")

    def create_vaults_video(self, bg_volume: float = 0.12, fps: int = 30) -> bool:
        import time
        t0 = time.time()

        print(f"\n{'='*70}")
        print(f"🏛  VAULTS OF HISTORY v3")
        print(f"{'='*70}")

        try:
            duration = self.get_audio_duration()
            print(f"⏱  {duration:.2f}s")
        except Exception as e:
            raise Exception(f"STEP 1 FAILED: {e}")

        print(f"\n[STEP 2] Transcribing...")
        transcription = self.transcribe_with_whisper()
        if not transcription:
            raise Exception("Transcription failed")

        full_text        = transcription.get('text', '').strip()
        whisper_segments = transcription.get('segments', [])
        print(f"  ✅ {len(full_text)} chars, {len(whisper_segments)} segments")

        print(f"\n[STEP 3] GPT Call 1: Story Beats...")
        try:
            topic_hint   = list(self.broll_dirs.keys())[0] if self.broll_dirs else "space"
            beats_result = analyze_story_beats(full_text, whisper_segments, topic_hint, duration)
            topic        = beats_result.get('topic', 'default')
            beats        = beats_result.get('beats', [])
        except Exception as e:
            raise Exception(f"STEP 3 FAILED: {e}")

        print(f"\n[STEP 4] GPT Call 2: Render Decisions...")
        try:
            decisions = generate_render_decisions(beats, topic)
        except Exception as e:
            raise Exception(f"STEP 4 FAILED: {e}")

        print(f"\n[STEP 5] Validating...")
        decisions = validate_decisions(decisions, beats)

        print(f"\n[STEP 6] Music...")
        bg_music = MUSIC_MAP.get(topic, MUSIC_MAP['default'])
        if not os.path.exists(bg_music):
            bg_music = MUSIC_MAP['default']
            if not os.path.exists(bg_music):
                for fname in (os.listdir('bg_musics') if os.path.exists('bg_musics') else []):
                    if fname.endswith('.mp3'):
                        bg_music = os.path.join('bg_musics', fname)
                        break
                else:
                    bg_music = None
        print(f"  🎵 {bg_music}")

        print(f"\n[STEP 7] B-roll matching...")
        top_categories = self.match_broll_categories(full_text)
        print(f"  📊 {top_categories}")

        print(f"\n[STEP 8] Segment plan...")
        try:
            video_segments = self.create_segment_plan(duration, beats, top_categories)
            print(f"  ✅ {len(video_segments)} segments")
        except Exception as e:
            raise Exception(f"STEP 8 FAILED: {e}")

        temp_files    = []
        concat_list   = "concat_list.txt"
        concat_output = "concatenated_video.mp4"
        audio_output  = "audio_mixed.mp4"

        try:
            print(f"\n[STEP 9] Processing segments...")
            try:
                from tqdm import tqdm
                use_tqdm = True
            except:
                use_tqdm = False

            for i, seg in enumerate(video_segments):
                temp_file = f"temp_segment_{i:02d}.mp4"
                t_seg     = time.time()

                if use_tqdm:
                    total_f = int(seg['duration'] * fps)
                    pbar    = tqdm(total=total_f,
                                   desc=f"  Seg {i+1}/{len(video_segments)}: {os.path.basename(seg['file'])[:28]}",
                                   unit='frame')
                    def upd(c, t, pb=pbar):
                        pb.n = min(c, t); pb.refresh()
                    result = None
                    try:
                        result = self.process_segment_to_file(seg, temp_file, fps, upd)
                    finally:
                        pbar.n = pbar.total; pbar.refresh(); pbar.close()
                        print(f"    ✓ {time.time()-t_seg:.1f}s")
                    if result is None:
                        continue
                else:
                    print(f"  {i+1}/{len(video_segments)}: {os.path.basename(seg['file'])}", end='', flush=True)
                    result = self.process_segment_to_file(seg, temp_file, fps)
                    if result is None:
                        print(f" ⚠ skipped")
                        continue
                    print(f" ✓ ({time.time()-t_seg:.1f}s)")

                if os.path.exists(temp_file):
                    temp_files.append(temp_file)

            if not temp_files:
                raise Exception("No segments processed")

            print(f"\n[STEP 10] Concatenating {len(temp_files)} segments...")
            with open(concat_list, 'w') as f:
                for tf in temp_files:
                    f.write(f"file '{tf}'\n")
            r = subprocess.run(['ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                                 '-i', concat_list, '-c', 'copy', concat_output],
                                capture_output=True)
            if r.returncode != 0:
                raise Exception(f"Concat failed: {r.stderr.decode()[-200:]}")
            print(f"  ✅ Done")

            print(f"\n[STEP 11] Audio mix...")
            cmd = ['ffmpeg', '-y', '-i', concat_output, '-i', self.audio_path]
            if bg_music and os.path.exists(bg_music):
                cmd += ['-i', bg_music]
                fc = (
                    f'[1:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,volume=1.0[voice];'
                    f'[2:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,'
                    f'volume={bg_volume},aloop=loop=-1:size=2e+09[bg];'
                    f'[voice][bg]amix=inputs=2:duration=first:dropout_transition=2,aresample=48000[aout]'
                )
                cmd += ['-filter_complex', fc, '-map', '0:v', '-map', '[aout]']
            else:
                cmd += ['-map', '0:v', '-map', '1:a']
            cmd += ['-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
                    '-ar', '48000', '-ac', '2', '-t', str(duration), audio_output]
            r = subprocess.run(cmd, capture_output=True)
            if r.returncode != 0:
                raise Exception(f"Audio failed: {r.stderr.decode()[-200:]}")
            print(f"  ✅ Mixed")

            print(f"\n[STEP 12] OpenCV text render...")
            try:
                render_text_overlay_opencv(audio_output, decisions, beats, whisper_segments, self.output_path)
            except Exception as e:
                print(f"  ❌ Render failed: {e}")
                traceback.print_exc()
                subprocess.run(['ffmpeg', '-y', '-i', audio_output, '-c', 'copy', self.output_path],
                               check=True, capture_output=True)

            if os.path.exists(audio_output):
                os.remove(audio_output)

            print(f"\n[STEP 13] CTA...")
            cta_output = self.output_path.replace(".mp4", "_cta.mp4")
            self._add_cta_overlay(self.output_path, cta_output, duration)
            self.output_path = cta_output

            if not os.path.exists(self.output_path):
                raise Exception(f"Output missing: {self.output_path}")

            file_size  = os.path.getsize(self.output_path) / (1024 * 1024)
            total_time = time.time() - t0

            print(f"\n{'='*70}")
            print(f"✅ COMPLETE!")
            print(f"📁 {self.output_path}")
            print(f"💾 {file_size:.2f} MB | ⏱ {duration:.1f}s | ⚡ {total_time:.0f}s")
            print(f"🎭 {len(beats)} beats | 📝 {len(decisions)} decisions")
            print(f"{'='*70}\n")
            return True

        except Exception as e:
            print(f"\n❌ Pipeline error: {e}")
            traceback.print_exc()
            return False

        finally:
            print(f"\n🧹 Cleanup...")
            for tf in temp_files:
                if os.path.exists(tf):
                    try: os.remove(tf)
                    except: pass
            for f in [concat_list, concat_output, audio_output]:
                if os.path.exists(f):
                    try: os.remove(f)
                    except: pass
            for tmp in glob.glob("*TEMP_MPY*.mp4") + glob.glob("*_noaudio_tmp.mp4") + glob.glob("*_cfr_tmp.mp4") + glob.glob("temp_segment_*.mp4"):
                try: os.remove(tmp)
                except: pass


# ============================================================
# NICHE TEMPLATES
# ============================================================
NICHE_TEMPLATES = {
    'vaults': {
        'broll_dirs': {
            'space':   'space_vids',
            'ancient': 'ancient_ruins_vids',
            'cosmic':  'cosmic_vids',
            'sky':     'dark_sky_vids',
            'temple':  'temple_vids',
        },
        'keyword_map': {
            'space':   ['universe', 'galaxy', 'black hole', 'star', 'planet', 'cosmos'],
            'ancient': ['ancient', 'civilization', 'pyramid', 'ruins', 'lost', 'forgotten'],
            'cosmic':  ['time', 'reality', 'dimension', 'quantum', 'existence', 'consciousness'],
            'sky':     ['sky', 'atmosphere', 'above', 'beyond', 'vast', 'endless'],
            'temple':  ['religion', 'god', 'sacred', 'ritual', 'belief', 'worship'],
        }
    },
}


# ============================================================
# FASTAPI
# ============================================================
@app.get("/")
def root():
    return {"service": "Vaults of History v3", "status": "running",
            "openai_key": bool(OPENAI_API_KEY)}

@app.post("/generate")
async def generate_video_api(background_tasks: BackgroundTasks, niche: str = "vaults"):
    global current_job
    if current_job["status"] == "processing":
        return {"message": "Already processing", "status": "processing"}
    current_job = {"status": "processing", "progress": 0, "output": None,
                   "error": None, "started_at": datetime.now().isoformat(), "niche": niche}
    background_tasks.add_task(process_video, niche)
    return {"message": f"Started niche={niche}", "status": "processing"}

def process_video(niche: str = "vaults"):
    global current_job
    try:
        current_job["progress"] = 5
        audio_url   = "https://raw.githubusercontent.com/RandomSci/Automation_For_Love_Niche/main/Audio_Voice/new_love.mp3"
        audio_file  = "Audio_Voice/vaults_narration.mp3"
        output_file = "vaults_output.mp4"
        trans_file  = f"{os.path.splitext(audio_file)[0]}_transcription.json"

        print(f"\n📥 Downloading audio...")
        os.makedirs("Audio_Voice", exist_ok=True)
        resp = requests.get(audio_url, timeout=30)
        if resp.status_code != 200:
            raise Exception(f"HTTP {resp.status_code}")
        with open(audio_file, "wb") as f:
            f.write(resp.content)
        print(f"  ✅ {len(resp.content)//1024}KB")

        current_job["progress"] = 10

        for old in [output_file, output_file.replace(".mp4", "_cta.mp4"),
                    "audio_mixed.mp4", trans_file]:
            if os.path.exists(old):
                os.remove(old)

        current_job["progress"] = 15
        niche_config = NICHE_TEMPLATES.get(niche, NICHE_TEMPLATES['vaults'])
        gen = VaultsGenerator(audio_path=audio_file, output_path=output_file,
                              niche_config=niche_config)

        current_job["progress"] = 20
        success = gen.create_vaults_video(bg_volume=0.12, fps=30)
        current_job["progress"] = 95

        final = output_file.replace(".mp4", "_cta.mp4")
        if success and os.path.exists(final):
            current_job.update({"status": "completed", "progress": 100, "output": final})
            print(f"\n🎉 DONE: {final}")
        else:
            raise Exception("Pipeline failed or output missing")

    except Exception as e:
        current_job.update({"status": "error", "error": str(e), "progress": 0})
        print(f"\n❌ FAILED: {e}")
        traceback.print_exc()

@app.get("/status")
def check_status():
    return {**current_job, "ready": current_job["status"] == "completed",
            "niche": current_job.get("niche", "vaults")}

@app.get("/download")
def download_video():
    if current_job["status"] != "completed":
        raise HTTPException(400, f"Not ready: {current_job['status']}")
    if not current_job["output"] or not os.path.exists(current_job["output"]):
        raise HTTPException(404, "File not found")
    return FileResponse(current_job["output"], media_type="video/mp4",
                        filename=f"vaults_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Vaults v3 on :{port} | Key: {'set' if OPENAI_API_KEY else 'MISSING'}")
    uvicorn.run(app, host="0.0.0.0", port=port)
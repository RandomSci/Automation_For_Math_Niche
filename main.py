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

# ============================================================
# FONT SYSTEM -- Montserrat ExtraBold = sackfeels primary font
# Rounded heavy sans-serif, identical to CapCut word-by-word style
# ============================================================
FONTS_DIR = "fonts"

GOOGLE_FONTS = {
    "Montserrat-ExtraBold": "https://fonts.googleapis.com/css2?family=Montserrat:wght@800&display=swap",
    "Montserrat-Black":     "https://fonts.googleapis.com/css2?family=Montserrat:wght@900&display=swap",
    "Barlow-Black":         "https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@900&display=swap",
    "Oswald-Bold":          "https://fonts.googleapis.com/css2?family=Oswald:wght@700&display=swap",
}

def download_fonts():
    import re
    os.makedirs(FONTS_DIR, exist_ok=True)
    headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'}
    downloaded = []
    for name, css_url in GOOGLE_FONTS.items():
        path = os.path.join(FONTS_DIR, f"{name}.ttf")
        if os.path.exists(path) and os.path.getsize(path) > 1000:
            downloaded.append(path)
            continue
        try:
            r = requests.get(css_url, headers=headers, timeout=10)
            urls = re.findall(r'url\((https://fonts\.gstatic\.com/[^)]+\.ttf)\)', r.text)
            if urls:
                fr = requests.get(urls[0], headers=headers, timeout=10)
                with open(path, 'wb') as f:
                    f.write(fr.content)
                if os.path.getsize(path) > 1000:
                    print(f"  ✅ Font: {name} ({os.path.getsize(path)//1024}KB)")
                    downloaded.append(path)
        except Exception as e:
            print(f"  ⚠ Font download failed {name}: {e}")
    return downloaded

# Download on startup
_downloaded_fonts = download_fonts()

def get_primary_font_path() -> str:
    """Always return Montserrat ExtraBold -- the sackfeels font"""
    for name in ["Montserrat-ExtraBold", "Montserrat-Black", "Barlow-Black", "Oswald-Bold"]:
        path = os.path.join(FONTS_DIR, f"{name}.ttf")
        if os.path.exists(path) and os.path.getsize(path) > 1000:
            return path
    for fallback in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]:
        if os.path.exists(fallback):
            return fallback
    return None

@app.on_event("startup")
async def startup_event():
    print("🚀 Vaults of History v3 starting...")


# ============================================================
# GPT CALL 1 -- STORY BEAT ANALYZER
# ============================================================
def analyze_story_beats(transcript_text: str, topic_hint: str, total_duration: float) -> dict:
    if not OPENAI_API_KEY:
        raise Exception("OPENAI_API_KEY not set.")

    print(f"  🎭 Call 1: Story beats ({len(transcript_text)} chars, {total_duration:.1f}s)...")
    client = OpenAI(api_key=OPENAI_API_KEY)

    system_prompt = f"""You are the story producer for VAULTS OF HISTORY -- a viral mind-bending facts channel.
Style: @sackfeels on TikTok. Dramatic. Eerie. Cinematic. Makes people stop scrolling.
Topics: space mysteries, ancient civilizations, lost history, religion, human consciousness.
Canvas: {OUTPUT_WIDTH}x{OUTPUT_HEIGHT} pixels. Total video duration: {total_duration:.1f} seconds.

TASK: Read the transcript. Segment it into story beats using EXACT timestamps from the audio.

CRITICAL RULES:
- Use ONLY words that are actually spoken in the transcript. NEVER add your own words.
- beat_type: "hook"|"buildup"|"reveal"|"shock"|"pause"|"resolution"|"outro"
- start_time and end_time MUST be real seconds where those words are spoken
- Cover the ENTIRE transcript with no gaps longer than 3 seconds
- hook beat: always first, start_time=0.0, uses the first spoken words
- Each beat text = EXACT verbatim words from transcript at that timestamp
- Keep beats short: 3-8 words each. Split long sentences into multiple beats.
- "pause" beats: 0.5-1.5 second natural silence gaps only

For EVERY beat provide:
- beat_type: "hook"|"buildup"|"reveal"|"shock"|"pause"|"resolution"|"outro"
- text: EXACT verbatim words from transcript
- start_time: seconds when FIRST word of this beat is spoken
- end_time: seconds when LAST word of this beat finishes
- intensity: 1-10
- broll_category: "space"|"ancient"|"cosmic"|"sky"|"temple"|"any"
- clip_duration: 2.0-6.0 seconds

Return ONLY valid JSON:
{{
  "topic": "space|ancient|religion|human|death|default",
  "music_mood": "eerie|dark|mysterious|sacred|cosmic|haunting",
  "beats": [
    {{
      "beat_type": "hook|buildup|reveal|shock|pause|resolution|outro",
      "text": "exact transcript words only",
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
                {"role": "user", "content": f"Topic: {topic_hint}\n\nFull transcript:\n{transcript_text}\n\nSegment this transcript into beats. Use ONLY the exact words spoken. Never add your own commentary."}
            ],
            response_format={"type": "json_object"},
            temperature=0.4,
            timeout=90,
        )
        result = json.loads(response.choices[0].message.content)
        # Filter out any ai_opinion beats that GPT might add despite instructions
        beats = [b for b in result.get('beats', []) if b.get('beat_type') != 'ai_opinion']
        result['beats'] = beats
        print(f"  ✅ {len(beats)} beats, topic={result.get('topic')}")
        return result
    except Exception as e:
        print(f"  ❌ Story beats failed: {e}")
        raise


# ============================================================
# GPT CALL 2 -- SEMANTIC HIGHLIGHT IDENTIFIER
# GPT's ONLY job: identify the emotionally important span per beat
# + importance score. All render decisions made by Python.
# ============================================================
def generate_render_decisions(beats: list, topic: str) -> list:
    if not OPENAI_API_KEY:
        raise Exception("OPENAI_API_KEY not set.")

    print(f"  🎨 Call 2: Semantic highlights for {len(beats)} beats...")
    client = OpenAI(api_key=OPENAI_API_KEY)

    system_prompt = """You are a professional video editor analyzing a transcript for a viral mind-bending facts channel (@sackfeels style on TikTok).

YOUR ONLY JOB: For each beat, identify the most emotionally powerful word or span of words, and score its importance.

RULES:
- highlight_span: the exact word or multi-word phrase that carries the most emotional/semantic weight
  Examples: "US", "dying?", "World War II", "precision", "guaranteed", "impossible"
- importance: float 0.0-1.0
  0.9-1.0 = must isolate visually (shocking, revelatory, emotionally heavy)
  0.6-0.89 = strong highlight, golden inline
  0.0-0.59 = mild, normal white text
- Can be multi-word: "World War II", "Stock Market Crash", "never existed"
- null if the beat has no highlight (pause beats, transitional filler)
- Think like an editor: what word would you PUNCH UP if editing this?

Return ONLY valid JSON:
{
  "highlights": [
    {
      "beat_index": 0,
      "highlight_span": "dying?",
      "importance": 0.95
    }
  ]
}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Topic: {topic}\n\nBeats:\n{json.dumps(beats, indent=2)}\n\nIdentify the emotional highlight span and importance for each beat. Think like a film editor."}
            ],
            response_format={"type": "json_object"},
            temperature=0.5,
            timeout=60,
        )
        result     = json.loads(response.choices[0].message.content)
        highlights = result.get('highlights', [])
        print(f"  ✅ {len(highlights)} highlight decisions")
        return highlights
    except Exception as e:
        print(f"  ❌ Highlight decisions failed: {e}")
        raise


# ============================================================
# PHRASE BUILDER + IMPORTANCE SCORER
# Whisper gaps drive phrase grouping.
# Importance = pause_score + gpt_score + sentence_bonus + position_bonus
# Render decision = relative ranking within full video
# ============================================================
def build_phrase_blocks(word_timestamps: list, highlights: list, beats: list) -> list:
    """
    Groups Whisper words into phrase blocks using gap detection.
    Scores each block. Returns list of phrase_block dicts.

    phrase_block = {
        words: [{word, start, end}],
        start: float,
        end: float,
        score: float,
        highlight_span: str or None,   # the GPT-identified span within this block
        importance: float,             # GPT importance 0-1
        beat_idx: int,
        render_mode: 'normal'|'highlighted'|'isolated'  # set after scoring
        block_color: 'white'|'gold'|'full_gold'         # set after scoring
        y_zone: 'upper'|'center'|'lower'                # set per beat
    }
    """
    if not word_timestamps:
        return []

    # Build highlight lookup: beat_idx → {span, importance}
    hi_map = {}
    for h in highlights:
        bi = int(h.get("beat_index", 0))
        hi_map[bi] = {
            "span":       (h.get("highlight_span") or "").lower().strip(".,!?-'\""),
            "importance": float(h.get("importance", 0.0)),
        }

    # Assign each word to a beat
    beat_windows = []
    for i, beat in enumerate(beats):
        if beat.get("beat_type") == "pause":
            continue
        beat_windows.append({
            "beat_idx": i,
            "start":    float(beat.get("start_time", 0.0)),
            "end":      float(beat.get("end_time", 0.0)),
        })

    def get_beat_for_word(w_start):
        for bw in beat_windows:
            if bw["start"] - 0.3 <= w_start <= bw["end"] + 0.3:
                return bw["beat_idx"]
        if beat_windows:
            return min(beat_windows,
                key=lambda bw: min(abs(w_start - bw["start"]), abs(w_start - bw["end"]))
            )["beat_idx"]
        return 0

    # Assign y_zone per beat (randomized once per beat, consistent within)
    y_zones = {}
    zone_options = ["upper", "center", "lower"]
    for bw in beat_windows:
        y_zones[bw["beat_idx"]] = random.choice(zone_options)

    # Group words into phrase blocks
    # Hard cap: 3 words max per block — sackfeels NEVER shows more than 3 words at once
    # Break on: word cap hit, gap > 0.5s (real breath), beat boundary change
    MAX_WORDS   = 3
    GAP_BREAK   = 0.50   # hard break — definite new thought
    blocks      = []
    current_block_words = []

    for i, ww in enumerate(word_timestamps):
        w_text  = ww["word"].strip()
        w_start = float(ww["start"])
        w_end   = float(ww["end"])
        if not w_text:
            continue

        if not current_block_words:
            current_block_words.append(ww)
        else:
            prev_end  = float(current_block_words[-1]["end"])
            gap       = w_start - prev_end
            prev_beat = get_beat_for_word(float(current_block_words[-1]["start"]))
            curr_beat = get_beat_for_word(w_start)
            at_cap    = len(current_block_words) >= MAX_WORDS

            # Break if: word cap hit OR real pause OR beat changed
            if at_cap or gap >= GAP_BREAK or prev_beat != curr_beat:
                blocks.append(current_block_words)
                current_block_words = [ww]
            else:
                current_block_words.append(ww)

    if current_block_words:
        blocks.append(current_block_words)

    # Score each block
    phrase_blocks = []
    for block_words in blocks:
        b_start   = float(block_words[0]["start"])
        b_end     = float(block_words[-1]["end"])
        beat_idx  = get_beat_for_word(b_start)
        hi_info   = hi_map.get(beat_idx, {"span": "", "importance": 0.0})
        hi_span   = hi_info["span"]
        gpt_imp   = hi_info["importance"]

        # pause_score: gap before this block (0-30)
        if phrase_blocks:
            prev_end   = phrase_blocks[-1]["end"]
            pause_secs = b_start - prev_end
        else:
            pause_secs = 0.0
        pause_score = min(30, pause_secs * 40)

        # gpt_score: GPT importance mapped to 0-40
        # Only applies if this block contains the highlight span
        block_text = " ".join(w["word"].strip().lower().strip(".,!?-'\"")
                               for w in block_words)
        contains_span = hi_span and hi_span in block_text
        gpt_score = (gpt_imp * 40) if contains_span else 0.0

        # sentence_bonus: block ends with sentence-ending punctuation (0-20)
        last_word    = block_words[-1]["word"].strip()
        sentence_end = last_word.endswith(('.', '?', '!'))
        sentence_bonus = 15 if sentence_end else 0

        # position_bonus: last block in beat gets +10
        # (we'll apply this after grouping all blocks by beat)
        score = pause_score + gpt_score + sentence_bonus

        phrase_blocks.append({
            "words":          block_words,
            "start":          b_start,
            "end":            b_end,
            "score":          score,
            "highlight_span": hi_span if contains_span else None,
            "importance":     gpt_imp if contains_span else 0.0,
            "beat_idx":       beat_idx,
            "render_mode":    "normal",
            "block_color":    "white",
            "y_zone":         y_zones.get(beat_idx, "center"),
        })

    # Apply position_bonus: last block per beat gets +10
    beat_last = {}
    for i, pb in enumerate(phrase_blocks):
        beat_last[pb["beat_idx"]] = i
    for i, pb in enumerate(phrase_blocks):
        if beat_last.get(pb["beat_idx"]) == i:
            pb["score"] += 10

    # Relative ranking: compute mean + stddev across all scores
    scores = [pb["score"] for pb in phrase_blocks]
    if len(scores) > 1:
        mean   = sum(scores) / len(scores)
        stddev = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5
    else:
        mean   = scores[0] if scores else 0
        stddev = 1.0

    FILLER_ONLY = {"a", "an", "the", "of", "in", "on", "at", "to", "and", "or",
                   "but", "so", "yet", "for", "nor", "is", "are", "was", "were",
                   "it", "its", "this", "that", "with", "by", "as", "be", "now",
                   "i", "we", "you", "he", "she", "they", "my", "our", "your"}

    isolated_count = 0
    MAX_ISOLATED   = 5  # cap isolated moments per video

    for pb in phrase_blocks:
        s   = pb["score"]
        imp = pb["importance"]

        # A block is ONLY isolated if:
        # 1. Score is high relative to video (mean + stddev)
        # 2. GPT gave it real importance (>= 0.70)
        # 3. It actually contains the highlight span
        # 4. It's not ALL filler words
        block_words_clean = [w["word"].strip().lower().strip(".,!?-'\"")
                             for w in pb["words"] if w["word"].strip()]
        all_filler = all(w in FILLER_ONLY for w in block_words_clean)
        has_span   = pb["highlight_span"] is not None

        can_isolate = (s > mean + stddev and
                       imp >= 0.70 and
                       has_span and
                       not all_filler and
                       isolated_count < MAX_ISOLATED)

        if can_isolate:
            pb["render_mode"] = "isolated"
            pb["block_color"] = "full_gold"
            isolated_count += 1
        elif s > mean and has_span:
            pb["render_mode"] = "highlighted"
            pb["block_color"] = "gold"
        elif pb["highlight_span"]:
            pb["render_mode"] = "highlighted"
            pb["block_color"] = "gold"
        else:
            pb["render_mode"] = "normal"
            pb["block_color"] = "white"

    print(f"  📊 {len(phrase_blocks)} phrase blocks | "
          f"isolated={isolated_count} highlighted={sum(1 for p in phrase_blocks if p['render_mode']=='highlighted')} "
          f"normal={sum(1 for p in phrase_blocks if p['render_mode']=='normal')}")
    return phrase_blocks


# ============================================================
# RENDERER v4 -- SACKFEELS KINETIC TYPOGRAPHY
# Phrase blocks drive everything. Two modes:
#   INLINE: words on one line, whole block fades in/out as unit
#   ISOLATED: single word/span alone, massive, fast flash
# Position: y_zone (upper/center/lower) per beat, centered horizontally
# Colors: white (normal), gold inline (highlighted), full gold (isolated)
# Opacity: setup phrases 75%, payoff 100%
# Between beats: 3-frame dark gap
# ============================================================
def render_text_overlay_opencv(video_path: str, phrase_blocks: list, beats: list,
                               word_timestamps: list, output_path: str):
    print(f"🎨 Renderer v4: {len(phrase_blocks)} phrase blocks...")

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

    _font_path = get_primary_font_path()

    def load_font(size):
        if _font_path:
            try:
                return ImageFont.truetype(_font_path, size)
            except:
                pass
        return ImageFont.load_default()

    def apply_glitch(frame):
        b, g, r = cv2.split(frame)
        shift = random.randint(6, 14)
        return cv2.merge([np.roll(b, -shift, axis=1), g, np.roll(r, shift, axis=1)])

    # Colors
    WHITE     = (255, 255, 255)
    GOLD      = (212, 160, 23)
    FADE_DARK = (0, 0, 0)

    # Font sizes
    INLINE_SIZE   = 90    # base inline phrase size
    ISOLATED_SIZE = 180   # isolated word — massive
    INLINE_HI_SIZE = 105  # highlighted word inline — slightly bigger

    # phrase_blocks already built and scored upstream
    if not phrase_blocks:
        print(f"  ⚠ No phrase blocks -- copying without text")
        subprocess.run(['ffmpeg', '-y', '-i', video_path, '-c', 'copy', output_path],
                       check=True, capture_output=True)
        return

    # Set display_end: hold until just before next block starts (no overlap)
    for i, pb in enumerate(phrase_blocks):
        if i + 1 < len(phrase_blocks):
            next_start = phrase_blocks[i + 1]["start"]
            # Hold for up to 0.15s but NEVER overlap next block
            pb["display_end"] = min(pb["end"] + 0.15, next_start - 0.01)
        else:
            pb["display_end"] = pb["end"] + 0.3

    FADE_FRAMES = 6   # frames for fade in / fade out
    DARK_FRAMES = 3   # brief dark gap between blocks

    # Video capture
    cap          = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_vid      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    vid_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    temp_video = output_path.replace(".mp4", "_noaudio_tmp.mp4")
    fourcc     = cv2.VideoWriter_fourcc(*'mp4v')
    out        = cv2.VideoWriter(temp_video, fourcc, fps_vid, (vid_w, vid_h))

    print(f"  🎬 {total_frames} frames @ {fps_vid:.0f}fps ({vid_w}x{vid_h})...")

    # Pre-compute y positions
    # Inline phrases: left-anchored, upper portion of screen
    # Isolated words: centered horizontally, centered vertically (alone) OR below phrase if phrase visible
    INLINE_X     = int(vid_w * 0.05)   # left anchor for inline phrases
    INLINE_Y     = int(vid_h * 0.22)   # upper area for inline phrases
    ISOLATED_Y   = int(vid_h * 0.42)   # center-ish for isolated words alone
    LINE_H       = INLINE_SIZE + 12

    def get_y_anchor(zone):
        # Keep for compatibility but not used for positioning anymore
        return INLINE_Y

    def measure_text_width(text, font):
        bbox = font.getbbox(text)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]

    def split_line_by_pixels(words_list, font, max_w):
        """Split words into lines based on pixel width, breaking at syntax boundaries."""
        SYNTAX_BREAK = {"and", "but", "or", "so", "yet", "nor", "of", "in",
                        "on", "at", "to", "for", "with", "by", "from"}
        lines   = []
        current = []
        for w in words_list:
            test = " ".join(current + [w])
            tw, _ = measure_text_width(test, font)
            if tw > max_w and current:
                # Try to find a syntax break point in current
                broke = False
                for j in range(len(current) - 1, 0, -1):
                    if current[j].lower().strip(".,!?") in SYNTAX_BREAK:
                        lines.append(current[:j + 1])
                        current = current[j + 1:] + [w]
                        broke = True
                        break
                if not broke:
                    lines.append(current)
                    current = [w]
            else:
                current.append(w)
        if current:
            lines.append(current)
        return lines

    def draw_phrase_block(draw, pb, alpha_mult, vid_w, vid_h, current_t, current_frame, fps_vid):
        """Draw a phrase block. Returns glitch flag."""
        mode       = pb["render_mode"]
        hi_span    = (pb["highlight_span"] or "").lower()
        words_list = [w["word"].strip() for w in pb["words"] if w["word"].strip()]
        do_glitch  = False

        if mode == "isolated":
            text  = " ".join(words_list)
            fsize = ISOLATED_SIZE
            font  = load_font(fsize)
            tw, th = measure_text_width(text, font)
            while tw > vid_w * 0.88 and fsize > 60:
                fsize -= 10
                font   = load_font(fsize)
                tw, th = measure_text_width(text, font)
            x     = (vid_w - tw) // 2
            y     = (vid_h - th) // 2
            color = (*GOLD, int(255 * alpha_mult))
            draw.text((x, y), text, font=font, fill=color)

            # Glitch ONLY on first 4 frames after isolated word appears
            frames_since_start = int((current_t - pb["start"]) * fps_vid)
            if pb["importance"] >= 0.88 and frames_since_start < 4 and current_frame % 2 == 0:
                do_glitch = True

        else:
            # INLINE: left-anchored, upper area, wraps if too long
            max_line_w  = int(vid_w * 0.90)
            font_inline = load_font(INLINE_SIZE)
            lines       = split_line_by_pixels(words_list, font_inline, max_line_w)
            base_alpha  = int(255 * alpha_mult * (0.80 if mode == "normal" else 1.0))
            cur_y       = INLINE_Y

            for line_words in lines:
                x_cursor = INLINE_X
                for wi, word in enumerate(line_words):
                    w_clean = word.lower().strip(".,!?-'\"")
                    in_span = hi_span and w_clean in hi_span
                    is_hi   = (mode == "highlighted") and in_span

                    if is_hi:
                        font_w = load_font(INLINE_HI_SIZE)
                        color  = (*GOLD, base_alpha)
                    else:
                        font_w = font_inline
                        color  = (*WHITE, base_alpha)

                    ww_text = word + (" " if wi < len(line_words) - 1 else "")
                    draw.text((x_cursor, cur_y), ww_text, font=font_w, fill=color)
                    tw_w, _ = measure_text_width(ww_text, font_w)
                    x_cursor += tw_w

                cur_y += LINE_H

        return do_glitch

    frame_idx = 0
    prev_pct  = -1
    n_blocks  = len(phrase_blocks)

    # Build frame→block lookup for efficiency
    # Each frame: find which block is active
    def find_active_block(t):
        for pb in phrase_blocks:
            if pb["start"] <= t <= pb["display_end"]:
                return pb
        return None

    def get_fade_alpha(t, pb):
        """Inline blocks appear instantly. Only isolated blocks fade in/out."""
        if pb["render_mode"] != "isolated":
            return 1.0  # instant — no flicker on inline phrases

        fade_dur    = 4 / fps_vid   # 4 frames fade in, 4 frames fade out
        since_start = t - pb["start"]
        until_end   = pb["display_end"] - t

        if since_start < fade_dur:
            return max(0.0, since_start / fade_dur)
        if until_end < fade_dur:
            return max(0.0, until_end / fade_dur)
        return 1.0

    prev_block_id = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t          = frame_idx / fps_vid
        active_pb  = find_active_block(t)

        if active_pb is not None:
            block_id = id(active_pb)

            # Only apply dark gap when coming FROM an isolated block
            prev_was_isolated = any(
                id(p) == prev_block_id and p["render_mode"] == "isolated"
                for p in phrase_blocks
            ) if prev_block_id is not None else False

            if prev_was_isolated and block_id != prev_block_id:
                since_block_start = t - active_pb["start"]
                if since_block_start < (DARK_FRAMES / fps_vid):
                    frame = cv2.addWeighted(frame, 0.3, np.zeros_like(frame), 0.7, 0)
                    out.write(frame)
                    frame_idx += 1
                    prev_block_id = block_id
                    continue

            prev_block_id = block_id
            alpha = get_fade_alpha(t, active_pb)

            frame_rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img    = Image.fromarray(frame_rgb)
            text_layer = Image.new('RGBA', (vid_w, vid_h), (0, 0, 0, 0))
            draw       = ImageDraw.Draw(text_layer)

            # If isolated, also draw the previous inline phrase faintly above
            if active_pb["render_mode"] == "isolated":
                prev_inline = None
                for pb2 in phrase_blocks:
                    if id(pb2) == block_id:
                        break
                    if pb2["render_mode"] != "isolated":
                        prev_inline = pb2
                if prev_inline:
                    draw_phrase_block(draw, prev_inline, 0.55, vid_w, vid_h, t, frame_idx, fps_vid)

            do_glitch = draw_phrase_block(draw, active_pb, alpha, vid_w, vid_h, t, frame_idx, fps_vid)

            pil_rgba = pil_img.convert('RGBA')
            combined = Image.alpha_composite(pil_rgba, text_layer)
            frame    = cv2.cvtColor(np.array(combined.convert('RGB')), cv2.COLOR_RGB2BGR)

            if do_glitch and frame_idx % 2 == 0:
                frame = apply_glitch(frame)
        else:
            prev_block_id = None

        out.write(frame)
        frame_idx += 1

        pct = int(frame_idx / max(total_frames, 1) * 20)
        if pct != prev_pct:
            bar = '█' * pct + '░' * (20 - pct)
            print(f"  [{bar}] {frame_idx}/{total_frames}", end='\r')
            prev_pct = pct

    cap.release()
    out.release()
    print(f"\n  ✓ Frames done")

    result = subprocess.run([
        'ffmpeg', '-y',
        '-i', temp_video, '-i', video_path,
        '-map', '0:v', '-map', '1:a',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
        '-c:a', 'copy', output_path
    ], capture_output=True)

    if os.path.exists(temp_video):
        os.remove(temp_video)

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
        return [os.path.join(directory, f) for f in os.listdir(directory)
                if f.lower().endswith(('.mp4', '.mov', '.avi'))]

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
        return top or list(self.broll_dirs.values())[:3]

    def create_segment_plan(self, duration: float, beats: list, top_categories: list) -> list:
        segments = []
        used     = set()
        cat_map  = {
            'space':   'space_vids',   'ancient': 'ancient_ruins_vids',
            'cosmic':  'cosmic_vids',  'sky':     'dark_sky_vids',
            'temple':  'temple_vids',  'any':     top_categories[0] if top_categories else 'space_vids',
        }
        base_dur = 4.0
        n_segs   = max(int(duration / base_dur), 1)

        for i in range(n_segs):
            if i < len(beats):
                cat     = cat_map.get(beats[i].get('broll_category', 'any'),
                                      top_categories[i % len(top_categories)])
                seg_dur = float(beats[i].get('clip_duration', base_dur))
            else:
                cat     = top_categories[i % len(top_categories)]
                seg_dur = base_dur + random.uniform(-1.0, 1.0)

            files = self.get_all_files_from_dir(cat)
            if not files:
                for fb in top_categories:
                    files = self.get_all_files_from_dir(fb)
                    if files:
                        cat = fb
                        break

            if not files:
                continue

            available = [f for f in files if f not in used]
            if not available:
                available = files
                used.clear()

            chosen = random.choice(available)
            used.add(chosen)

            if chosen.lower().endswith(('.mp4', '.mov', '.avi')):
                segments.append({'type': 'broll', 'category': cat,
                                  'file': chosen, 'duration': seg_dur})

        if not segments:
            raise Exception("No segments. Check broll folders.")

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

        vf += ["eq=brightness=0.02:contrast=1.05:saturation=1.1", "format=yuv420p"]
        cmd += ['-vf', ','.join(vf), '-c:v', 'libx264', '-preset', 'ultrafast',
                '-crf', '23', '-pix_fmt', 'yuv420p', '-an', output_file]

        if progress_callback:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    universal_newlines=True, bufsize=1)
            total_f  = int(dur * fps)
            last_f   = 0
            for line in proc.stderr:
                if 'frame=' in line:
                    try:
                        for part in line.split():
                            if part.startswith('frame='):
                                cf = int(part.split('=')[1])
                                if cf > last_f:
                                    last_f = cf
                                    progress_callback(cf, total_f)
                                break
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

        full_text = transcription.get('text', '').strip()
        all_words = []
        for seg in transcription.get('segments', []):
            for w in seg.get('words', []):
                word = w.get('word', '').strip()
                if word:
                    all_words.append({
                        'word':  word,
                        'start': float(w.get('start', 0)),
                        'end':   float(w.get('end', 0)),
                    })
        print(f"  ✅ {len(full_text)} chars, {len(all_words)} words")

        print(f"\n[STEP 3] GPT Call 1: Story Beats...")
        try:
            topic_hint   = list(self.broll_dirs.keys())[0] if self.broll_dirs else "space"
            beats_result = analyze_story_beats(full_text, topic_hint, duration)
            topic        = beats_result.get('topic', 'default')
            beats        = beats_result.get('beats', [])
        except Exception as e:
            raise Exception(f"STEP 3 FAILED: {e}")

        print(f"\n[STEP 4] GPT Call 2: Semantic Highlights...")
        try:
            highlights = generate_render_decisions(beats, topic)
        except Exception as e:
            raise Exception(f"STEP 4 FAILED: {e}")

        print(f"\n[STEP 5] Building phrase blocks...")
        phrase_blocks = build_phrase_blocks(all_words, highlights, beats)

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
                    '-ar', '48000', '-ac', '2', '-shortest', audio_output]
            r = subprocess.run(cmd, capture_output=True)
            if r.returncode != 0:
                raise Exception(f"Audio failed: {r.stderr.decode()[-200:]}")
            print(f"  ✅ Mixed")

            print(f"\n[STEP 12] OpenCV text render...")
            try:
                render_text_overlay_opencv(audio_output, phrase_blocks, beats, all_words, self.output_path)
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
            print(f"🎭 {len(beats)} beats | 📝 {len(phrase_blocks)} phrase blocks")
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
            for tmp in glob.glob("*TEMP_MPY*.mp4") + glob.glob("*_noaudio_tmp.mp4"):
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
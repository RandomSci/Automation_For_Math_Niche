# ============================================================
# VAULTS OF HISTORY - AI Video Generator
# GPT-4o powered editing decisions + MoviePy text rendering
# FFmpeg for video assembly + audio mixing
# ============================================================

import subprocess
import os
import json
import random
import requests
import traceback
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# ============================================================
# APP SETUP
# ============================================================
app = FastAPI(title="Vaults of History Generator")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

current_job = {
    "status": "idle",
    "progress": 0,
    "output": None,
    "error": None,
    "started_at": None
}

OUTPUT_WIDTH  = 1920
OUTPUT_HEIGHT = 1080
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

if not OPENAI_API_KEY:
    print("⚠  WARNING: OPENAI_API_KEY not set. GPT analysis will fail.")

# ============================================================
# CONSTANTS
# ============================================================
MUSIC_MAP = {
    "space":    "bg_musics/space_ambient.mp3",
    "death":    "bg_musics/dark_ambient.mp3",
    "ancient":  "bg_musics/ancient_ambient.mp3",
    "religion": "bg_musics/sacred_ambient.mp3",
    "human":    "bg_musics/human_ambient.mp3",
    "default":  "bg_musics/vaults_ambient.mp3",
}

# Fallback font paths -- tries multiple locations
FONT_BOLD_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
]
FONT_REGULAR_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
]

def find_font(candidates: list) -> str | None:
    for path in candidates:
        if os.path.exists(path):
            print(f"  ✓ Font found: {path}")
            return path
    print(f"  ⚠ No font found from candidates, MoviePy will use default")
    return None

FONT_BOLD    = find_font(FONT_BOLD_CANDIDATES)
FONT_REGULAR = find_font(FONT_REGULAR_CANDIDATES)


# ============================================================
# GPT-4o TRANSCRIPT ANALYZER
# ============================================================
def analyze_transcript_with_gpt(transcript_text: str, topic_hint: str) -> dict:
    if not OPENAI_API_KEY:
        raise Exception("OPENAI_API_KEY is not set. Cannot run GPT analysis.")

    print(f"  🤖 Sending transcript to GPT-4o ({len(transcript_text)} chars)...")
    client = OpenAI(api_key=OPENAI_API_KEY)

    system_prompt = """You are a world-class viral video editor specializing in mind-blowing fact content.
You edit exactly like the TikTok account @sackfeels -- dramatic, emotional, cinematic.

Your task: read a transcript and make PRECISE editing decisions for EVERY chunk of text.

For each segment decide:
- style: "single_word" | "phrase" | "sentence" | "none"
  * single_word = one shocking word, huge, dead center screen
  * phrase = 2-4 key words, center screen, large
  * sentence = full thought, bottom of screen, white with last word highlighted
  * none = intentional silence for dramatic effect (use sparingly but powerfully)
- text: the exact text to display (must match words from transcript)
- color: hex color -- vary dramatically, never repeat same color twice in a row
  * Use: #FFFFFF #FFD700 #FF6B35 #C8A8FF #A8D8FF #FF4444 #A8FFD8
- highlight_color: hex color for last word on sentences, null for others
- font_size: 130 for single_word, 95 for phrase, 68 for sentence
- position: "center" for single_word/phrase, "bottom" for sentence, "top" for rare emphasis
- duration_override: null for natural timing, float seconds for dramatic holds

CRITICAL RULES:
1. Cover the ENTIRE transcript -- every word must appear somewhere
2. Vary styles constantly -- no two same styles in a row
3. "none" pauses create tension -- use before shocking reveals
4. Single words are MONEY SHOTS -- only for the most jaw-dropping words
5. Colors must feel intentional -- darker emotions get warmer colors, cosmic gets blues/purples
6. Think like a human editor who has watched 10000 viral videos

Return ONLY valid JSON:
{
  "topic": "space|death|ancient|religion|human|default",
  "music_mood": "eerie|dark|mysterious|sacred|cosmic|haunting",
  "segments": [
    {
      "text": "exact text",
      "style": "single_word|phrase|sentence|none",
      "color": "#HEXCOLOR",
      "highlight_color": "#HEXCOLOR or null",
      "font_size": 130,
      "position": "center|bottom|top",
      "duration_override": null
    }
  ]
}"""

    user_prompt = (
        f"Topic hint: {topic_hint}\n\n"
        f"Full transcript:\n{transcript_text}\n\n"
        f"Make dramatic editing decisions. Cover EVERY word. Vary everything. Make jaws drop."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.85,
            timeout=60,
        )
        raw = response.choices[0].message.content
        result = json.loads(raw)

        topic    = result.get('topic', 'default')
        mood     = result.get('music_mood', 'eerie')
        segments = result.get('segments', [])

        print(f"  ✅ GPT done. Topic={topic} Mood={mood} Segments={len(segments)}")
        return result

    except json.JSONDecodeError as e:
        print(f"  ❌ GPT returned invalid JSON: {e}")
        raise Exception(f"GPT JSON parse error: {e}")
    except Exception as e:
        print(f"  ❌ GPT API error: {e}")
        raise Exception(f"GPT API failed: {e}")


# ============================================================
# MOVIEPY TEXT RENDERER
# ============================================================
def render_text_overlay(video_path: str, segments: list, word_timestamps: list, output_path: str):
    print(f"🎨 MoviePy: rendering {len(segments)} text segments...")

    # Import MoviePy -- handle both v1 and v2 API
    try:
        from moviepy import VideoFileClip, TextClip, CompositeVideoClip
        MOVIEPY_V2 = True
        print(f"  ✓ MoviePy v2 loaded")
    except ImportError:
        try:
            from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
            MOVIEPY_V2 = False
            print(f"  ✓ MoviePy v1 loaded")
        except ImportError as e:
            print(f"  ❌ MoviePy not available: {e}")
            raise Exception(f"MoviePy import failed: {e}")

    if not os.path.exists(video_path):
        raise Exception(f"Video input not found: {video_path}")

    try:
        video    = VideoFileClip(video_path)
        duration = video.duration
        print(f"  ✓ Video loaded: {duration:.2f}s {video.size}")
    except Exception as e:
        raise Exception(f"Failed to load video {video_path}: {e}")

    text_clips     = []
    matched        = 0
    skipped        = 0
    error_count    = 0

    # Build word lookup for fast matching
    word_lookup = {}
    for i, wt in enumerate(word_timestamps):
        clean = wt['word'].upper().strip('.,!?;:\'"()[]')
        if clean not in word_lookup:
            word_lookup[clean] = []
        word_lookup[clean].append(i)

    used_word_indices = set()

    for seg_idx, seg in enumerate(segments):
        style        = seg.get("style", "sentence")
        text         = seg.get("text", "").strip()
        color        = seg.get("color", "#FFFFFF")
        hi_color     = seg.get("highlight_color")
        fontsize     = int(seg.get("font_size") or 68)
        position     = seg.get("position", "bottom")
        dur_override = seg.get("duration_override")

        if style == "none" or not text:
            continue

        # Find timing by matching first word of segment
        text_words  = text.upper().split()
        first_clean = text_words[0].strip('.,!?;:\'"()[]')

        start_time = None
        end_time   = None

        candidates = word_lookup.get(first_clean, [])
        for idx in candidates:
            if idx in used_word_indices:
                continue
            start_time = word_timestamps[idx]['start']
            end_idx    = min(idx + len(text_words) - 1, len(word_timestamps) - 1)
            end_time   = word_timestamps[end_idx]['end']
            # Mark these words used
            for u in range(idx, end_idx + 1):
                used_word_indices.add(u)
            break

        if start_time is None:
            print(f"  ⚠ [{seg_idx}] No timestamp match for: '{text[:30]}' -- skipping")
            skipped += 1
            continue

        # Calculate clip duration
        natural_dur  = end_time - start_time
        clip_duration = float(dur_override) if dur_override else natural_dur
        clip_duration = max(clip_duration, 0.25)

        # Clamp to video duration
        if start_time >= duration:
            print(f"  ⚠ [{seg_idx}] start_time {start_time:.2f} >= duration {duration:.2f} -- skipping")
            skipped += 1
            continue
        clip_duration = min(clip_duration, duration - start_time)

        # Select font
        font = FONT_BOLD if style in ["single_word", "phrase"] else FONT_REGULAR

        # Compute position
        if position == "center":
            x_jitter = random.randint(-60, 60) if style in ["single_word", "phrase"] else 0
            y_jitter = random.randint(-30, 30) if style in ["single_word", "phrase"] else 0
            pos = ("center", OUTPUT_HEIGHT // 2 - fontsize // 2 + y_jitter)
        elif position == "bottom":
            pos = ("center", OUTPUT_HEIGHT - 160)
        elif position == "top":
            pos = ("center", 80)
        else:
            pos = ("center", "center")

        # Convert hex color to MoviePy color name or tuple
        def hex_to_rgb(hex_str):
            hex_str = hex_str.lstrip('#')
            try:
                return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
            except:
                return (255, 255, 255)

        color_rgb    = hex_to_rgb(color)
        hi_color_rgb = hex_to_rgb(hi_color) if hi_color else None

        # Build text clip(s)
        display_text = text.upper() if style in ["single_word", "phrase"] else text.upper()

        try:
            # Sentence with last word highlighted
            if style == "sentence" and hi_color_rgb and len(text_words) > 1:
                white_words = ' '.join(text_words[:-1])
                last_word   = text_words[-1]

                def make_clip(txt, clr, fsz, fnt):
                    kwargs = dict(
                        text         = txt,
                        font_size    = fsz,
                        color        = clr,
                        stroke_color = (0, 0, 0),
                        stroke_width = 2,
                    )
                    if fnt:
                        kwargs['font'] = fnt
                    clip = TextClip(**kwargs)
                    if MOVIEPY_V2:
                        return clip.with_start(start_time).with_duration(clip_duration)
                    else:
                        return clip.set_start(start_time).set_duration(clip_duration)

                wclip = make_clip(white_words, color_rgb,    fontsize, font)
                yclip = make_clip(last_word,   hi_color_rgb, fontsize, font)

                ww = wclip.size[0]
                yw = yclip.size[0]
                total_w = ww + 16 + yw
                x0      = max(0, (OUTPUT_WIDTH - total_w) // 2)
                y0      = OUTPUT_HEIGHT - 160

                if MOVIEPY_V2:
                    wclip = wclip.with_position((x0, y0))
                    yclip = yclip.with_position((x0 + ww + 16, y0))
                else:
                    wclip = wclip.set_position((x0, y0))
                    yclip = yclip.set_position((x0 + ww + 16, y0))

                text_clips.extend([wclip, yclip])
                matched += 1
                continue

            # Single text clip
            kwargs = dict(
                text         = display_text,
                font_size    = fontsize,
                color        = color_rgb,
                stroke_color = (0, 0, 0),
                stroke_width = 3 if style == "single_word" else 2,
            )
            if font:
                kwargs['font'] = font

            clip = TextClip(**kwargs)

            if MOVIEPY_V2:
                clip = clip.with_start(start_time).with_duration(clip_duration).with_position(pos)
            else:
                clip = clip.set_start(start_time).set_duration(clip_duration).set_position(pos)

            text_clips.append(clip)
            matched += 1

        except Exception as e:
            print(f"  ⚠ [{seg_idx}] TextClip error for '{text[:25]}': {e}")
            error_count += 1
            continue

    print(f"  📊 Text clips: matched={matched} skipped={skipped} errors={error_count}")

    if not text_clips:
        print(f"  ⚠ No text clips created -- saving video without text overlay")
        try:
            video.write_videofile(output_path, codec='libx264', audio_codec='aac',
                                  fps=30, logger=None)
        finally:
            video.close()
        return

    print(f"  🎬 Compositing {len(text_clips)} clips onto video...")
    try:
        final = CompositeVideoClip([video] + text_clips, size=(OUTPUT_WIDTH, OUTPUT_HEIGHT))
        final.write_videofile(
            output_path,
            codec       = 'libx264',
            audio_codec = 'aac',
            fps         = 30,
            logger      = None,
            threads     = 4,
        )
        print(f"  ✅ Text overlay complete: {output_path}")
    except Exception as e:
        print(f"  ❌ Composite write failed: {e}")
        raise
    finally:
        try:
            video.close()
        except:
            pass
        try:
            final.close()
        except:
            pass


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
                'space':   ['universe', 'galaxy', 'black hole', 'star', 'planet', 'cosmos', 'light year', 'nasa', 'orbit'],
                'ancient': ['ancient', 'civilization', 'pyramid', 'ruins', 'lost', 'buried', 'forgotten', 'artifact'],
                'cosmic':  ['time', 'reality', 'dimension', 'quantum', 'existence', 'consciousness', 'infinity'],
                'sky':     ['sky', 'atmosphere', 'cloud', 'above', 'beyond', 'vast', 'endless', 'horizon', 'void'],
                'temple':  ['religion', 'god', 'sacred', 'ritual', 'belief', 'worship', 'divine', 'spiritual'],
            }

    # ----------------------------------------------------------
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
            data   = json.loads(result.stdout)
            width  = data['streams'][0]['width']
            height = data['streams'][0]['height']
            return width, height, width / height
        except Exception as e:
            print(f"  ⚠ get_video_info failed for {filepath}: {e}")
            return None, None, None

    def get_all_files_from_dir(self, directory: str) -> list:
        if not os.path.exists(directory):
            return []
        valid = ('.mp4', '.mov', '.avi')
        files = [os.path.join(directory, f)
                 for f in os.listdir(directory)
                 if f.lower().endswith(valid)]
        return files

    def is_video(self, filepath: str) -> bool:
        return filepath.lower().endswith(('.mp4', '.mov', '.avi'))

    # ----------------------------------------------------------
    def transcribe_with_whisper(self, model: str = "base") -> dict | None:
        cache_file = f"{os.path.splitext(self.audio_path)[0]}_transcription.json"

        if os.path.exists(cache_file):
            print(f"  ✅ Using cached transcription: {cache_file}")
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"  ⚠ Cache read failed: {e} -- re-transcribing")

        try:
            import whisper
            if not hasattr(whisper, 'load_model'):
                raise ImportError("Wrong whisper package installed. Run: pip install openai-whisper")
            print(f"  🎤 Transcribing with Whisper model={model}...")
            wm     = whisper.load_model(model)
            result = wm.transcribe(self.audio_path, word_timestamps=True, language="en")
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
            print(f"  💾 Transcription cached: {cache_file}")
            return result
        except ImportError as e:
            print(f"  ❌ Whisper import error: {e}")
            return None
        except Exception as e:
            print(f"  ❌ Whisper transcription error: {e}")
            traceback.print_exc()
            return None

    # ----------------------------------------------------------
    def match_broll_categories(self, full_text: str) -> list:
        text   = full_text.lower()
        scores = {cat: sum(text.count(k) for k in kws)
                  for cat, kws in self.keyword_map.items()}
        sorted_cats = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        detected    = [c for c, s in sorted_cats if s > 0]

        folder_map = {k: v for k, v in self.broll_dirs.items()}
        top = [self.broll_dirs[d] for d in detected if d in self.broll_dirs]

        if not top:
            top = list(self.broll_dirs.values())[:3]

        return top

    # ----------------------------------------------------------
    def create_segment_plan(self, duration: float, top_categories: list) -> list:
        segments  = []
        base_dur  = 4.0
        n_segs    = int(duration / base_dur)
        used      = set()

        for i in range(n_segs):
            cat   = top_categories[i % len(top_categories)]
            files = self.get_all_files_from_dir(cat)

            if not files:
                print(f"  ⚠ No files in folder: {cat} -- skipping")
                continue

            available = [f for f in files if f not in used]
            if not available:
                available = files
                used.clear()

            seg_dur = base_dur + random.uniform(-1.0, 1.0)
            chosen  = random.choice(available)
            used.add(chosen)

            if self.is_video(chosen):
                segments.append({
                    'type':     'broll',
                    'category': cat,
                    'file':     chosen,
                    'duration': seg_dur,
                })

        if not segments:
            raise Exception("No video segments could be created. Check your broll folders have video files.")

        total = sum(s['duration'] for s in segments)
        if total < duration:
            segments[-1]['duration'] += (duration - total)

        return segments

    # ----------------------------------------------------------
    def process_segment_to_file(self, segment: dict, output_file: str,
                                fps: int = 30, progress_callback=None) -> str:
        seg_duration = segment['duration']
        width, height, aspect = self.get_video_info(segment['file'])

        cmd = ['ffmpeg', '-y', '-progress', 'pipe:1', '-nostats',
               '-i', segment['file'], '-t', str(seg_duration)]

        vf = []
        if aspect and aspect < (OUTPUT_WIDTH / OUTPUT_HEIGHT):
            vf.append(f"scale={OUTPUT_WIDTH}:-2:force_original_aspect_ratio=decrease")
            vf.append(f"pad={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:(ow-iw)/2:(oh-ih)/2:black")
        else:
            vf.append(f"scale=-2:{OUTPUT_HEIGHT}:force_original_aspect_ratio=increase")
            vf.append(f"crop={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}")

        vf.append("eq=brightness=0.02:contrast=1.05:saturation=1.1")
        vf.append("format=yuv420p")

        cmd += ['-vf', ','.join(vf),
                '-c:v', 'libx264', '-preset', 'ultrafast',
                '-crf', '23', '-pix_fmt', 'yuv420p', '-an', output_file]

        if progress_callback:
            proc         = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                            universal_newlines=True, bufsize=1)
            total_frames = int(seg_duration * fps)
            last_frame   = 0
            for line in proc.stderr:
                if 'frame=' in line:
                    try:
                        for part in line.split():
                            if part.startswith('frame='):
                                cf = int(part.split('=')[1])
                                if cf > last_frame:
                                    last_frame = cf
                                    progress_callback(cf, total_frames)
                                break
                    except:
                        pass
            proc.wait()
            if proc.returncode != 0:
                raise subprocess.CalledProcessError(proc.returncode, cmd)
        else:
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, cmd,
                                                    stderr=result.stderr)
        return output_file

    # ----------------------------------------------------------
    def _add_cta_overlay(self, video_input: str, output_path: str, duration: float):
        # Only show "Vaults of History" watermark at the END of the video
        # NO start text -- it interrupts the content
        end_text = "Vaults of History"
        end_time = round(max(duration - 4, 1), 3)

        vf = (
            f"drawtext=text='{end_text}'"
            f":fontcolor=yellow:fontsize=42:font=Arial"
            f":borderw=2:bordercolor=black:shadowx=2:shadowy=2"
            f":x=(w-text_w)/2:y=h*0.91:enable='gt(t\\,{end_time})'"
        )
        result = subprocess.run(
            ['ffmpeg', '-y', '-i', video_input, '-vf', vf, '-c:a', 'copy', output_path],
            capture_output=True
        )
        if result.returncode != 0:
            print(f"  ⚠ CTA overlay failed: {result.stderr.decode()[-300:]}")
            print(f"  ⚠ Copying without CTA overlay")
            subprocess.run(['ffmpeg', '-y', '-i', video_input, '-c', 'copy', output_path],
                           check=True, capture_output=True)
        else:
            print(f"  ✨ CTA overlay added")

    # ----------------------------------------------------------
    def create_vaults_video(self, bg_volume: float = 0.12, fps: int = 30) -> bool:
        import time
        t0 = time.time()

        print(f"\n{'='*70}")
        print(f"🏛  VAULTS OF HISTORY PIPELINE STARTING")
        print(f"{'='*70}")

        # STEP 1 -- Audio duration
        try:
            duration = self.get_audio_duration()
            print(f"⏱  Audio duration: {duration:.2f}s")
        except Exception as e:
            print(f"❌ STEP 1 FAILED (get_audio_duration): {e}")
            raise

        # STEP 2 -- Transcribe
        print(f"\n[STEP 2] Transcribing audio...")
        transcription = self.transcribe_with_whisper()
        if not transcription:
            raise Exception("STEP 2 FAILED: Whisper transcription returned None")

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

        print(f"  ✅ Transcript: {len(full_text)} chars, {len(all_words)} words")

        # STEP 3 -- GPT edit decisions
        print(f"\n[STEP 3] GPT-4o analyzing transcript...")
        try:
            topic_hint   = list(self.broll_dirs.keys())[0] if self.broll_dirs else "default"
            gpt_result   = analyze_transcript_with_gpt(full_text, topic_hint)
            topic        = gpt_result.get('topic', 'default')
            segments_gpt = gpt_result.get('segments', [])
            print(f"  ✅ {len(segments_gpt)} edit decisions received")
        except Exception as e:
            print(f"  ❌ STEP 3 FAILED (GPT): {e}")
            raise

        # STEP 4 -- Music selection
        print(f"\n[STEP 4] Selecting background music...")
        bg_music = MUSIC_MAP.get(topic, MUSIC_MAP['default'])
        if not os.path.exists(bg_music):
            print(f"  ⚠ Topic music not found: {bg_music}")
            bg_music = MUSIC_MAP['default']
            if not os.path.exists(bg_music):
                print(f"  ⚠ Default music not found: {bg_music}")
                # Try any mp3 in bg_musics/
                for fname in os.listdir('bg_musics') if os.path.exists('bg_musics') else []:
                    if fname.endswith('.mp3'):
                        bg_music = os.path.join('bg_musics', fname)
                        print(f"  ✓ Using fallback music: {bg_music}")
                        break
                else:
                    bg_music = None
                    print(f"  ⚠ No music found -- continuing without background music")
        print(f"  🎵 Music: {bg_music}")

        # STEP 5 -- B-roll matching
        print(f"\n[STEP 5] Matching B-roll folders...")
        top_categories = self.match_broll_categories(full_text)
        print(f"  📊 Folders: {top_categories}")

        # STEP 6 -- Segment plan
        print(f"\n[STEP 6] Creating segment plan...")
        try:
            video_segments = self.create_segment_plan(duration, top_categories)
            print(f"  ✅ {len(video_segments)} segments planned")
        except Exception as e:
            print(f"  ❌ STEP 6 FAILED: {e}")
            raise

        # STEP 7 -- Process segments
        print(f"\n[STEP 7] Processing video segments...")
        temp_files    = []
        concat_list   = "concat_list.txt"
        concat_output = "concatenated_video.mp4"
        audio_output  = "audio_mixed.mp4"

        try:
            try:
                from tqdm import tqdm
                use_tqdm = True
            except ImportError:
                use_tqdm = False

            for i, seg in enumerate(video_segments):
                temp_file = f"temp_segment_{i:02d}.mp4"
                t_seg     = time.time()

                if use_tqdm:
                    total_frames = int(seg['duration'] * fps)
                    pbar = tqdm(total=total_frames,
                                desc=f"  Seg {i+1}/{len(video_segments)}: {os.path.basename(seg['file'])[:28]}",
                                unit='frame')
                    def upd(cur, tot, pb=pbar):
                        pb.n = min(cur, tot)
                        pb.refresh()
                    try:
                        self.process_segment_to_file(seg, temp_file, fps, progress_callback=upd)
                    except Exception as e:
                        print(f"\n  ❌ Segment {i+1} failed: {e}")
                        raise
                    finally:
                        pbar.n = pbar.total
                        pbar.refresh()
                        pbar.close()
                        print(f"    ✓ {time.time()-t_seg:.1f}s")
                else:
                    print(f"  Seg {i+1}/{len(video_segments)}: {os.path.basename(seg['file'])}", end='', flush=True)
                    self.process_segment_to_file(seg, temp_file, fps)
                    print(f" ✓ ({time.time()-t_seg:.1f}s)")

                temp_files.append(temp_file)

            # STEP 8 -- Concatenate
            print(f"\n[STEP 8] Concatenating segments...")
            with open(concat_list, 'w') as f:
                for tf in temp_files:
                    f.write(f"file '{tf}'\n")

            result = subprocess.run(
                ['ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                 '-i', concat_list, '-c', 'copy', concat_output],
                capture_output=True
            )
            if result.returncode != 0:
                raise Exception(f"Concatenation failed: {result.stderr.decode()[-500:]}")
            print(f"  ✅ Concatenation complete")

            # STEP 9 -- Audio mix
            print(f"\n[STEP 9] Mixing audio...")
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

            result = subprocess.run(cmd, capture_output=True)
            if result.returncode != 0:
                raise Exception(f"Audio mix failed: {result.stderr.decode()[-500:]}")
            print(f"  ✅ Audio mixed")

            # STEP 10 -- MoviePy text overlay
            print(f"\n[STEP 10] MoviePy dynamic text overlay...")
            try:
                render_text_overlay(audio_output, segments_gpt, all_words, self.output_path)
            except Exception as e:
                print(f"  ❌ Text overlay failed: {e}")
                print(f"  ⚠ Falling back -- copying video without text overlay")
                traceback.print_exc()
                subprocess.run(['ffmpeg', '-y', '-i', audio_output, '-c', 'copy', self.output_path],
                               check=True, capture_output=True)

            if os.path.exists(audio_output):
                os.remove(audio_output)

            # STEP 11 -- CTA overlay
            print(f"\n[STEP 11] Adding CTA overlay...")
            cta_output = self.output_path.replace(".mp4", "_cta.mp4")
            self._add_cta_overlay(self.output_path, cta_output, duration)
            self.output_path = cta_output

            # Final check
            if not os.path.exists(self.output_path):
                raise Exception(f"Final output not found: {self.output_path}")

            file_size = os.path.getsize(self.output_path) / (1024 * 1024)
            if file_size < 1.0:
                raise Exception(f"Output file suspiciously small: {file_size:.2f} MB")

            total_time = time.time() - t0
            print(f"\n{'='*70}")
            print(f"✅ VAULTS OF HISTORY VIDEO COMPLETE!")
            print(f"{'='*70}")
            print(f"📁 Output   : {self.output_path}")
            print(f"💾 Size     : {file_size:.2f} MB")
            print(f"⏱  Duration : {duration:.2f}s")
            print(f"⚡ Total    : {total_time:.1f}s ({total_time/60:.1f} min)")
            print(f"📐 Format   : {OUTPUT_WIDTH}x{OUTPUT_HEIGHT} 16:9")
            print(f"🎬 Segments : {len(video_segments)}")
            print(f"📝 Text segs: {len(segments_gpt)}")
            print(f"{'='*70}\n")
            return True

        except Exception as e:
            print(f"\n❌ Pipeline error: {e}")
            traceback.print_exc()
            return False

        finally:
            print(f"\n🧹 Cleaning up temp files...")
            for tf in temp_files:
                if os.path.exists(tf):
                    try:
                        os.remove(tf)
                    except:
                        pass
            for f in [concat_list, concat_output]:
                if os.path.exists(f):
                    try:
                        os.remove(f)
                    except:
                        pass


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
            'space':   ['universe', 'galaxy', 'black hole', 'star', 'planet', 'cosmos', 'light year', 'nasa', 'orbit'],
            'ancient': ['ancient', 'civilization', 'pyramid', 'ruins', 'lost', 'buried', 'forgotten', 'artifact'],
            'cosmic':  ['time', 'reality', 'dimension', 'quantum', 'existence', 'consciousness', 'infinity'],
            'sky':     ['sky', 'atmosphere', 'cloud', 'above', 'beyond', 'vast', 'endless', 'horizon', 'void'],
            'temple':  ['religion', 'god', 'sacred', 'ritual', 'belief', 'worship', 'divine', 'spiritual'],
        }
    },
    'love': {
        'broll_dirs': {
            'couple':  'couple_romantic_vids',
            'candle':  'romantic_candle_vids',
            'flowers': 'romantic_flowers_vids',
        },
        'keyword_map': {
            'couple':  ['love', 'together', 'relationship', 'partner', 'couple', 'romance'],
            'candle':  ['intimate', 'warm', 'cozy', 'soft', 'gentle', 'candlelight'],
            'flowers': ['beauty', 'gift', 'surprise', 'flowers', 'bloom', 'garden'],
        }
    },
}


# ============================================================
# FASTAPI
# ============================================================
@app.get("/")
def root():
    return {
        "service": "Vaults of History Generator",
        "status":  "running",
        "openai_key_set": bool(OPENAI_API_KEY),
    }


@app.post("/generate")
async def generate_video_api(background_tasks: BackgroundTasks, niche: str = "vaults"):
    global current_job
    if current_job["status"] == "processing":
        return {"message": "Already processing", "status": "processing",
                "started_at": current_job["started_at"]}
    current_job = {
        "status":     "processing",
        "progress":   0,
        "output":     None,
        "error":      None,
        "started_at": datetime.now().isoformat(),
        "niche":      niche,
    }
    background_tasks.add_task(process_video, niche)
    return {"message": f"Started niche={niche}", "status": "processing"}


def process_video(niche: str = "vaults"):
    global current_job
    try:
        current_job["progress"] = 5

        audio_url          = "https://raw.githubusercontent.com/RandomSci/Automation_For_Love_Niche/main/Audio_Voice/new_love.mp3"
        audio_file         = "Audio_Voice/vaults_narration.mp3"
        output_file        = "vaults_output.mp4"
        transcription_file = f"{os.path.splitext(audio_file)[0]}_transcription.json"

        # Download audio
        print(f"\n📥 Downloading audio from GitHub...")
        os.makedirs("Audio_Voice", exist_ok=True)
        try:
            resp = requests.get(audio_url, timeout=30)
            if resp.status_code != 200:
                raise Exception(f"HTTP {resp.status_code}")
            with open(audio_file, "wb") as f:
                f.write(resp.content)
            print(f"  ✅ Audio saved: {audio_file} ({len(resp.content)//1024}KB)")
        except Exception as e:
            raise Exception(f"Audio download failed: {e}")

        current_job["progress"] = 15

        # Clean old outputs
        for old in [output_file, output_file.replace(".mp4", "_cta.mp4"),
                    "audio_mixed.mp4", transcription_file]:
            if os.path.exists(old):
                os.remove(old)
                print(f"  🗑 Removed: {old}")

        current_job["progress"] = 20

        niche_config = NICHE_TEMPLATES.get(niche, NICHE_TEMPLATES['vaults'])
        gen = VaultsGenerator(
            audio_path   = audio_file,
            output_path  = output_file,
            niche_config = niche_config,
        )

        current_job["progress"] = 25
        success = gen.create_vaults_video(bg_volume=0.12, fps=30)
        current_job["progress"] = 95

        final_output = output_file.replace(".mp4", "_cta.mp4")

        if success and os.path.exists(final_output):
            current_job["status"]   = "completed"
            current_job["progress"] = 100
            current_job["output"]   = final_output
            print(f"\n🎉 JOB COMPLETE: {final_output}")
        else:
            raise Exception("Pipeline returned False or output file missing")

    except Exception as e:
        err_msg = str(e)
        current_job["status"]   = "error"
        current_job["error"]    = err_msg
        current_job["progress"] = 0
        print(f"\n❌ JOB FAILED: {err_msg}")
        traceback.print_exc()


@app.get("/status")
def check_status():
    return {
        "status":      current_job["status"],
        "progress":    current_job["progress"],
        "error":       current_job["error"],
        "started_at":  current_job["started_at"],
        "niche":       current_job.get("niche", "unknown"),
        "ready":       current_job["status"] == "completed",
        "output_file": current_job.get("output"),
    }


@app.get("/download")
def download_video():
    if current_job["status"] != "completed":
        raise HTTPException(status_code=400,
                            detail=f"Not ready. Status: {current_job['status']} Error: {current_job.get('error')}")
    if not current_job["output"] or not os.path.exists(current_job["output"]):
        raise HTTPException(status_code=404, detail=f"Output file not found: {current_job.get('output')}")
    filename = f"vaults_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    return FileResponse(current_job["output"], media_type="video/mp4", filename=filename)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Starting Vaults of History on port {port}")
    print(f"🔑 OpenAI key set: {bool(OPENAI_API_KEY)}")
    uvicorn.run(app, host="0.0.0.0", port=port)
import subprocess
import os
import json
import random
import requests
from datetime import datetime

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Vaults of History Generator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

current_job = {
    "status": "idle",
    "progress": 0,
    "output": None,
    "error": None,
    "started_at": None
}

OUTPUT_WIDTH  = 1920
OUTPUT_HEIGHT = 1080


class VaultsGenerator:
    def __init__(self, audio_path, output_path="output.mp4", niche_config=None):
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
                'ancient': ['ancient', 'civilization', 'pyramid', 'ruins', 'lost', 'buried', 'forgotten', 'artifact', 'archaeology'],
                'cosmic':  ['time', 'reality', 'dimension', 'quantum', 'existence', 'consciousness', 'infinity', 'parallel'],
                'sky':     ['sky', 'atmosphere', 'cloud', 'above', 'beyond', 'vast', 'endless', 'horizon', 'void'],
                'temple':  ['religion', 'god', 'sacred', 'ritual', 'belief', 'worship', 'divine', 'spiritual', 'ancient'],
            }

    def get_audio_duration(self):
        cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            self.audio_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return float(result.stdout.strip())

    def get_video_info(self, filepath):
        cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height',
            '-of', 'json',
            filepath
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        try:
            data   = json.loads(result.stdout)
            width  = data['streams'][0]['width']
            height = data['streams'][0]['height']
            return width, height, width / height
        except:
            return None, None, None

    def get_all_files_from_dir(self, directory):
        if not os.path.exists(directory):
            return []
        valid = ('.mp4', '.mov', '.avi')
        return [os.path.join(directory, f)
                for f in os.listdir(directory)
                if f.lower().endswith(valid)]

    def is_video(self, filepath):
        return filepath.lower().endswith(('.mp4', '.mov', '.avi'))

    def _format_srt_time(self, seconds):
        h  = int(seconds // 3600)
        m  = int((seconds % 3600) // 60)
        s  = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    def generate_subtitles_with_whisper(self, model="base"):
        cache_file = f"{os.path.splitext(self.audio_path)[0]}_transcription.json"

        if os.path.exists(cache_file):
            print(f"✅ Using cached transcription")
            with open(cache_file, 'r', encoding='utf-8') as f:
                result = json.load(f)
        else:
            try:
                import whisper
                print(f"🎤 Transcribing with Whisper ({model})...")
                if not hasattr(whisper, 'load_model'):
                    raise ImportError("Wrong whisper package")
                wm     = whisper.load_model(model)
                result = wm.transcribe(self.audio_path, word_timestamps=True, language="en")
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2)
                print(f"💾 Cached transcription")
            except Exception as e:
                print(f"❌ Whisper error: {e}")
                return None

        srt_path = "subtitles.srt"
        with open(srt_path, 'w', encoding='utf-8') as f:
            counter    = 1
            chunk_size = 2
            for segment in result['segments']:
                words = segment.get('words', [])
                if not words:
                    continue
                for i in range(0, len(words), chunk_size):
                    chunk = words[i:i + chunk_size]
                    if not chunk:
                        continue
                    start = chunk[0]['start']
                    end   = chunk[-1]['end']
                    text  = ' '.join([w['word'].strip() for w in chunk])
                    f.write(f"{counter}\n")
                    f.write(f"{self._format_srt_time(start)} --> {self._format_srt_time(end)}\n")
                    f.write(f"{text.upper()}\n\n")
                    counter += 1

        print(f"✅ Subtitles saved")
        return srt_path

    def analyze_subtitles_for_keywords(self, srt_path):
        if not srt_path or not os.path.exists(srt_path):
            return list(self.broll_dirs.values())[:3]

        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read().lower()

        scores = {}
        for cat, kws in self.keyword_map.items():
            scores[cat] = sum(content.count(k) for k in kws)

        sorted_cats = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        detected    = [c for c, s in sorted_cats if s > 0]

        folder_map = {
            'space':   'space_vids',
            'ancient': 'ancient_ruins_vids',
            'cosmic':  'cosmic_vids',
            'sky':     'dark_sky_vids',
            'temple':  'temple_vids',
        }

        top = []
        for d in detected:
            folder = folder_map.get(d)
            if folder and folder in self.broll_dirs.values():
                top.append(folder)

        if not top:
            top = list(self.broll_dirs.values())[:3]

        return top

    def create_segment_plan(self, duration, top_categories):
        segments              = []
        base_segment_duration = 4.0
        num_segments          = int(duration / base_segment_duration)
        used_files            = set()

        for i in range(num_segments):
            category = top_categories[i % len(top_categories)]
            files    = self.get_all_files_from_dir(category)

            if files:
                available = [f for f in files if f not in used_files]
                if not available:
                    available = files
                    used_files.clear()

                seg_dur       = base_segment_duration + random.uniform(-1.0, 1.0)
                selected_file = random.choice(available)
                used_files.add(selected_file)

                if self.is_video(selected_file):
                    segments.append({
                        'type':     'broll',
                        'category': category,
                        'file':     selected_file,
                        'duration': seg_dur,
                    })

        total = sum(s['duration'] for s in segments)
        if segments and total < duration:
            segments[-1]['duration'] += (duration - total)

        return segments

    def process_segment_to_file(self, segment, output_file, fps=30, progress_callback=None):
        duration = segment['duration']
        width, height, aspect = self.get_video_info(segment['file'])

        cmd = ['ffmpeg', '-y', '-progress', 'pipe:1', '-nostats']
        cmd.extend(['-i', segment['file'], '-t', str(duration)])

        vf_filters = []

        if aspect and aspect < (OUTPUT_WIDTH / OUTPUT_HEIGHT):
            vf_filters.append(f"scale={OUTPUT_WIDTH}:-2:force_original_aspect_ratio=decrease")
            vf_filters.append(f"pad={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:(ow-iw)/2:(oh-ih)/2:black")
        else:
            vf_filters.append(f"scale=-2:{OUTPUT_HEIGHT}:force_original_aspect_ratio=increase")
            vf_filters.append(f"crop={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}")

        vf_filters.append(f"zoompan=z='min(zoom+0.0008,1.05)':d={int(duration*fps)}:s={OUTPUT_WIDTH}x{OUTPUT_HEIGHT}:fps={fps}")
        vf_filters.append("format=yuv420p")

        cmd.extend(['-vf', ','.join(vf_filters)])
        cmd.extend([
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
            '-an',
            output_file
        ])

        if progress_callback:
            process      = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                            universal_newlines=True, bufsize=1)
            total_frames = int(duration * fps)
            last_frame   = 0
            for line in process.stderr:
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
            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd)
        else:
            subprocess.run(cmd, check=True, capture_output=True)

        return output_file

    def _add_cta_overlay(self, video_input, output_path, duration):
        start_texts = [
            "Follow if this broke your brain",
            "Save this and watch again later",
            "Share if you never knew this",
            "Comment what shocked you most",
        ]
        start_text = random.choice(start_texts)
        end_text   = "Vaults of History"
        end_time   = round(duration - 4, 3)

        vf = (
            f"drawtext=text='{start_text}'"
            f":fontcolor=white:fontsize=38:font=Arial"
            f":borderw=3:bordercolor=black:shadowx=2:shadowy=2"
            f":x=(w-text_w)/2:y=h*0.06:enable='lt(t\\,4)',"
            f"drawtext=text='{end_text}'"
            f":fontcolor=yellow:fontsize=42:font=Arial"
            f":borderw=3:bordercolor=black:shadowx=2:shadowy=2"
            f":x=(w-text_w)/2:y=h*0.91:enable='gt(t\\,{end_time})'"
        )

        cmd = ['ffmpeg', '-y', '-i', video_input, '-vf', vf, '-c:a', 'copy', output_path]
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"✨ CTA overlay added!")

    def create_vaults_video(self, auto_generate_subs=True, bg_music=None, bg_volume=0.12, fps=30):
        import time
        overall_start = time.time()

        duration = self.get_audio_duration()
        print(f"\n{'='*70}")
        print(f"🏛  CREATING VAULTS OF HISTORY VIDEO  (16:9)")
        print(f"{'='*70}")
        print(f"⏱  Duration: {duration:.2f}s")

        srt_path = None
        if auto_generate_subs:
            if os.path.exists('subtitles.srt'):
                print(f"✅ Using existing subtitles.srt")
                srt_path = 'subtitles.srt'
            else:
                srt_path = self.generate_subtitles_with_whisper()
                if not srt_path:
                    print(f"⚠  Continuing without subtitles...")

        print(f"\n🧠 Analyzing for keyword matched B-roll...")
        top_categories = self.analyze_subtitles_for_keywords(srt_path)
        print(f"📊 Matched themes: {', '.join(top_categories)}")

        segments = self.create_segment_plan(duration, top_categories)

        print(f"\n📋 SEGMENT PLAN ({len(segments)} clips):")
        for i, seg in enumerate(segments):
            print(f"  {i+1}. {seg['duration']:.1f}s  {seg['category']}  {os.path.basename(seg['file'])}")

        print(f"\n🎬 PASS 1: Processing segments...")
        temp_files    = []
        concat_list   = "concat_list.txt"
        concat_output = "concatenated_video.mp4"

        try:
            try:
                from tqdm import tqdm
                use_tqdm = True
            except ImportError:
                use_tqdm = False

            for i, seg in enumerate(segments):
                temp_file = f"temp_segment_{i:02d}.mp4"
                t0        = time.time()

                if use_tqdm:
                    total_frames = int(seg['duration'] * fps)
                    pbar = tqdm(total=total_frames,
                                desc=f"  Seg {i+1}/{len(segments)}: {os.path.basename(seg['file'])[:30]}",
                                unit='frame')
                    def upd(cur, tot, pb=pbar):
                        pb.n = min(cur, tot)
                        pb.refresh()
                    try:
                        self.process_segment_to_file(seg, temp_file, fps, progress_callback=upd)
                    finally:
                        pbar.n = pbar.total
                        pbar.refresh()
                        pbar.close()
                        print(f"    ✓ {time.time()-t0:.1f}s")
                else:
                    print(f"  Seg {i+1}/{len(segments)}: {os.path.basename(seg['file'])}", end='', flush=True)
                    self.process_segment_to_file(seg, temp_file, fps)
                    print(f" ✓ ({time.time()-t0:.1f}s)")

                temp_files.append(temp_file)

            print(f"\n🎬 PASS 2: Concatenating...")
            with open(concat_list, 'w') as f:
                for tf in temp_files:
                    f.write(f"file '{tf}'\n")
            subprocess.run(['ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                            '-i', concat_list, '-c', 'copy', concat_output],
                           check=True, capture_output=True)
            print(f"  ✓ Done")

            print(f"\n🎬 PASS 3: Adding subtitles, audio, music...")

            cmd = ['ffmpeg', '-y', '-i', concat_output, '-i', self.audio_path]
            if bg_music and os.path.exists(bg_music):
                cmd.extend(['-i', bg_music])
                print(f"  🎵 Background music: {bg_music}")

            vaults_style = (
                "force_style='FontName=Arial,"
                "FontSize=24,"
                "PrimaryColour=&H00FFFFFF,"
                "OutlineColour=&H00000000,"
                "BorderStyle=1,"
                "Outline=3,"
                "Shadow=1,"
                "Bold=1,"
                "MarginV=60,"
                "Alignment=2'"
            )

            if srt_path and os.path.exists(srt_path):
                sub_path = srt_path.replace('\\', '/').replace(':', '\\:')
                vf       = f"subtitles='{sub_path}':{vaults_style}"
                cmd.extend(['-vf', vf])
                print(f"  📝 Vaults subtitle style applied")
            else:
                print(f"  ⚠  No subtitles")

            if bg_music and os.path.exists(bg_music):
                filter_complex = (
                    f'[1:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,volume=1.0[voice];'
                    f'[2:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,volume={bg_volume},aloop=loop=-1:size=2e+09[bg];'
                    f'[voice][bg]amix=inputs=2:duration=first:dropout_transition=2,aresample=48000[aout]'
                )
                cmd.extend(['-filter_complex', filter_complex,
                             '-map', '0:v', '-map', '[aout]'])
            else:
                cmd.extend(['-map', '0:v',
                             '-af', 'aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo',
                             '-map', '1:a'])

            cmd.extend([
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                '-c:a', 'aac',
                '-b:a', '192k',
                '-ar', '48000',
                '-ac', '2',
                '-movflags', '+faststart',
                '-shortest',
                self.output_path
            ])

            subprocess.run(cmd, check=True, capture_output=True)
            print(f"  ✓ Video assembled")

            cta_output = self.output_path.replace(".mp4", "_cta.mp4")
            self._add_cta_overlay(self.output_path, cta_output, duration)
            self.output_path = cta_output

            total_time = time.time() - overall_start
            file_size  = os.path.getsize(self.output_path) / (1024 * 1024)

            if file_size < 2.0:
                print(f"\n⚠  WARNING: Output suspiciously small ({file_size:.2f} MB)")
                return False

            print(f"\n{'='*70}")
            print(f"✅ VAULTS OF HISTORY VIDEO READY!")
            print(f"{'='*70}")
            print(f"📁 Output  : {self.output_path}")
            print(f"💾 Size    : {file_size:.2f} MB")
            print(f"⏱  Duration: {duration:.2f}s")
            print(f"⚡ Time    : {total_time:.1f}s ({total_time/60:.1f} min)")
            print(f"📐 Format  : {OUTPUT_WIDTH}x{OUTPUT_HEIGHT} 16:9")
            print(f"{'='*70}\n")
            return True

        except subprocess.CalledProcessError as e:
            print(f"\n❌ FFmpeg error: {e}")
            if e.stderr:
                print(e.stderr.decode()[-1000:])
            return False

        finally:
            print(f"\n🧹 Cleaning up...")
            for tf in temp_files:
                if os.path.exists(tf):
                    os.remove(tf)
            for f in [concat_list, concat_output]:
                if os.path.exists(f):
                    os.remove(f)


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


@app.get("/")
def root():
    return {
        "service": "Vaults of History Generator",
        "status":  "running",
        "endpoints": {
            "POST /generate": "Generate video (query: niche=vaults)",
            "GET /status":    "Check status",
            "GET /download":  "Download video",
        }
    }


@app.post("/generate")
async def generate_video_api(background_tasks: BackgroundTasks, niche: str = "vaults"):
    global current_job

    if current_job["status"] == "processing":
        return {"message": "Already processing", "status": "processing"}

    current_job = {
        "status":     "processing",
        "progress":   0,
        "output":     None,
        "error":      None,
        "started_at": datetime.now().isoformat(),
        "niche":      niche,
    }

    background_tasks.add_task(process_video, niche)
    return {"message": f"Started for niche: {niche}", "status": "processing"}


def process_video(niche="vaults"):
    global current_job

    try:
        current_job["progress"] = 10

        audio_url          = "https://raw.githubusercontent.com/RandomSci/Automation_For_Love_Niche/main/Audio_Voice/new_love.mp3"
        audio_file         = "Audio_Voice/vaults_narration.mp3"
        output_file        = "vaults_output.mp4"
        transcription_file = "Audio_Voice/vaults_narration_transcription.json"
        bg_music           = "bg_musics/vaults_ambient.mp3"

        print(f"📥 Downloading audio...")
        os.makedirs("Audio_Voice", exist_ok=True)
        response = requests.get(audio_url)
        if response.status_code != 200:
            raise Exception(f"Audio download failed: HTTP {response.status_code}")
        with open(audio_file, "wb") as f:
            f.write(response.content)
        print(f"✅ Audio ready")

        current_job["progress"] = 20

        for old in [output_file, output_file.replace(".mp4", "_cta.mp4"),
                    "subtitles.srt", transcription_file]:
            if os.path.exists(old):
                os.remove(old)
                print(f"🗑 Removed {old}")

        current_job["progress"] = 30

        niche_config = NICHE_TEMPLATES.get(niche, NICHE_TEMPLATES['vaults'])

        gen = VaultsGenerator(
            audio_path   = audio_file,
            output_path  = output_file,
            niche_config = niche_config,
        )

        success = gen.create_vaults_video(
            auto_generate_subs = True,
            bg_music           = bg_music if os.path.exists(bg_music) else None,
            bg_volume          = 0.12,
            fps                = 30,
        )

        current_job["progress"] = 90

        final_output = output_file.replace(".mp4", "_cta.mp4")

        for tmp in ["subtitles.srt", transcription_file]:
            if os.path.exists(tmp):
                os.remove(tmp)

        if success and os.path.exists(final_output):
            current_job["status"]   = "completed"
            current_job["progress"] = 100
            current_job["output"]   = final_output
            print(f"✅ Done: {final_output}")
        else:
            raise Exception("Video generation failed — output not found")

    except Exception as e:
        current_job["status"]   = "error"
        current_job["error"]    = str(e)
        current_job["progress"] = 0
        print(f"❌ Error: {e}")
        import traceback
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
                            detail=f"Not ready. Status: {current_job['status']}")
    if not current_job["output"] or not os.path.exists(current_job["output"]):
        raise HTTPException(status_code=404, detail="File not found")

    filename = f"vaults_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    return FileResponse(current_job["output"], media_type="video/mp4", filename=filename)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
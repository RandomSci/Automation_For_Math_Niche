import subprocess
import os
import json
import random
import tempfile
import requests
from datetime import datetime

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Viral Shorts Generator")

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

class ViralShortsGenerator:
    def __init__(self, main_image, audio_path, output_path="output.mp4", niche_config=None):
        self.main_image = main_image
        self.audio_path = audio_path
        self.output_path = output_path
        
        if niche_config:
            self.broll_dirs = niche_config.get('broll_dirs', {})
            self.keyword_map = niche_config.get('keyword_map', {})
        else:
            self.broll_dirs = {
                'castle': 'medieval_castle_imgs',
                'chess': 'chess_strategy_imgs',
                'book': 'old_book_pages_turning_vids',
                'candle': 'candle_flame_vids',
                'ink': 'ink_writing_vids',
                'storm': 'storm_clouds_time_lapse_vids'
            }
            
            self.keyword_map = {
                'castle': ['power', 'prince', 'war', 'kingdom', 'ruler', 'conquer', 'throne', 'empire'],
                'chess': ['strategy', 'wise', 'think', 'plan', 'move', 'game', 'cunning', 'clever'],
                'book': ['write', 'book', 'knowledge', 'teach', 'learn', 'wisdom', 'read'],
                'ink': ['write', 'author', 'pen', 'letter', 'word', 'text', 'document'],
                'storm': ['chaos', 'turbulent', 'conflict', 'danger', 'dark', 'fear', 'storm'],
                'candle': ['light', 'truth', 'reveal', 'illuminate', 'see', 'darkness', 'flame']
            }
            
    def _add_cta_overlay(self, video_input, output_path, duration, niche="love"):
        filters = []
        
        if niche == "love":
            start_text = random.choice([
                "Double tap if you felt this ❤",
                "This hit deep... double tap ♡",
                "Tag someone who needs this ❤",
                "Save this for later ♡"
            ])
            end_text = "Follow for more love ♡"
        elif niche == "philosophy":
            start_text = "Pause if this made you think"
            end_text = "Follow for daily wisdom"
        else:
            start_text = "Double tap if this hit home"
            end_text = "Follow for more"
            
        filters.append(
            f"drawtext=text='{start_text}':fontcolor=#FFB6FF:fontsize=52:font=Dancing Script:"
            f"borderw=3:bordercolor=#80000000:shadowx=3:shadowy=3:x=(w-text_w)/2:y=h*0.68:enable='lt(t,5)'"
        )
        filters.append(
            f"drawtext=text='{end_text}':fontcolor=white:fontsize=48:font=Dancing Script:"
            f"borderw=3:bordercolor=black:shadowx=2:shadowy=2:x=(w-text_w)/2:y=h*0.75:enable='gt(t,{duration-3})'"
        )
        
        cmd = ['ffmpeg', '-y', '-i', video_input, '-vf', ','.join(filters), '-c:a', 'copy', output_path]
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"✨ {niche.upper()} CTA added!")       
        
    def get_audio_duration(self):
        """Get audio duration in seconds"""
        cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            self.audio_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return float(result.stdout.strip())
    
    def get_video_info(self, filepath):
        """Get video/image dimensions and type"""
        cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height',
            '-of', 'json',
            filepath
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        try:
            data = json.loads(result.stdout)
            width = data['streams'][0]['width']
            height = data['streams'][0]['height']
            aspect_ratio = width / height
            return width, height, aspect_ratio
        except:
            return None, None, None
    
    def get_all_files_from_dir(self, directory):
        """Get all VIDEO files from a directory (no images)"""
        if not os.path.exists(directory):
            return []
        
        valid_extensions = ('.mp4', '.mov', '.avi')
        files = [os.path.join(directory, f) for f in os.listdir(directory) 
                if f.lower().endswith(valid_extensions)]
        return files
    
    def analyze_subtitles_for_keywords(self, srt_path):
        """Analyze subtitles and create timeline with matched categories"""
        if not os.path.exists(srt_path):
            return list(self.broll_dirs.keys())[:3]

        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read().lower()

        category_scores = {}
        for category, keywords in self.keyword_map.items():
            score = sum(content.count(keyword) for keyword in keywords)
            category_scores[category] = score

        sorted_cats = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
        detected = [cat for cat, score in sorted_cats if score > 0]

        keyword_to_folder = {
            'hands': 'holding_hands_imgs',
            'candle': 'romantic_candle_vids',
            'couple': 'couple_romantic_vids',
            'sunset': 'sunset_romantic_vids',
            'flowers': 'romantic_flowers_vids',
            'city': 'date_night_city_vids',
            'book': 'old_book_pages_turning_vids',
            'ink': 'ink_writing_vids',
            'castle': 'medieval_castle_imgs',
            'chess': 'chess_strategy_imgs',
            'storm': 'storm_clouds_time_lapse_vids',
        }

        top_categories = []
        for d in detected:
            folder = keyword_to_folder.get(d)
            if folder and folder in self.broll_dirs:
                top_categories.append(folder)
 
        # fallback: if nothing valid, take first 3 available folders
        if not top_categories:
            top_categories = list(self.broll_dirs.keys())[:3]

        return top_categories


    def create_segment_plan(self, duration, top_categories):
        """Create a plan for video segments - VIDEOS ONLY, NO IMAGES"""
        segments = []
        remaining_time = duration
        base_segment_duration = 5.0
        num_segments = int(remaining_time / base_segment_duration)
        
        used_files = set()  # Track used files to prevent reuse

        for i in range(num_segments):
            category = top_categories[i % len(top_categories)]
            files = self.get_all_files_from_dir(self.broll_dirs.get(category, [])) 
            
            if files:
                # Filter out already used files
                available_files = [f for f in files if f not in used_files]
                
                # If all files in this category have been used, reset and allow reuse
                if not available_files:
                    print(f"  ⚠️  All files in '{category}' used, allowing reuse...")
                    available_files = files
                    used_files.clear()
                
                segment_duration = base_segment_duration + random.uniform(-1.5, 1.5)
                selected_file = random.choice(available_files)
                used_files.add(selected_file)  # Mark this file as used
                
                if self.is_video(selected_file):
                    segments.append({
                        'type': 'broll',
                        'category': category,
                        'file': selected_file,
                        'duration': segment_duration
                    })

        total_duration = sum(s['duration'] for s in segments)
        if segments and total_duration < duration:
            segments[-1]['duration'] += (duration - total_duration)

        return segments
    
    def is_video(self, filepath):
        """Check if file is a video"""
        return filepath.lower().endswith(('.mp4', '.mov', '.avi'))
    
    def process_segment_to_file(self, segment, output_file, fps=30, progress_callback=None):
        """Process a single segment - VIDEOS ONLY"""
        duration = segment['duration']
        width, height, aspect = self.get_video_info(segment['file'])
        
        cmd = ['ffmpeg', '-y', '-progress', 'pipe:1', '-nostats']
        cmd.extend(['-i', segment['file'], '-t', str(duration)])
        
        filters = []
        
        if aspect and aspect < 0.7:  
            filters.append("scale=1080:1920:force_original_aspect_ratio=decrease")
            filters.append("pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black")
        else:  
            filters.append("scale=-2:1920:force_original_aspect_ratio=increase")
            filters.append("crop=1080:1920")
        
        #filters.append(f"fade=t=in:st=0:d=0.3")
        #filters.append(f"fade=t=out:st={duration-0.3}:d=0.3")
        
        filters.append("format=yuv420p")
        
        cmd.extend(['-vf', ','.join(filters)])
        cmd.extend([
            '-c:v', 'libx264',
            '-preset', 'ultrafast',  
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
            '-an',  
            output_file
        ])
        
        if progress_callback:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, bufsize=1)
            total_frames = int(duration * fps)
            last_frame = 0
            
            for line in process.stderr:
                if 'frame=' in line:
                    try:
                        for part in line.split():
                            if part.startswith('frame='):
                                frame_str = part.split('=')[1]
                                current_frame = int(frame_str)
                                if current_frame > last_frame:
                                    last_frame = current_frame
                                    progress_callback(current_frame, total_frames)
                                break
                    except:
                        pass
            
            process.wait()
            if last_frame > 0:
                progress_callback(total_frames, total_frames)
            
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd)
        else:
            subprocess.run(cmd, check=True, capture_output=True)
        
        return output_file
    
    def generate_subtitles_with_whisper(self, model="base"):
        """Generate subtitles using Whisper with caching"""
        cache_file = f"{os.path.splitext(self.audio_path)[0]}_transcription.json"
        
        if os.path.exists(cache_file):
            print(f"✅ Using cached transcription from {cache_file}")
            with open(cache_file, 'r', encoding='utf-8') as f:
                result = json.load(f)
        else:
            try:
                import whisper
                print(f"🎤 Transcribing audio with Whisper ({model} model)...")
                
                if not hasattr(whisper, 'load_model'):
                    raise ImportError("Wrong whisper package installed")
                
                model_whisper = whisper.load_model(model)
                result = model_whisper.transcribe(
                    self.audio_path,
                    word_timestamps=True,
                    language="en"
                )
                
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2)
                print(f"💾 Cached transcription to {cache_file}")
                
            except (ImportError, AttributeError) as e:
                print(f"\n❌ Whisper Error: {e}")
                print("\n" + "="*70)
                print("🔧 WHISPER INSTALLATION ISSUE")
                print("="*70)
                print("\n⚠️  You have the WRONG 'whisper' package!")
                print("\n📝 Fix with:")
                print("  pip uninstall whisper -y")
                print("  pip install openai-whisper")
                print("="*70 + "\n")
                return None
            except Exception as e:
                print(f"\n❌ Error during transcription: {e}")
                return None
        
        srt_path = "subtitles.srt"
        with open(srt_path, 'w', encoding='utf-8') as f:
            counter = 1
            for segment in result['segments']:
                words = segment.get('words', [])
                if not words:
                    continue
                
                chunk_size = 3
                for i in range(0, len(words), chunk_size):
                    chunk = words[i:i+chunk_size]
                    if not chunk:
                        continue
                    
                    start = chunk[0]['start']
                    end = chunk[-1]['end']
                    text = ' '.join([w['word'].strip() for w in chunk])
                    
                    f.write(f"{counter}\n")
                    f.write(f"{self._format_srt_time(start)} --> {self._format_srt_time(end)}\n")
                    f.write(f"{text.upper()}\n\n")
                    counter += 1
        
        print(f"✅ Subtitles saved to {srt_path}")
        return srt_path
    
    def _format_srt_time(self, seconds):
        """Format seconds to SRT timestamp"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    def create_viral_video(self, auto_generate_subs=True, subtitle_style="cinematic",
                       bg_music=None, bg_volume=0.15, fps=30):
        
        import time
        overall_start = time.time()
        
        duration = self.get_audio_duration()
        print(f"\n{'='*70}")
        print(f"🎬 CREATING VIRAL VIDEO (FAST MODE)")
        print(f"{'='*70}")
        print(f"⏱️  Total Duration: {duration:.2f} seconds")
        print(f"⚡ Optimized for SPEED with subtle motion effects")
        
        srt_path = None
        if auto_generate_subs:
            if os.path.exists('subtitles.srt'):
                print(f"✅ Using existing subtitles.srt")
                srt_path = 'subtitles.srt'
            else:
                srt_path = self.generate_subtitles_with_whisper()
                if not srt_path:
                    print(f"⚠️  Continuing without subtitles...")
        
        print(f"\n🧠 Analyzing content for smart B-roll matching...")
        top_categories = self.analyze_subtitles_for_keywords(srt_path) if srt_path else list(self.broll_dirs.keys())[:3]
        print(f"📊 Top themes detected: {', '.join(top_categories)}")
        
        segments = self.create_segment_plan(duration, top_categories)
        
        print(f"\n📋 VIDEO SEGMENTS PLAN:")
        print(f"{'='*70}")
        for i, seg in enumerate(segments):
            w, h, aspect = self.get_video_info(seg['file'])
            ratio = f"{w}x{h}" if w else "unknown"
            print(f"  {i+1}. B-roll ({seg['duration']:.1f}s) - {seg['category']} - {os.path.basename(seg['file'])} [{ratio}]")
        
        print(f"\n🎬 PASS 1: Processing {len(segments)} segments (FAST)...")
        temp_files = []
        concat_list = "concat_list.txt"
        concat_output = "concatenated_video.mp4"
        
        try:
            try:
                from tqdm import tqdm
                use_tqdm = True
            except ImportError:
                use_tqdm = False
            
            for i, seg in enumerate(segments):
                temp_file = f"temp_segment_{i:02d}.mp4"
                start_time = time.time()
                
                if use_tqdm:
                    total_frames = int(seg['duration'] * fps)
                    pbar = tqdm(total=total_frames, 
                            desc=f"  Segment {i+1}/{len(segments)}: {os.path.basename(seg['file'])[:30]}", 
                            unit='frame',
                            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
                    
                    def update_progress(current, total):
                        pbar.n = min(current, total)
                        pbar.refresh()
                    
                    try:
                        self.process_segment_to_file(seg, temp_file, fps, progress_callback=update_progress)
                    finally:
                        pbar.n = pbar.total 
                        pbar.refresh()
                        pbar.close()
                        elapsed = time.time() - start_time
                        print(f"    ✓ Done in {elapsed:.1f}s")
                else:
                    print(f"  Processing segment {i+1}/{len(segments)}: {os.path.basename(seg['file'])}", end='', flush=True)
                    self.process_segment_to_file(seg, temp_file, fps)
                    elapsed = time.time() - start_time
                    print(f" ✓ ({elapsed:.1f}s)")
                
                temp_files.append(temp_file)
            
            print(f"\n🎬 PASS 2: Concatenating {len(temp_files)} segments...")
            concat_start = time.time()
            
            with open(concat_list, 'w') as f:
                for tf in temp_files:
                    f.write(f"file '{tf}'\n")
            
            if use_tqdm:
                print("  Merging segments...", end='', flush=True)
            
            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', concat_list,
                '-c', 'copy',
                concat_output
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            concat_elapsed = time.time() - concat_start
            
            if use_tqdm:
                print(f" ✓ ({concat_elapsed:.1f}s)")
            else:
                print(f"  ✓ Concatenation complete ({concat_elapsed:.1f}s)")
            
            # PASS 3: Add subtitles + audio
            print(f"\n🎬 PASS 3: Adding subtitles, audio, and music...")
            final_start = time.time()
            
            cmd = ['ffmpeg', '-y', '-i', concat_output, '-i', self.audio_path]
            if bg_music and os.path.exists(bg_music):
                cmd.extend(['-i', bg_music])
                print(f"  🎵 Including background music")
            
            # Subtitles
            if srt_path and os.path.exists(srt_path):
                print(f"  📝 Adding {subtitle_style} style subtitles")
                sub_path = srt_path.replace('\\', '/').replace(':', '\\:')
                subtitle_styles = {
                    'love_pink': "force_style='FontName=Comic Sans MS,FontSize=16,PrimaryColour=&H00FFB6FF,OutlineColour=&H00FFFFFF,BorderStyle=1,Outline=2,Shadow=3,MarginV=130,Alignment=2,Bold=0'",
                    'cursive_elegant': "force_style='FontName=Great Vibes,FontSize=18,PrimaryColour=&H00FFFFFF,OutlineColour=&H80000000,BackColour=&H40000000,BorderStyle=3,Outline=1,Shadow=1,Blur=1.5,MarginV=130,Alignment=2,Bold=0'",
                    'cursive_pink_soft': "force_style='FontName=Dancing Script,FontSize=17,PrimaryColour=&H00FFB6FF,OutlineColour=&H80000000,Outline=1,Shadow=0,Blur=2,MarginV=210,Alignment=2,Bold=0'",
                    'cursive_pink_blur': "force_style='FontName=Dancing Script,FontSize=17,PrimaryColour=&H00FFB6FF,OutlineColour=&H80000000,BackColour=&H25FF5588,BorderStyle=3,Outline=0.8,Shadow=0,Blur=2.5,MarginV=125,Alignment=2,Bold=0'",
                    'cursive_red_glow': "force_style='FontName=Dancing Script,FontSize=17,PrimaryColour=&H00FF8888,OutlineColour=&H00FFFFFF,BackColour=&H00000000,BorderStyle=1,Outline=0,Shadow=0,Blur=3.5,MarginV=125,Alignment=2,Bold=0'",
                    'cursive_white_softpink': "force_style='FontName=Dancing Script,FontSize=17,PrimaryColour=&H00FFFFFF,OutlineColour=&H80000000,BackColour=&H30FF99BB,BorderStyle=3,Outline=1,Shadow=0,Blur=2,MarginV=125,Alignment=2,Bold=0'",
                    'cursive_luxury': "force_style='FontName=Alex Brush,FontSize=18,PrimaryColour=&H00FFDDAA,OutlineColour=&H80000000,BackColour=&H35000000,BorderStyle=3,Outline=1,Shadow=1,Blur=1.5,MarginV=130,Alignment=2,Bold=0'",
                    'handwriting_white': "force_style='FontName=Reenie Beanie,FontSize=18,PrimaryColour=&H00FFFFFF,OutlineColour=&H80000000,BackColour=&H30000000,BorderStyle=3,Outline=1,Shadow=1,Blur=1,MarginV=125,Alignment=2,Bold=0'",
                    'romantic_gold': "force_style='FontName=Great Vibes,FontSize=18,PrimaryColour=&H00C19A6B,OutlineColour=&H80000000,BackColour=&H40000000,BorderStyle=3,Outline=1,Shadow=1,Blur=2,MarginV=130,Alignment=2,Bold=0'",
                    'brush_script': "force_style='FontName=Brush Script MT Italic,FontSize=17,PrimaryColour=&H00FFD700,OutlineColour=&H80000000,BackColour=&H35000000,BorderStyle=3,Outline=1,Shadow=1,Blur=1.5,MarginV=125,Alignment=2,Bold=0'",
                }
                sub_style = subtitle_styles.get(subtitle_style, subtitle_styles['love_pink'])
                vf = f"subtitles='{sub_path}':{sub_style}"
                cmd.extend(['-vf', vf])
            else:
                print(f"  ⚠️  Skipping subtitles (not available)")
            
            if bg_music and os.path.exists(bg_music):
                filter_complex = (
                    f'[1:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,volume=1.0[voice];'
                    f'[2:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,volume={bg_volume},aloop=loop=-1:size=2e+09[bg];'
                    f'[voice][bg]amix=inputs=2:duration=first:dropout_transition=2,aresample=48000[aout]'
                )
                cmd.extend([
                    '-filter_complex', filter_complex,
                    '-map', '0:v',
                    '-map', '[aout]'
                ])
            else:
                cmd.extend([
                    '-map', '0:v',
                    '-af', 'aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo',
                    '-map', '1:a'
                ])
            
            # Final encoding
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
            final_elapsed = time.time() - final_start
            print(f"  ✓ Final video complete ({final_elapsed:.1f}s)")
            
            cta_output = self.output_path.replace(".mp4", "_cta.mp4")
            self._add_cta_overlay(self.output_path, cta_output, duration, niche=top_categories[0])
            self.output_path = cta_output
        
            total_time = time.time() - overall_start
            file_size = os.path.getsize(self.output_path) / (1024 * 1024)
            
            if file_size < 5.0:
                print(f"\n⚠️  WARNING: Output file is suspiciously small ({file_size:.2f} MB)")
                return False
            
            print(f"\n{'='*70}")
            print(f"✅ VIRAL VIDEO READY WITH CTA!")
            print(f"{'='*70}")
            print(f"📁 Output: {self.output_path}")
            print(f"💾 Size: {file_size:.2f} MB")
            print(f"⏱️  Duration: {duration:.2f}s")
            print(f"⚡ Processing Time: {total_time:.1f}s ({total_time/60:.1f} min)")
            print(f"🎨 Segments: {len(segments)} clips with subtle motion")
            print(f"🎯 Themes: {', '.join(top_categories)}")
            if not srt_path:
                print(f"⚠️  Note: Video created WITHOUT subtitles")
            print(f"\n🚀 Ready to GO VIRAL on TikTok/Shorts/Reels!")
            print(f"{'='*70}\n")
            return True
        
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Error creating video: {e}")
            if e.stderr:
                print(f"Error details:\n{e.stderr.decode()[-1000:]}")
            return False
        
        finally:
            print(f"\n🧹 Cleaning up temporary files...")
            for tf in temp_files:
                if os.path.exists(tf):
                    os.remove(tf)
            if os.path.exists(concat_list):
                os.remove(concat_list)
            if os.path.exists(concat_output):
                os.remove(concat_output)


NICHE_TEMPLATES = {
    'philosophy': {
        'broll_dirs': {
            'castle': 'medieval_castle_imgs',
            'chess': 'chess_strategy_imgs',
            'book': 'old_book_pages_turning_vids',
            'candle': 'candle_flame_vids',
            'ink': 'ink_writing_vids',
            'storm': 'storm_clouds_time_lapse_vids'
        },
        'keyword_map': {
            'castle': ['power', 'prince', 'war', 'kingdom', 'ruler', 'conquer', 'throne', 'empire'],
            'chess': ['strategy', 'wise', 'think', 'plan', 'move', 'game', 'cunning', 'clever'],
            'book': ['write', 'book', 'knowledge', 'teach', 'learn', 'wisdom', 'read'],
            'ink': ['write', 'author', 'pen', 'letter', 'word', 'text', 'document'],
            'storm': ['chaos', 'turbulent', 'conflict', 'danger', 'dark', 'fear', 'storm'],
            'candle': ['light', 'truth', 'reveal', 'illuminate', 'see', 'darkness', 'flame']
        }
    },
    'love': {
        'broll_dirs': {
            'couple': 'couple_romantic_vids',
            'candle': 'romantic_candle_vids',
            'flowers': 'romantic_flowers_vids',
            'book': 'old_book_pages_turning_vids',

        },
        'keyword_map': {
            'book': ['write', 'book', 'knowledge', 'teach', 'learn', 'wisdom', 'read'],
            'couple': ['love', 'together', 'relationship', 'partner', 'couple', 'romance', 'us', 'we'],
            'hands': ['hold', 'touch', 'feel', 'hand', 'embrace', 'close', 'connect', 'comfort'],
            'sunset': ['beautiful', 'moment', 'sunset', 'golden', 'magic', 'special', 'forever', 'dream'],
            'candle': ['intimate', 'warm', 'cozy', 'soft', 'gentle', 'candlelight', 'romantic', 'tender'],
            'flowers': ['beauty', 'gift', 'surprise', 'flowers', 'bloom', 'garden', 'rose', 'bouquet'],
            'city': ['date', 'night', 'city', 'dinner', 'walk', 'adventure', 'explore', 'lights']
        }
    },
    'fitness': {
        'broll_dirs': {
            'gym': 'gym_workout_vids',
            'running': 'running_outdoor_vids',
            'weights': 'weightlifting_vids',
            'protein': 'healthy_food_vids',
            'motivation': 'motivation_quotes_imgs',
            'transformation': 'body_transformation_vids'
        },
        'keyword_map': {
            'gym': ['workout', 'train', 'exercise', 'gym', 'fitness', 'lift', 'muscle'],
            'running': ['run', 'cardio', 'endurance', 'sprint', 'marathon', 'distance'],
            'weights': ['strength', 'power', 'lift', 'weight', 'barbell', 'dumbbell', 'squat'],
            'protein': ['nutrition', 'diet', 'eat', 'protein', 'meal', 'food', 'healthy'],
            'motivation': ['mindset', 'discipline', 'goals', 'push', 'grind', 'hustle', 'motivation'],
            'transformation': ['progress', 'change', 'transform', 'before', 'after', 'results', 'journey']
        }
    },
    'business': {
        'broll_dirs': {
            'office': 'modern_office_vids',
            'money': 'money_cash_vids',
            'charts': 'business_charts_vids',
            'handshake': 'business_handshake_imgs',
            'city': 'city_skyline_vids',
            'laptop': 'working_laptop_vids'
        },
        'keyword_map': {
            'office': ['work', 'business', 'company', 'corporate', 'professional', 'career'],
            'money': ['money', 'profit', 'revenue', 'income', 'earn', 'wealth', 'rich'],
            'charts': ['growth', 'data', 'analytics', 'metrics', 'performance', 'results'],
            'handshake': ['deal', 'partnership', 'agreement', 'negotiate', 'contract', 'client'],
            'city': ['success', 'ambition', 'empire', 'scale', 'expand', 'global'],
            'laptop': ['digital', 'online', 'remote', 'technology', 'startup', 'entrepreneur']
        }
    },
    'nature': {
        'broll_dirs': {
            'ocean': 'ocean_waves_vids',
            'mountain': 'mountain_landscape_vids',
            'forest': 'forest_trees_vids',
            'sunset': 'sunset_timelapse_vids',
            'wildlife': 'wildlife_animals_vids',
            'flowers': 'flowers_nature_vids'
        },
        'keyword_map': {
            'ocean': ['sea', 'ocean', 'water', 'wave', 'beach', 'coast', 'marine'],
            'mountain': ['mountain', 'peak', 'climb', 'summit', 'altitude', 'high', 'ridge'],
            'forest': ['forest', 'tree', 'wood', 'nature', 'green', 'wilderness'],
            'sunset': ['sun', 'light', 'dawn', 'dusk', 'sky', 'golden', 'beautiful'],
            'wildlife': ['animal', 'wild', 'creature', 'species', 'habitat', 'natural'],
            'flowers': ['flower', 'bloom', 'garden', 'beauty', 'color', 'petals', 'plant']
        }
    }
}


# =============== FASTAPI ENDPOINTS ===============

@app.get("/")
def root():
    return {
        "service": "Viral Shorts Generator",
        "status": "running",
        "endpoints": {
            "POST /generate": "Generate video from GitHub audio",
            "GET /status": "Check status",
            "GET /download": "Download video"
        }
    }

@app.post("/generate")
async def generate_video_api(background_tasks: BackgroundTasks):
    global current_job
    
    if current_job["status"] == "processing":
        return {"message": "Already processing", "status": "processing"}
    
    current_job = {
        "status": "processing",
        "progress": 0,
        "output": None,
        "error": None,
        "started_at": datetime.now().isoformat()
    }
    
    background_tasks.add_task(process_video)
    
    return {
        "message": "Video generation started",
        "status": "processing"
    }

def process_video():
    global current_job
    
    try:
        current_job["progress"] = 10
        
        # Download latest audio from GitHub
        print("📥 Downloading audio from GitHub...")
        url = "https://raw.githubusercontent.com/RandomSci/Automation_For_Love_Niche/main/Audio_Voice/new_love.mp3"
        response = requests.get(url)
        
        # Ensure directory exists
        os.makedirs("Audio_Voice", exist_ok=True)
        
        with open("Audio_Voice/new_love.mp3", "wb") as f:
            f.write(response.content)
        
        current_job["progress"] = 20
        
        # Clean up old files
        for old_file in ["new_love.mp4", "new_love_cta.mp4", "subtitles.srt", "Audio_Voice/new_love_transcription.json"]:
            if os.path.exists(old_file):
                os.remove(old_file)
        
        current_job["progress"] = 30
        
        # Generate video using the existing main() logic
        NICHE = 'love'
        main_image = "main_images/Dating_.jpg"
        audio = "Audio_Voice/new_love.mp3"
        output = "new_love.mp4"
        bg_music = "bg_musics/For_Dating.mp3"
        
        if not os.path.exists(main_image):
            raise Exception(f"Main image not found: {main_image}")
        
        if not os.path.exists(audio):
            raise Exception(f"Audio file not found: {audio}")
        
        niche_config = NICHE_TEMPLATES.get(NICHE)
        
        print(f"\n🎯 Using '{NICHE.upper()}' niche template")
        
        gen = ViralShortsGenerator(main_image, audio, output, niche_config=niche_config)
        
        success = gen.create_viral_video(
            auto_generate_subs=True,
            subtitle_style="cursive_pink_soft",
            bg_music=bg_music if bg_music and os.path.exists(bg_music) else None,
            bg_volume=0.25,
            fps=30
        )
        
        # Clean up temp files
        subtitle_file = "subtitles.srt" 
        new_love_transcription_file = "Audio_Voice/new_love_transcription.json"
        
        if success and os.path.exists(subtitle_file):
            try:
                os.remove(subtitle_file)
                if os.path.exists(new_love_transcription_file):
                    os.remove(new_love_transcription_file)
                print(f"🗑 Deleted temporary files")
            except Exception as e:
                print(f"⚠ Failed to delete temp files: {e}")
        
        if success and os.path.exists("new_love_cta.mp4"):
            current_job["status"] = "completed"
            current_job["progress"] = 100
            current_job["output"] = "new_love_cta.mp4"
            print("✅ Video ready!")
        else:
            raise Exception("Video generation failed")
        
    except Exception as e:
        current_job["status"] = "error"
        current_job["error"] = str(e)
        print(f"❌ Error: {e}")

@app.get("/status")
def check_status():
    return {
        "status": current_job["status"],
        "progress": current_job["progress"],
        "error": current_job["error"],
        "started_at": current_job["started_at"],
        "ready": current_job["status"] == "completed"
    }

@app.get("/download")
def download_video():
    if current_job["status"] != "completed":
        raise HTTPException(400, f"Not ready. Status: {current_job['status']}")
    
    if not current_job["output"] or not os.path.exists(current_job["output"]):
        raise HTTPException(404, "Video file not found")
    
    return FileResponse(
        current_job["output"],
        media_type="video/mp4",
        filename=f"viral_video.mp4"
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
import subprocess
import sys
import os

video = "vaults_output_nav_cta.mp4"
audio = "Audio_Voice/vaults_narration.mp3"
output = "test_clean_merge.mp4"

# Strip video to silent mp4 first
subprocess.run([
    "ffmpeg", "-y", "-i", video,
    "-an", "-c:v", "copy",
    "video_silent.mp4"
], capture_output=True)

# Merge silent video with raw audio using stream copy — zero re-encoding
result = subprocess.run([
    "ffmpeg", "-y",
    "-i", "video_silent.mp4",
    "-i", audio,
    "-map", "0:v",
    "-map", "1:a",
    "-c:v", "copy",
    "-c:a", "copy",
    "-shortest",
    output
], capture_output=True, text=True)

if result.returncode == 0:
    print(f"Done: {output}")
else:
    print(result.stderr[-400:])

os.remove("video_silent.mp4")

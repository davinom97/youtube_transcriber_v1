# YouTube Transcriber v1

Runner-friendly transcription for YouTube videos, YouTube playlists, and Instagram Reels.

The main script downloads audio with `yt-dlp`, transcribes it with `faster-whisper`, and writes:

- plain `.txt`
- timestamped `.timestamped.txt`
- `.json` segment data
- `.srt` subtitles
- `.vtt` subtitles
- `.metadata.json`
- `manifest.json` for runner status

## Install

```powershell
python -m pip install -r requirements-transcribe.txt
```

The script needs `ffmpeg`. If system `ffmpeg` is not available, `imageio-ffmpeg` from the requirements file is used automatically.

## Transcribe A Playlist

```powershell
python transcribe_social_video.py "https://www.youtube.com/playlist?list=PLAYLIST_ID" --playlist --quality high --device cuda --output-dir transcripts
```

Use `--device cpu` if the remote machine does not have CUDA.

## Remote Runner

Use `remote_transcription_job.example.json` as the project-runner job template. Replace `PLAYLIST_ID` with the real playlist URL.

You can also use the PowerShell wrapper:

```powershell
.\run_playlist_transcription.ps1 -PlaylistUrl "https://www.youtube.com/playlist?list=PLAYLIST_ID" -OutputDir transcripts -Device cuda
```

## Quality Presets

- `fast`: Whisper `base`
- `standard`: Whisper `small`
- `high`: Whisper `medium`
- `best`: Whisper `large-v3`

You can override the preset with `--model`, `--beam-size`, or `--compute-type`.

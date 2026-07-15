# Remote Runner Setup

Use these values in the Remote Project Runner **Add App** form.

| Field | Value |
| --- | --- |
| ID | `youtube-transcriber` |
| Display Name | `Youtube Transcriber` |
| Working Directory | `C:\RemoteProjectRunner\projects\youtube_transcriber_v1` |
| Runtime | `PYTHON_VENV` |
| Command | `python` |
| Default Args | `transcribe_social_video.py "https://youtube.com/playlist?list=PL3O-wzvCVtzWg0O3q3hV-Hnge__cHVn0D&si=OA2qGX0fZ83j5thh" --playlist --quality best --device cuda --output-dir transcripts` |
| Env JSON | `{"PYTHONUNBUFFERED":"1"}` |
| Git Repo URL | `https://github.com/davinom97/youtube_transcriber_v1.git` |
| Git Branch | `main` |
| Enabled | checked |

## Runner Token

If Save shows `Missing or invalid runner token`, set the runner service token before starting the Remote Project Runner backend.

PowerShell example:

```powershell
$env:RUNNER_TOKEN="choose-a-long-random-secret"
```

Then start the runner backend from that same terminal.

The browser/UI must send the same token when saving apps. If the UI has a token, settings, or login field, paste the exact same value there.

Do not put `RUNNER_TOKEN` in the app Env JSON. The Env JSON belongs to the transcription job, not the runner admin API.

## CPU Fallback

The configured app uses:

```text
--quality best --device cuda
```

That gives the highest quality and expects a CUDA-capable NVIDIA GPU. If the remote machine does not have CUDA working, change `--device cuda` to `--device cpu`. CPU transcription with `--quality best` will be slow.

#!/usr/bin/env python3
"""
Runner-friendly YouTube/Instagram transcription.

Designed for unattended execution by a remote project runner:
  - accepts one URL, many URLs, or a urls.txt file
  - expands YouTube playlists
  - transcribes every item with a reusable Whisper model
  - writes txt, timestamped txt, json, srt, vtt, metadata, and a job manifest
  - keeps going after per-video failures and reports them at the end

Install:
  pip install -r requirements-transcribe.txt

Example high-quality playlist run:
  python transcribe_social_video.py "https://www.youtube.com/playlist?list=PLAYLIST_ID" ^
    --playlist --quality high --output-dir transcripts
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


MEDIA_SUFFIXES = {".m4a", ".mp3", ".opus", ".ogg", ".wav", ".webm", ".mp4", ".mkv"}


QUALITY_PRESETS: dict[str, dict[str, Any]] = {
    "fast": {"model": "base", "beam_size": 3, "compute_type": "int8"},
    "standard": {"model": "small", "beam_size": 5, "compute_type": "int8"},
    "high": {"model": "medium", "beam_size": 5, "compute_type": "auto"},
    "best": {"model": "large-v3", "beam_size": 5, "compute_type": "auto"},
}


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class VideoJob:
    url: str
    title: str
    video_id: str
    playlist_title: str | None = None
    playlist_index: int | None = None


@dataclass
class JobResult:
    status: str
    url: str
    title: str | None = None
    video_id: str | None = None
    outputs: dict[str, str] | None = None
    error: str | None = None


def log(message: str) -> None:
    print(message, flush=True)


def sanitize_filename(name: str, max_length: int = 120) -> str:
    safe = "".join(ch if ch.isalnum() or ch in " ._-()" else "_" for ch in name)
    safe = " ".join(safe.split()).strip(" ._")
    return (safe or "transcript")[:max_length]


def timestamp(seconds: float, decimal: str = ",") -> str:
    millis = int(round(seconds * 1000))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{decimal}{millis:03d}"


def import_yt_dlp():
    try:
        import yt_dlp
    except ImportError as exc:
        raise SystemExit("Missing package: yt-dlp. Install with: pip install yt-dlp") from exc
    return yt_dlp


def import_whisper_model():
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise SystemExit(
            "Missing package: faster-whisper. Install with: pip install faster-whisper"
        ) from exc
    return WhisperModel


def find_ffmpeg(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def read_urls(args: argparse.Namespace) -> list[str]:
    urls = list(args.urls or [])
    env_url = os.environ.get("TRANSCRIBE_URL")
    env_urls = os.environ.get("TRANSCRIBE_URLS")
    if args.urls_file:
        urls.extend(
            line.strip()
            for line in args.urls_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    if env_url:
        urls.append(env_url.strip())
    if env_urls:
        urls.extend(part.strip() for part in env_urls.splitlines() if part.strip())

    deduped = []
    seen = set()
    for url in urls:
        if url not in seen:
            deduped.append(url)
            seen.add(url)
    return deduped


def ydl_base_options(args: argparse.Namespace) -> dict[str, Any]:
    options: dict[str, Any] = {
        "quiet": args.quiet,
        "no_warnings": False,
        "ignoreerrors": False,
        "retries": args.retries,
        "fragment_retries": args.fragment_retries,
    }
    if args.cookies:
        options["cookiefile"] = str(args.cookies)
    if args.cookies_from_browser:
        options["cookiesfrombrowser"] = (args.cookies_from_browser,)
    if args.ffmpeg_location:
        options["ffmpeg_location"] = args.ffmpeg_location
    if args.js_runtime:
        options["js_runtimes"] = {args.js_runtime_name: {"path": args.js_runtime}}
    return options


def video_url_from_entry(entry: dict[str, Any]) -> str | None:
    if entry.get("webpage_url"):
        return entry["webpage_url"]
    if entry.get("url", "").startswith(("http://", "https://")):
        return entry["url"]
    if entry.get("ie_key") == "Youtube" and entry.get("id"):
        return f"https://www.youtube.com/watch?v={entry['id']}"
    if entry.get("extractor_key") == "Youtube" and entry.get("id"):
        return f"https://www.youtube.com/watch?v={entry['id']}"
    return entry.get("url")


def expand_urls(urls: list[str], args: argparse.Namespace) -> list[VideoJob]:
    yt_dlp = import_yt_dlp()
    jobs: list[VideoJob] = []
    options = ydl_base_options(args)
    options.update(
        {
            "extract_flat": "in_playlist",
            "skip_download": True,
            "noplaylist": not args.playlist,
        }
    )

    with yt_dlp.YoutubeDL(options) as ydl:
        for url in urls:
            log(f"Inspecting: {url}")
            info = ydl.extract_info(url, download=False)
            if not info:
                continue

            if info.get("_type") == "playlist" and args.playlist:
                playlist_title = info.get("title")
                for fallback_index, entry in enumerate(info.get("entries") or [], start=1):
                    if not entry:
                        continue
                    item_url = video_url_from_entry(entry)
                    if not item_url:
                        continue
                    jobs.append(
                        VideoJob(
                            url=item_url,
                            title=entry.get("title") or entry.get("id") or f"video-{fallback_index}",
                            video_id=entry.get("id") or "",
                            playlist_title=playlist_title,
                            playlist_index=entry.get("playlist_index") or fallback_index,
                        )
                    )
            else:
                jobs.append(
                    VideoJob(
                        url=url,
                        title=info.get("title") or info.get("id") or "video",
                        video_id=info.get("id") or "",
                    )
                )

    if args.limit:
        jobs = jobs[: args.limit]
    return jobs


def make_stem(job: VideoJob) -> str:
    parts = []
    if job.playlist_index is not None:
        parts.append(f"{job.playlist_index:03d}")
    parts.append(job.title)
    if job.video_id:
        parts.append(job.video_id)
    return sanitize_filename(" - ".join(parts), max_length=150)


def expected_outputs(output_dir: Path, stem: str) -> dict[str, Path]:
    return {
        "txt": output_dir / f"{stem}.txt",
        "timestamped": output_dir / f"{stem}.timestamped.txt",
        "json": output_dir / f"{stem}.json",
        "srt": output_dir / f"{stem}.srt",
        "vtt": output_dir / f"{stem}.vtt",
        "metadata": output_dir / f"{stem}.metadata.json",
    }


def outputs_complete(paths: dict[str, Path]) -> bool:
    return all(path.exists() and path.stat().st_size > 0 for path in paths.values())


def download_audio(job: VideoJob, temp_dir: Path, args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    yt_dlp = import_yt_dlp()
    options = ydl_base_options(args)
    options.update(
        {
            "format": args.ytdlp_format,
            "noplaylist": True,
            "outtmpl": str(temp_dir / "%(title).120s [%(id)s].%(ext)s"),
            "writethumbnail": False,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": args.audio_codec,
                    "preferredquality": "0",
                }
            ],
        }
    )

    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(job.url, download=True)

    candidates = [
        item
        for item in temp_dir.rglob("*")
        if item.is_file() and item.suffix.lower() in MEDIA_SUFFIXES and item.stat().st_size > 0
    ]
    if not candidates:
        raise RuntimeError("Download finished, but no audio/media file was found.")
    return max(candidates, key=lambda item: item.stat().st_size), info or {}


def transcribe_audio(model: Any, audio_path: Path, args: argparse.Namespace) -> tuple[list[Segment], Any]:
    segments_iter, info = model.transcribe(
        str(audio_path),
        language=args.language,
        vad_filter=not args.no_vad,
        beam_size=args.beam_size,
        condition_on_previous_text=not args.no_condition_on_previous_text,
    )
    segments = [
        Segment(start=segment.start, end=segment.end, text=segment.text.strip())
        for segment in segments_iter
        if segment.text.strip()
    ]
    return segments, info


def write_txt(path: Path, segments: Iterable[Segment]) -> None:
    path.write_text(
        "\n".join(segment.text for segment in segments if segment.text).strip() + "\n",
        encoding="utf-8",
    )


def write_timestamped(path: Path, segments: Iterable[Segment]) -> None:
    path.write_text(
        "\n".join(f"[{timestamp(s.start, decimal='.')}] {s.text}" for s in segments).strip()
        + "\n",
        encoding="utf-8",
    )


def write_json(path: Path, job: VideoJob, segments: list[Segment], info: Any) -> None:
    payload = {
        "source": asdict(job),
        "language": getattr(info, "language", None),
        "language_probability": getattr(info, "language_probability", None),
        "duration": getattr(info, "duration", None),
        "segments": [asdict(segment) for segment in segments],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_srt(path: Path, segments: Iterable[Segment]) -> None:
    blocks = []
    for index, segment in enumerate(segments, start=1):
        blocks.append(
            f"{index}\n"
            f"{timestamp(segment.start)} --> {timestamp(segment.end)}\n"
            f"{segment.text}\n"
        )
    path.write_text("\n".join(blocks), encoding="utf-8")


def write_vtt(path: Path, segments: Iterable[Segment]) -> None:
    lines = ["WEBVTT", ""]
    for segment in segments:
        lines.extend(
            [
                f"{timestamp(segment.start, decimal='.')} --> "
                f"{timestamp(segment.end, decimal='.')}",
                segment.text,
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def save_outputs(
    paths: dict[str, Path],
    job: VideoJob,
    segments: list[Segment],
    info: Any,
    metadata: dict[str, Any],
) -> dict[str, str]:
    paths["txt"].parent.mkdir(parents=True, exist_ok=True)
    write_txt(paths["txt"], segments)
    write_timestamped(paths["timestamped"], segments)
    write_json(paths["json"], job, segments, info)
    write_srt(paths["srt"], segments)
    write_vtt(paths["vtt"], segments)
    paths["metadata"].write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return {key: str(path) for key, path in paths.items()}


def resolve_presets(args: argparse.Namespace) -> argparse.Namespace:
    preset = QUALITY_PRESETS[args.quality]
    if args.model is None:
        args.model = preset["model"]
    if args.beam_size is None:
        args.beam_size = preset["beam_size"]
    if args.compute_type is None:
        args.compute_type = preset["compute_type"]
    if args.device == "cuda" and args.compute_type == "auto":
        args.compute_type = "float16"
    elif args.compute_type == "auto":
        args.compute_type = "int8"
    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe videos or YouTube playlists in unattended runner jobs."
    )
    parser.add_argument("urls", nargs="*", help="Video, Reel, or playlist URLs.")
    parser.add_argument("--urls-file", type=Path, help="Text file containing one URL per line.")
    parser.add_argument("--playlist", action="store_true", help="Expand playlist URLs.")
    parser.add_argument("--limit", type=int, help="Optional max number of videos to process.")
    parser.add_argument("--output-dir", type=Path, default=Path("transcripts"))
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--quality", choices=sorted(QUALITY_PRESETS), default="high")
    parser.add_argument("--model", help="Override preset model, e.g. medium, medium.en, large-v3.")
    parser.add_argument("--language", help="Optional spoken language code, e.g. en.")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--compute-type", help="Override compute type, e.g. int8, float16.")
    parser.add_argument("--beam-size", type=int, help="Override preset beam size.")
    parser.add_argument("--no-vad", action="store_true", help="Disable voice activity filtering.")
    parser.add_argument("--no-condition-on-previous-text", action="store_true")
    parser.add_argument("--ytdlp-format", default="bestaudio/best")
    parser.add_argument("--audio-codec", default="m4a", help="Audio codec produced by yt-dlp/ffmpeg.")
    parser.add_argument("--ffmpeg-location", help="Path to ffmpeg or ffmpeg directory.")
    parser.add_argument("--cookies", type=Path, help="Netscape cookies.txt file.")
    parser.add_argument("--cookies-from-browser", help="Browser name for yt-dlp cookie import.")
    parser.add_argument("--js-runtime", help="Path to node/deno for YouTube player JS extraction.")
    parser.add_argument("--js-runtime-name", default="node")
    parser.add_argument("--retries", type=int, default=10)
    parser.add_argument("--fragment-retries", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-audio", action="store_true")
    parser.add_argument("--ignore-failures", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    return resolve_presets(args)


def main() -> int:
    args = parse_args()
    urls = read_urls(args)
    if not urls:
        raise SystemExit(
            "No URLs supplied. Pass URLs as arguments, --urls-file, TRANSCRIBE_URL, or TRANSCRIBE_URLS."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir = args.work_dir or args.output_dir / "_work"
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.ffmpeg_location = find_ffmpeg(args.ffmpeg_location)
    if not args.ffmpeg_location:
        raise SystemExit(
            "ffmpeg was not found. Install ffmpeg or install imageio-ffmpeg with pip."
        )

    jobs = expand_urls(urls, args)
    log(f"Found {len(jobs)} video(s) to transcribe.")
    log(
        f"Using quality={args.quality}, model={args.model}, device={args.device}, "
        f"compute_type={args.compute_type}, beam_size={args.beam_size}"
    )

    WhisperModel = import_whisper_model()
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)

    results: list[JobResult] = []
    manifest_path = args.output_dir / "manifest.json"
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    for index, job in enumerate(jobs, start=1):
        stem = make_stem(job)
        paths = expected_outputs(args.output_dir, stem)
        log(f"[{index}/{len(jobs)}] {job.title}")

        if not args.overwrite and outputs_complete(paths):
            log(f"Skipping existing transcript: {stem}")
            results.append(
                JobResult(
                    status="skipped",
                    url=job.url,
                    title=job.title,
                    video_id=job.video_id,
                    outputs={key: str(path) for key, path in paths.items()},
                )
            )
            continue

        try:
            with tempfile.TemporaryDirectory(dir=args.work_dir) as temp_name:
                temp_dir = Path(temp_name)
                audio_path, metadata = download_audio(job, temp_dir, args)
                segments, info = transcribe_audio(model, audio_path, args)
                output_paths = save_outputs(paths, job, segments, info, metadata)

                if args.keep_audio:
                    audio_dir = args.output_dir / "audio"
                    audio_dir.mkdir(parents=True, exist_ok=True)
                    kept_audio = audio_dir / f"{stem}{audio_path.suffix}"
                    shutil.copy2(audio_path, kept_audio)
                    output_paths["audio"] = str(kept_audio)

            results.append(
                JobResult(
                    status="ok",
                    url=job.url,
                    title=job.title,
                    video_id=job.video_id,
                    outputs=output_paths,
                )
            )
            log(f"Finished: {stem}")
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            log(f"Failed: {job.url}\n{message}")
            results.append(
                JobResult(
                    status="failed",
                    url=job.url,
                    title=job.title,
                    video_id=job.video_id,
                    error=message,
                )
            )

        manifest = {
            "started_at": started_at,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "quality": args.quality,
            "model": args.model,
            "device": args.device,
            "compute_type": args.compute_type,
            "beam_size": args.beam_size,
            "results": [asdict(result) for result in results],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    failures = [result for result in results if result.status == "failed"]
    log(f"Complete: {len(results) - len(failures)}/{len(results)} succeeded.")
    return 0 if args.ignore_failures or not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

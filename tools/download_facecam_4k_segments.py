#!/usr/bin/env python3
"""Download only the selected highlight ranges in 4K for higher-quality facecam crops.

The full source is already downloaded by the main workflow. If it is below
2160p, this helper asks yt-dlp for 2160p+ video-only sections for each selected
highlight. When no 4K format exists or a download fails, it leaves that manifest
entry untouched so the splitter automatically uses the original source.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


def log(message: str) -> None:
    print(message, flush=True)


def _manifest_list(data: Any) -> tuple[list[dict[str, Any]], str | None]:
    if isinstance(data, list):
        return data, None
    if isinstance(data, dict):
        for key in ("results", "clips", "manifest"):
            value = data.get(key)
            if isinstance(value, list):
                return value, key
    raise RuntimeError("Unsupported render manifest structure.")


def probe_video(path: Path) -> tuple[int, int, float]:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height:format=duration",
            "-of", "json", str(path),
        ],
        check=True, capture_output=True, text=True,
    )
    info = json.loads(result.stdout)
    streams = info.get("streams") or []
    if not streams:
        raise RuntimeError(f"No video stream found in {path}")
    width = int(streams[0].get("width", 0))
    height = int(streams[0].get("height", 0))
    duration = float((info.get("format") or {}).get("duration", 0) or 0)
    return width, height, duration


def _safe_rank(value: Any, fallback: int) -> str:
    raw = str(value if value is not None else fallback)
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", raw).strip("_")
    return cleaned or str(fallback)


def download_sections(
    url: str,
    manifest_path: Path,
    source_path: Path,
    cookies_path: Path,
    output_dir: Path,
) -> int:
    if not source_path.is_file():
        raise RuntimeError(f"Base source file does not exist: {source_path}")
    if not manifest_path.is_file():
        raise RuntimeError(f"Render manifest does not exist: {manifest_path}")
    if not cookies_path.is_file():
        raise RuntimeError(f"YouTube cookies file does not exist: {cookies_path}")

    width, height, _ = probe_video(source_path)
    log(f"Base source is {width}x{height}.")
    if height >= 2160:
        log("Base source is already 4K or higher; no extra 4K segments needed.")
        return 0

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    clips, nested_key = _manifest_list(manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    successes = 0

    for index, clip in enumerate(clips, 1):
        if not isinstance(clip, dict):
            continue
        try:
            start = float(clip.get("start_time", 0))
            end = float(clip.get("end_time", 0))
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue

        rank = _safe_rank(clip.get("rank"), index)
        # Add a short guard on both sides so seek/keyframe rounding cannot
        # remove the first or last frames needed for the facecam tracking.
        section_start = max(0.0, start - 1.5)
        section_end = end + 1.5
        facecam_start = start - section_start
        output_template = output_dir / f"facecam_rank_{rank}.%(ext)s"
        for stale in output_dir.glob(f"facecam_rank_{rank}.*"):
            if stale.is_file():
                stale.unlink(missing_ok=True)

        command = [
            sys.executable, "-m", "yt_dlp",
            "--cookies", str(cookies_path),
            "--remote-components", "ejs:github",
            "--no-warnings",
            "--download-sections", f"*{section_start:.3f}-{section_end:.3f}",
            "--format", "bestvideo[height>=2160]",
            "--output", str(output_template),
            url,
        ]
        log(
            f"[{index}/{len(clips)}] Requesting 4K facecam segment for rank {rank}: "
            f"{section_start:.2f}s–{section_end:.2f}s"
        )
        result = subprocess.run(command, check=False)
        candidates = [
            Path(name) for name in glob.glob(str(output_dir / f"facecam_rank_{rank}.*"))
            if not name.endswith((".part", ".ytdl"))
        ]
        valid_path: Path | None = None
        if result.returncode == 0:
            for candidate in candidates:
                try:
                    _w, candidate_height, duration = probe_video(candidate)
                    if candidate_height >= 2160 and duration >= max(1.0, (end - start) * 0.70):
                        valid_path = candidate.resolve()
                        break
                except Exception:
                    continue

        if valid_path is None:
            log(
                f"  WARNING: No valid 2160p+ segment for rank {rank}; "
                "the splitter will use the full base source instead."
            )
            for candidate in candidates:
                candidate.unlink(missing_ok=True)
            continue

        clip["facecam_source_path"] = str(valid_path)
        clip["facecam_source_start"] = round(facecam_start, 3)
        successes += 1
        log(f"  Using {valid_path} ({probe_video(valid_path)[1]}p).")

    if nested_key is None:
        manifest_to_write = clips
    else:
        manifest[nested_key] = clips
        manifest_to_write = manifest
    manifest_path.write_text(
        json.dumps(manifest_to_write, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    log(f"4K facecam sections ready: {successes}/{len(clips)}. Any failures will fall back automatically.")
    return successes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cookies", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    download_sections(
        args.url, args.manifest.resolve(), args.source.resolve(),
        args.cookies.resolve(), args.output_dir.resolve(),
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        log(f"ERROR: {exc}")
        raise SystemExit(1)

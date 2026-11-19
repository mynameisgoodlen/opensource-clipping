#!/usr/bin/env python3
"""Re-render selected highlights as streamer facecam (top) + gameplay (bottom).

The input is the original composited stream video. MediaPipe finds the most
persistent face location in each selected time range. The top crop is enlarged
with Lanczos + mild sharpening; the bottom crop searches for the largest 3:2
window that avoids the detected facecam region as much as possible.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path
from statistics import median
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)
MODEL_PATH = Path("blaze_face_short_range_v1.tflite")


def log(message: str) -> None:
    print(message, flush=True)


def load_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        data = data.get("results") or data.get("clips") or data.get("manifest")
    if not isinstance(data, list):
        raise RuntimeError(f"Unsupported render manifest structure: {path}")
    clips = [item for item in data if isinstance(item, dict) and item.get("video_path")]
    if not clips:
        raise RuntimeError("The render manifest contains no clips with video_path.")
    return clips


def create_detector(model_path: Path):
    if not model_path.exists():
        log("Downloading MediaPipe face detector model...")
        urllib.request.urlretrieve(MODEL_URL, model_path)
    options = mp_vision.FaceDetectorOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp_vision.RunningMode.IMAGE,
        min_detection_confidence=0.50,
    )
    detector = mp_vision.FaceDetector.create_from_options(options)
    # Trigger graph/model validation before processing the clips.
    detector.detect(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=np.zeros((128, 128, 3), dtype=np.uint8))
    )
    return detector


def _face_boxes(detector, frame: np.ndarray) -> list[tuple[float, float, float, float]]:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    found = []
    for detection in result.detections or []:
        box = detection.bounding_box
        x1 = float(box.origin_x)
        y1 = float(box.origin_y)
        x2 = float(box.origin_x + box.width)
        y2 = float(box.origin_y + box.height)
        if x2 > x1 and y2 > y1:
            found.append((x1, y1, x2, y2))
    return found


def detect_facecam_region(
    cap: cv2.VideoCapture,
    detector,
    start: float,
    end: float,
    width: int,
    height: int,
    sample_interval: float,
    facecam_corner: str = "auto",
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    """Return (facecam crop rect, gameplay crop rect) for this clip."""
    duration = max(0.1, end - start)
    sample_count = min(90, max(3, int(duration / sample_interval) + 1))
    times = np.linspace(start, max(start, end - 0.05), sample_count)

    # Cluster observations by normalized face-center position. Streamer facecam
    # overlays tend to remain in one location, unlike faces in gameplay footage.
    clusters: list[list[tuple[float, float, float, float, int]]] = []
    frame_count = 0
    for sample_no, timestamp in enumerate(times):
        cap.set(cv2.CAP_PROP_POS_MSEC, float(timestamp) * 1000.0)
        ok, frame = cap.read()
        if not ok:
            continue
        frame_count += 1
        boxes = _face_boxes(detector, frame)
        # Limit pathological crowd frames; keep the larger detections.
        boxes = sorted(
            boxes,
            key=lambda b: (b[2] - b[0]) * (b[3] - b[1]),
            reverse=True,
        )[:8]
        assigned: set[int] = set()
        for box in boxes:
            cx = ((box[0] + box[2]) / 2.0) / width
            cy = ((box[1] + box[3]) / 2.0) / height
            best_idx = None
            best_dist = 1e9
            for idx, cluster in enumerate(clusters):
                if idx in assigned:
                    continue
                ccx = median(item[0] for item in cluster)
                ccy = median(item[1] for item in cluster)
                distance = ((cx - ccx) ** 2 + (cy - ccy) ** 2) ** 0.5
                if distance < best_dist:
                    best_idx, best_dist = idx, distance
            if best_idx is not None and best_dist <= 0.105:
                clusters[best_idx].append((cx, cy, *box, sample_no))
                assigned.add(best_idx)
            else:
                clusters.append([(cx, cy, *box, sample_no)])
                assigned.add(len(clusters) - 1)

    if not clusters:
        raise RuntimeError(
            f"No face was detected between {start:.2f}s and {end:.2f}s. "
            "This split layout needs a visible facecam; check the source or set a manual facecam crop."
        )

    # A streamer inset is normally anchored near a screen corner. A GTA/NPC face
    # in a centered cutscene can remain visible for an entire clip, so raw
    # persistence alone is not enough to identify the webcam.
    corners = {
        "top-left": (lambda x, y: x <= 0.48 and y <= 0.48, 1.08),
        "top-right": (lambda x, y: x >= 0.52 and y <= 0.48, 1.05),
        "bottom-left": (lambda x, y: x <= 0.48 and y >= 0.52, 1.55),
        "bottom-right": (lambda x, y: x >= 0.52 and y >= 0.52, 1.35),
    }
    if facecam_corner != "auto" and facecam_corner not in corners:
        raise ValueError(f"Unsupported facecam corner: {facecam_corner}")

    ranked = []
    for cluster in clusters:
        cx = float(median(item[0] for item in cluster))
        cy = float(median(item[1] for item in cluster))
        areas = [max(1.0, (item[4] - item[2]) * (item[5] - item[3])) for item in cluster]
        avg_area_fraction = float(np.median(areas)) / float(width * height)
        xs = [item[0] for item in cluster]
        ys = [item[1] for item in cluster]
        positional_jitter = float(np.std(xs) + np.std(ys))
        stability = max(0.15, 1.0 - 2.2 * positional_jitter)
        # Prefer a small inset-camera face over large cutscene close-ups.
        size_score = float(np.exp(-0.22 * abs(np.log(max(avg_area_fraction, 1e-7) / 0.010))))
        coverage = len(cluster) / max(1, frame_count)

        matching_corner = None
        corner_multiplier = 0.22  # centered gameplay faces are only a fallback
        for name, (matches, multiplier) in corners.items():
            if matches(cx, cy):
                matching_corner = name
                corner_multiplier = multiplier
                break
        if facecam_corner != "auto":
            if matching_corner != facecam_corner:
                continue
            corner_multiplier = 1.0

        score = coverage * stability * size_score * corner_multiplier
        ranked.append((score, len(cluster), matching_corner or "center", cluster))

    if not ranked:
        raise RuntimeError(
            f"No face detections matched the requested {facecam_corner} facecam corner. "
            "Use --facecam-corner auto or choose the corner containing the streamer."
        )

    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    best_score, best_count, best_corner, best = ranked[0]
    min_observations = min(3, max(1, int(frame_count * 0.04)))
    if best_count < min_observations:
        raise RuntimeError("Face detections were too inconsistent to locate the streamer facecam reliably.")
    if best_corner == "center":
        log("  WARNING: No convincing corner facecam was found; using the best stable fallback face.")
    log(f"  Facecam candidate: {best_corner}, observations={best_count}/{frame_count}, score={best_score:.3f}")

    face = tuple(int(round(median(item[i] for item in best))) for i in range(2, 6))
    fx1, fy1, fx2, fy2 = face
    face_w = max(2, fx2 - fx1)
    face_h = max(2, fy2 - fy1)
    face_cx = (fx1 + fx2) / 2.0
    face_cy = (fy1 + fy2) / 2.0

    # Expand from the face to capture the webcam panel, not just the face box.
    # The 3:2 crop matches each 1080x720 panel of the 1080x1440 result.
    cam_h = min(height * 0.58, width * 0.58 / 1.5, max(220.0, face_h * 3.5, face_w * 2.4))
    cam_w = cam_h * 1.5
    cam_x1 = int(round(np.clip(face_cx - cam_w / 2, 0, width - cam_w)))
    cam_y1 = int(round(np.clip(face_cy - cam_h / 2, 0, height - cam_h)))
    cam_rect = (cam_x1, cam_y1, cam_x1 + int(cam_w), cam_y1 + int(cam_h))

    game_rect = choose_gameplay_crop(width, height, cam_rect)
    return cam_rect, game_rect


def choose_gameplay_crop(
    width: int, height: int, avoid_rect: tuple[int, int, int, int]
) -> tuple[int, int, int, int]:
    """Pick the largest useful gameplay crop without over-zooming to avoid the facecam.

    The old implementation kept shrinking until it found a zero-overlap crop,
    sometimes down to 42% of the original size. This version balances crop area
    against overlap, so gameplay stays large and sharp even when the webcam is
    baked into a corner of the source frame.
    """
    base_h = min(float(height), width / 1.5)
    base_w = base_h * 1.5
    ax1, ay1, ax2, ay2 = avoid_rect
    best_score = -1e9
    best_rect = (0, 0, int(base_w), int(base_h))
    base_area = max(1.0, base_w * base_h)

    for scale in np.linspace(1.0, 0.72, 15):
        cw = max(2, int(round(base_w * float(scale))))
        ch = max(2, int(round(cw / 1.5)))
        if ch > height:
            ch = height
            cw = int(ch * 1.5)
        max_x = max(0, width - cw)
        max_y = max(0, height - ch)
        xs = set(int(round(x)) for x in np.linspace(0, max_x, 11))
        ys = set(int(round(y)) for y in np.linspace(0, max_y, 11))
        xs.update((int(np.clip(ax1 - cw, 0, max_x)), int(np.clip(ax2, 0, max_x))))
        ys.update((int(np.clip(ay1 - ch, 0, max_y)), int(np.clip(ay2, 0, max_y))))

        area = float(cw * ch)
        area_fraction = area / base_area
        for x1 in xs:
            for y1 in ys:
                x2, y2 = x1 + cw, y1 + ch
                iw = max(0, min(x2, ax2) - max(x1, ax1))
                ih = max(0, min(y2, ay2) - max(y1, ay1))
                overlap_fraction = (iw * ih) / max(1.0, area)
                center_dist = ((x1 + cw / 2 - width / 2) ** 2 + (y1 + ch / 2 - height / 2) ** 2) ** 0.5
                center_penalty = 0.015 * (center_dist / max(1.0, (width ** 2 + height ** 2) ** 0.5))
                # Penalize webcam overlap, but not enough to sacrifice most of the gameplay.
                score = area_fraction * (1.0 - 2.0 * overlap_fraction) - center_penalty
                if score > best_score:
                    best_score = score
                    best_rect = (x1, y1, x2, y2)

    return best_rect


def crop_rect(frame: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = rect
    h, w = frame.shape[:2]
    x1 = int(np.clip(x1, 0, max(0, w - 2)))
    y1 = int(np.clip(y1, 0, max(0, h - 2)))
    x2 = int(np.clip(x2, x1 + 2, w))
    y2 = int(np.clip(y2, y1 + 2, h))
    return frame[y1:y2, x1:x2]


def render_visual(
    source: Path,
    silent_output: Path,
    start: float,
    end: float,
    width_out: int,
    height_out: int,
    detector,
    sample_interval: float,
    crf: int,
    preset: str,
    facecam_sharpen: float,
    facecam_corner: str,
) -> float:
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source video: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps < 1 or fps > 120:
        fps = 30.0
    source_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if source_w < 2 or source_h < 2:
        cap.release()
        raise RuntimeError("Source video has invalid dimensions.")

    face_rect, game_rect = detect_facecam_region(
        cap, detector, start, end, source_w, source_h, sample_interval, facecam_corner
    )
    log(f"  Facecam crop: {face_rect}; gameplay crop: {game_rect}; source: {source_w}x{source_h}")

    panel_h = height_out // 2
    # Ensure an even output width/height for H.264.
    width_out -= width_out % 2
    panel_h -= panel_h % 2
    height_out = panel_h * 2

    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width_out}x{height_out}", "-r", f"{fps:.6f}", "-i", "-",
        "-an", "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(silent_output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)
    frame_count = 0
    duration = end - start
    try:
        while frame_count / fps < duration:
            ok, frame = cap.read()
            if not ok:
                break
            top = crop_rect(frame, face_rect)
            bottom = crop_rect(frame, game_rect)
            top = cv2.resize(top, (width_out, panel_h), interpolation=cv2.INTER_LANCZOS4)
            bottom = cv2.resize(bottom, (width_out, panel_h), interpolation=cv2.INTER_LANCZOS4)
            if facecam_sharpen > 0:
                blurred = cv2.GaussianBlur(top, (0, 0), 1.0)
                top = cv2.addWeighted(top, 1.0 + facecam_sharpen, blurred, -facecam_sharpen, 0)
            stacked = np.vstack((top, bottom))
            process.stdin.write(stacked.tobytes())
            frame_count += 1
            if frame_count % max(1, int(fps * 5)) == 0:
                pct = min(100, int((frame_count / fps) / max(duration, 0.01) * 100))
                log(f"  Render progress: {pct}%")
    except BrokenPipeError:
        pass
    finally:
        cap.release()
        if process.stdin:
            process.stdin.close()

    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    return_code = process.wait()
    if return_code != 0 or frame_count == 0:
        raise RuntimeError(f"FFmpeg visual render failed (code {return_code}): {stderr[-3000:]}")
    log(f"  Rendered {frame_count} frames at {fps:.2f} fps.")
    return fps


def mux_original_audio(
    source: Path, silent_video: Path, final_video: Path, start: float, duration: float
) -> None:
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(silent_video), "-ss", f"{start:.3f}", "-i", str(source),
        "-map", "0:v:0", "-map", "1:a:0?", "-t", f"{duration:.3f}",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
        "-movflags", "+faststart", str(final_video),
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Audio mux failed: {result.stderr[-3000:]}")


def resolve_video_path(value: str, root: Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (root / candidate).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Original downloaded stream video")
    parser.add_argument("--manifest", required=True, type=Path, help="outputs/render_manifest.json")
    parser.add_argument("--width", type=int, default=1080)
    parser.add_argument("--height", type=int, default=1440)
    parser.add_argument("--sample-interval", type=float, default=0.75)
    parser.add_argument("--facecam-sharpen", type=float, default=0.22)
    parser.add_argument(
        "--facecam-corner",
        choices=("auto", "top-left", "top-right", "bottom-left", "bottom-right"),
        default="auto",
        help="Prefer the corner containing the streamer webcam; auto favors the usual lower corners.",
    )
    parser.add_argument("--crf", type=int, default=17)
    parser.add_argument("--preset", default="slow")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    args = parser.parse_args()

    source = args.source.resolve()
    manifest_path = args.manifest.resolve()
    if not source.is_file():
        raise RuntimeError(f"Original source video is missing: {source}")
    if not manifest_path.is_file():
        raise RuntimeError(f"Render manifest is missing: {manifest_path}")
    if args.width <= 0 or args.height <= 0 or args.width * 4 != args.height * 3:
        raise ValueError("Output dimensions must be 3:4, for example 1080x1440.")

    clips = load_manifest(manifest_path)
    detector = create_detector(args.model)
    success = 0
    try:
        for index, clip in enumerate(clips, 1):
            start = float(clip.get("start_time", 0))
            end = float(clip.get("end_time", 0))
            if end <= start:
                log(f"Skipping manifest entry {index}: invalid time range.")
                continue
            output = resolve_video_path(str(clip["video_path"]), manifest_path.parent.parent)
            if not output.parent.exists():
                output.parent.mkdir(parents=True, exist_ok=True)
            silent = output.with_name(output.stem + ".split_visual.mp4")
            muxed = output.with_name(output.stem + ".split_mux.mp4")
            log(f"\n[{index}/{len(clips)}] Split-rendering highlight rank {clip.get('rank', index)} ({start:.2f}s–{end:.2f}s)")
            render_visual(
                source, silent, start, end, args.width, args.height, detector,
                args.sample_interval, args.crf, args.preset, args.facecam_sharpen,
                args.facecam_corner,
            )
            mux_original_audio(source, silent, muxed, start, end - start)
            os.replace(muxed, output)
            silent.unlink(missing_ok=True)
            success += 1
            log(f"  Saved: {output}")
    finally:
        try:
            detector.close()
        except Exception:
            pass

    if success == 0:
        raise RuntimeError("No selected clips were successfully split-rendered.")
    log(f"\nCompleted: {success} clip(s) in 1080x1440 facecam/gameplay layout.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        log(f"ERROR: {exc}")
        raise SystemExit(1)

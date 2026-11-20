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


def _face_boxes(detector, frame: np.ndarray) -> list[tuple[float, float, float, float, float]]:
    """Return face boxes and confidence, checking an enlarged copy for small insets."""
    h, w = frame.shape[:2]
    detections: list[tuple[float, float, float, float, float]] = []
    # The webcam is often much smaller than game characters. A second, enlarged
    # pass improves detection of the inset without relying on its screen corner.
    for scale in (1.0, min(1.6, 2560.0 / max(w, h))):
        if scale <= 1.01:
            candidate = frame
        else:
            candidate = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        rgb = cv2.cvtColor(candidate, cv2.COLOR_BGR2RGB)
        result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        for detection in result.detections or []:
            box = detection.bounding_box
            x1 = float(box.origin_x) / scale
            y1 = float(box.origin_y) / scale
            x2 = float(box.origin_x + box.width) / scale
            y2 = float(box.origin_y + box.height) / scale
            score = float(detection.categories[0].score) if detection.categories else 0.0
            if x2 > x1 and y2 > y1:
                detections.append((x1, y1, x2, y2, score))

    # Merge duplicates from the original-size and enlarged passes.
    detections.sort(key=lambda item: item[4], reverse=True)
    merged: list[tuple[float, float, float, float, float]] = []
    for box in detections:
        x1, y1, x2, y2, _ = box
        area = max(1.0, (x2 - x1) * (y2 - y1))
        duplicate = False
        for kept in merged:
            kx1, ky1, kx2, ky2, _ = kept
            iw = max(0.0, min(x2, kx2) - max(x1, kx1))
            ih = max(0.0, min(y2, ky2) - max(y1, ky1))
            inter = iw * ih
            kept_area = max(1.0, (kx2 - kx1) * (ky2 - ky1))
            if inter / max(1.0, area + kept_area - inter) > 0.52:
                duplicate = True
                break
        if not duplicate:
            merged.append(box)
    return merged

def _clamp_rect(
    x1: float, y1: float, x2: float, y2: float, width: int, height: int
) -> tuple[int, int, int, int]:
    x1 = int(np.clip(round(x1), 0, width - 2))
    y1 = int(np.clip(round(y1), 0, height - 2))
    x2 = int(np.clip(round(x2), x1 + 2, width))
    y2 = int(np.clip(round(y2), y1 + 2, height))
    return x1, y1, x2, y2


def _rect_border_score(frame: np.ndarray, rect: tuple[int, int, int, int], edges: np.ndarray | None = None) -> float:
    """Measure whether a proposed webcam crop follows a visible inset-frame border."""
    x1, y1, x2, y2 = rect
    h, w = frame.shape[:2]
    if x2 - x1 < 24 or y2 - y1 < 24:
        return 0.0
    if edges is None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 55, 145)
    band = max(2, min(7, int(round(min(x2 - x1, y2 - y1) * 0.025))))
    inset = max(4, int(round(min(x2 - x1, y2 - y1) * 0.06)))
    densities: list[float] = []

    # Ignore edges that coincide with the video boundary: the webcam box may
    # deliberately touch the left/right/bottom edge of the livestream.
    if y1 > 1:
        roi = edges[max(0, y1 - band):min(h, y1 + band + 1), min(w, x1 + inset):max(0, x2 - inset)]
        if roi.size:
            densities.append(float(np.count_nonzero(roi)) / roi.size)
    if y2 < h - 1:
        roi = edges[max(0, y2 - band):min(h, y2 + band + 1), min(w, x1 + inset):max(0, x2 - inset)]
        if roi.size:
            densities.append(float(np.count_nonzero(roi)) / roi.size)
    if x1 > 1:
        roi = edges[min(h, y1 + inset):max(0, y2 - inset), max(0, x1 - band):min(w, x1 + band + 1)]
        if roi.size:
            densities.append(float(np.count_nonzero(roi)) / roi.size)
    if x2 < w - 1:
        roi = edges[min(h, y1 + inset):max(0, y2 - inset), max(0, x2 - band):min(w, x2 + band + 1)]
        if roi.size:
            densities.append(float(np.count_nonzero(roi)) / roi.size)

    if not densities:
        return 0.0
    # A clean border can have a high edge density, but cap the score so HUD
    # text or busy gameplay edges cannot dominate face tracking.
    return float(np.clip(np.mean(densities) / 0.20, 0.0, 1.0))


def _estimate_webcam_frame(
    frame: np.ndarray, face_box: tuple[float, float, float, float]
) -> tuple[tuple[int, int, int, int], float]:
    """Expand a tracked face to the surrounding webcam frame, using border evidence."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = face_box
    face_w = max(2.0, x2 - x1)
    face_h = max(2.0, y2 - y1)
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0

    # A streamer face generally occupies roughly one-third of the webcam
    # frame's height. Test several conventional overlay shapes and let visible
    # border lines select the best fit. The 16:10 / 3x-height prior is only a
    # fallback when the overlay has no visible border.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edge_map = cv2.Canny(gray, 55, 145)
    best_rect = None
    best_score = -1.0
    for aspect in (1.50, 1.60, 1.78, 1.91):
        for height_scale in (3.15, 3.30, 3.45, 3.60):
            cam_h = max(face_h * height_scale, face_w * 2.65)
            cam_w = cam_h * aspect
            for offset_x in (-0.06, 0.0, 0.06):
                for offset_y in (-0.07, 0.0, 0.07):
                    cx = center_x + cam_w * offset_x
                    cy = center_y + cam_h * offset_y
                    rect = _clamp_rect(
                        cx - cam_w / 2.0,
                        cy - cam_h / 2.0,
                        cx + cam_w / 2.0,
                        cy + cam_h / 2.0,
                        width,
                        height,
                    )
                    actual_w = rect[2] - rect[0]
                    actual_h = rect[3] - rect[1]
                    if actual_w < face_w * 2.5 or actual_h < face_h * 2.2:
                        continue
                    border = _rect_border_score(frame, rect, edge_map)
                    prior = np.exp(
                        -abs(aspect - 1.60) * 1.6
                        -abs(height_scale - 3.30) * 0.70
                        -abs(offset_x) * 2.0
                        -abs(offset_y) * 2.0
                    )
                    score = 1.30 * border + 0.70 * float(prior)
                    if score > best_score:
                        best_score = score
                        best_rect = rect

    if best_rect is None:
        raise RuntimeError("Could not estimate the webcam frame around the detected face.")
    return best_rect, max(0.0, best_score)


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
    """Find the streamer inset by tracking persistent faces and locating its frame.

    The game may contain many faces. The webcam candidate is selected using
    temporal position/size consistency, small face scale, screen-edge location,
    and evidence of a rectangular inset border. No corner is assumed.
    """
    corners = ("top-left", "top-right", "bottom-left", "bottom-right")
    if facecam_corner not in ("auto", *corners):
        raise ValueError(f"Unsupported facecam corner: {facecam_corner}")
    if detector is None:
        raise RuntimeError("Face detector was not initialized; install mediapipe and retry.")

    duration = max(0.0, end - start)
    interval = max(0.25, sample_interval)
    sample_count = min(90, max(3, int(np.ceil(duration / interval))))
    sample_times = np.linspace(start + min(0.15, duration * 0.05), max(start, end - 0.15), sample_count)
    tracks: list[dict[str, Any]] = []
    valid_sample_count = 0

    for sample_index, timestamp in enumerate(sample_times):
        cap.set(cv2.CAP_PROP_POS_MSEC, float(timestamp) * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        valid_sample_count += 1
        boxes = _face_boxes(detector, frame)
        # Reject impossible detections and very large gameplay close-ups.
        usable = []
        for x1, y1, x2, y2, confidence in boxes:
            fw, fh = x2 - x1, y2 - y1
            area_ratio = (fw * fh) / float(width * height)
            aspect = fw / max(fh, 1.0)
            if fw < 18 or fh < 18 or area_ratio > 0.085 or not 0.52 <= aspect <= 1.65:
                continue
            if facecam_corner != "auto":
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                in_corner = {
                    "top-left": cx < width * 0.48 and cy < height * 0.48,
                    "top-right": cx > width * 0.52 and cy < height * 0.48,
                    "bottom-left": cx < width * 0.48 and cy > height * 0.52,
                    "bottom-right": cx > width * 0.52 and cy > height * 0.52,
                }[facecam_corner]
                if not in_corner:
                    continue
            usable.append((x1, y1, x2, y2, confidence))

        # Associate detections across sampled frames. A webcam face can move
        # within its box, so use a tolerant center-distance gate plus scale.
        assigned: set[int] = set()
        for detection in usable:
            x1, y1, x2, y2, confidence = detection
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            fw, fh = x2 - x1, y2 - y1
            best_track, best_distance = None, float("inf")
            for track_index, track in enumerate(tracks):
                if track_index in assigned:
                    continue
                last = track["items"][-1]
                lx1, ly1, lx2, ly2, _ = last["box"]
                lcx, lcy = (lx1 + lx2) / 2.0, (ly1 + ly2) / 2.0
                lfh = max(1.0, ly2 - ly1)
                distance = float(np.hypot((cx - lcx) / width, (cy - lcy) / height))
                scale_change = abs(np.log(max(1.0, fh) / lfh))
                if distance < 0.135 and scale_change < 0.85 and distance < best_distance:
                    best_track, best_distance = track_index, distance
            item = {"box": detection, "sample": sample_index}
            if best_track is None:
                tracks.append({"items": [item], "representative_frame": frame.copy()})
                assigned.add(len(tracks) - 1)
            else:
                tracks[best_track]["items"].append(item)
                assigned.add(best_track)

    if not tracks:
        raise RuntimeError(
            "AI could not find a facecam face in the selected highlight. "
            "The source may be too small/blurry, or the facecam may be hidden."
        )

    total_samples = max(1, valid_sample_count)
    scored_tracks = []
    for track in tracks:
        items = track["items"]
        if len(items) < max(2, int(np.ceil(total_samples * 0.12))):
            continue
        boxes = np.array([item["box"][:4] for item in items], dtype=np.float32)
        centers = np.column_stack(((boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2))
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        median_box = tuple(float(v) for v in np.median(boxes, axis=0))
        center = np.median(centers, axis=0)
        edge_distance = min(center[0] / width, 1.0 - center[0] / width,
                            center[1] / height, 1.0 - center[1] / height)
        area_ratio = float(np.median(widths * heights) / (width * height))
        persistence = len(items) / total_samples
        center_jitter = float(np.mean(np.linalg.norm((centers - center) / np.array([width, height]), axis=1)))
        stable = float(np.exp(-center_jitter / 0.055))
        edge_score = float(np.clip(1.0 - edge_distance / 0.36, 0.0, 1.0))
        small_face_score = float(np.clip((0.025 - area_ratio) / 0.025, 0.0, 1.0))
        confidence = float(np.mean([item["box"][4] for item in items]))
        representative = track["representative_frame"]
        frame_rect, border_score = _estimate_webcam_frame(representative, median_box)
        score = (5.0 * persistence + 1.35 * edge_score + 1.05 * small_face_score
                 + 0.9 * stable + 1.35 * border_score + 0.35 * confidence)
        scored_tracks.append((score, track, median_box, frame_rect, persistence, edge_score, border_score))

    if not scored_tracks:
        raise RuntimeError(
            "Face detections were too brief to track a webcam frame. "
            "Try a longer highlight or lower --sample-interval."
        )

    scored_tracks.sort(key=lambda entry: entry[0], reverse=True)
    best = scored_tracks[0]
    if best[0] < 3.0 or best[4] < 0.18:
        raise RuntimeError(
            "No reliable streamer webcam frame was found. Refusing to crop a random game character. "
            "Try --sample-interval 0.4 or set --facecam-corner to the known webcam corner."
        )

    face_rect = best[3]
    game_rect = choose_gameplay_crop(width, height, face_rect)
    log(
        "  AI webcam detection: "
        f"facecam={face_rect}, detections={len(best[1]['items'])}/{total_samples}, "
        f"edge={best[5]:.2f}, frame-border={best[6]:.2f}, score={best[0]:.2f}"
    )
    return face_rect, game_rect

def choose_gameplay_crop(
    width: int, height: int, avoid_rect: tuple[int, int, int, int]
) -> tuple[int, int, int, int]:
    """Find the largest 3:2 gameplay crop that avoids the detected webcam frame."""
    ax1, ay1, ax2, ay2 = avoid_rect
    target_ratio = 1.5
    if width / max(height, 1) >= target_ratio:
        max_h = height
        max_w = int(round(height * target_ratio))
    else:
        max_w = width
        max_h = int(round(width / target_ratio))
    max_area = float(max_w * max_h)

    candidates: list[tuple[float, tuple[int, int, int, int]]] = []
    # Add exact clear-space sizes for the four sides of the webcam box. This
    # avoids relying only on a coarse scale grid when the ideal width is, for
    # example, exactly the space to the right of a bottom-left facecam.
    scales = list(np.linspace(1.0, 0.56, 12))
    clear_widths = (ax1, width - ax2)
    clear_heights = (ay1, height - ay2)
    scales.extend(min(max_w, max(0, clear_w)) / max_w for clear_w in clear_widths)
    scales.extend(min(max_h, max(0, clear_h)) / max_h for clear_h in clear_heights)
    scales = sorted({float(np.clip(scale, 0.56, 1.0)) for scale in scales}, reverse=True)

    # Permit a slightly smaller crop when that is what it takes to remove the
    # webcam overlay completely. This is better than leaving a second tiny
    # streamer window inside the gameplay panel.
    for scale in scales:
        crop_w = max(2, min(width, int(round(max_w * float(scale)))))
        crop_h = max(2, min(height, int(round(crop_w / target_ratio))))
        if crop_w / crop_h > target_ratio + 0.01:
            crop_w = int(round(crop_h * target_ratio))
        max_x = max(0, width - crop_w)
        max_y = max(0, height - crop_h)
        x_steps = min(25, max(2, max_x + 1))
        y_steps = min(25, max(2, max_y + 1))
        xs = set(int(x) for x in np.linspace(0, max_x, x_steps))
        ys = set(int(y) for y in np.linspace(0, max_y, y_steps))
        # Add exact alignment positions beside the webcam frame. A coarse grid
        # alone could leave a thin strip of the overlay in the gameplay crop.
        xs.update(int(np.clip(x, 0, max_x)) for x in (ax1 - crop_w, ax2))
        ys.update(int(np.clip(y, 0, max_y)) for y in (ay1 - crop_h, ay2))
        xs = sorted(xs)
        ys = sorted(ys)
        area_fraction = (crop_w * crop_h) / max_area
        area_loss = 1.0 - area_fraction

        for x1 in xs:
            for y1 in ys:
                rect = (x1, y1, x1 + crop_w, y1 + crop_h)
                ix = max(0, min(ax2, rect[2]) - max(ax1, rect[0]))
                iy = max(0, min(ay2, rect[3]) - max(ay1, rect[1]))
                overlap_fraction = (ix * iy) / float(crop_w * crop_h)
                center_penalty = (
                    abs((x1 + crop_w / 2) - width / 2) / max(width, 1)
                    + abs((y1 + crop_h / 2) - height / 2) / max(height, 1)
                )
                score = 8.0 * overlap_fraction + 0.65 * area_loss + 0.015 * center_penalty
                candidates.append((score, rect))

    return min(candidates, key=lambda item: item[0])[1]

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
        help="Auto-detect the streamer webcam frame; optionally restrict to a corner.",
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
        if detector is not None:
            detector.close()

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

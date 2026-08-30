"""
Standalone GoPro video rectification pipeline.

this file only rectifies videos, nothing else.

Accepts either a single video file or a folder of videos as input. All the
knobs that used to be loose module-level variables in main.py now live on
one RunConfig dataclass, so a run is fully described by one object instead
of a dozen scattered globals.

Usage:
    python rectify_pipeline.py "path\\to\\video.mp4"
    python rectify_pipeline.py "path\\to\\folder_of_videos"
    python rectify_pipeline.py                      # uses the defaults in RunConfig at the bottom
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
from tqdm import tqdm

import general_useful_functions as guf
from general_useful_functions import dprint

SCRIPT_DIR = Path(__file__).resolve().parent
CAMERA_MODEL_JSON_PATH = SCRIPT_DIR / "gopro_DG_rectilinear_model_lrv.json"

# Allow tiny height/width differences between the JSON model and the actual video frames.
SIZE_TOLERANCE_PX = 4
ASPECT_TOLERANCE = 0.01
ALLOW_RESCALED_INPUT = True  # GoPro videos are often a different resolution than the LRV calibration model.

# Margin used only for the valid remap area.
VALID_BORDER_MARGIN_PX = 1.0


# ============================================================
# GoPro camera model math (copied from 04_analysis_and_plots_from_saved_sam2_V1_58.py
# by way of test.py)
# ============================================================

def _poly_eval(coeffs, r):
    """Evaluate c0 + c1*r + c2*r^2 + ... for scalar or array r."""
    r = np.asarray(r, dtype=np.float64)
    out = np.zeros_like(r, dtype=np.float64)
    power = np.ones_like(r, dtype=np.float64)

    for c in coeffs:
        out += float(c) * power
        power *= r

    return out


def _pad_coeffs(values, length):
    """Pad a coefficient list with zeros to the requested length."""
    out = list(values)
    while len(out) < length:
        out.append(0.0)
    return out[:length]


def _prepare_poly_inverse_lookup(poly):
    """
    POLY maps normalized GoPro radius r to angle theta. For undistortion we
    need r(theta), so we interpolate over the monotonic part of POLY.
    """
    r_grid = np.linspace(0.0, 1.30, 10001, dtype=np.float64)
    theta_grid = _poly_eval(poly, r_grid)

    diffs = np.diff(theta_grid)
    bad = np.where(diffs <= 0)[0]
    good_until = len(theta_grid) if len(bad) == 0 else int(bad[0] + 1)

    r_grid = r_grid[:good_until]
    theta_grid = theta_grid[:good_until]
    theta_max = float(theta_grid[-1])

    def inverse(theta):
        theta = np.asarray(theta, dtype=np.float64)
        theta_clipped = np.clip(theta, 0.0, theta_max)
        return np.interp(theta_clipped, theta_grid, r_grid)

    return inverse, theta_max


def _gopro_map_forward(x, y, mapx, mapy):
    """GoPro MAPX/MAPY polynomial forward map."""
    mx = _pad_coeffs(mapx, 3)
    my = _pad_coeffs(mapy, 6)

    ax1, ax3, ax5 = mx
    ay1, ay3, ay5, ay1x2, ay3x2, ay1x4 = my

    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    x5 = x4 * x

    y2 = y * y
    y3 = y2 * y
    y5 = y3 * y2

    new_x = ax1 * x + ax3 * x3 + ax5 * x5
    new_y = (
        ay1 * y
        + ay3 * y3
        + ay5 * y5
        + ay1x2 * y * x2
        + ay3x2 * y3 * x2
        + ay1x4 * y * x4
    )

    return new_x, new_y


def _gopro_map_inverse(xt, yt, mapx, mapy, n_iter=10):
    """Invert GoPro MAPX/MAPY with Newton iterations."""
    x = np.asarray(xt, dtype=np.float64).copy()
    y = np.asarray(yt, dtype=np.float64).copy()

    mx = _pad_coeffs(mapx, 3)
    my = _pad_coeffs(mapy, 6)

    ax1, ax3, ax5 = mx
    ay1, ay3, ay5, ay1x2, ay3x2, ay1x4 = my

    for _ in range(n_iter):
        fx, fy = _gopro_map_forward(x, y, mx, my)
        rx = fx - xt
        ry = fy - yt

        x2 = x * x
        x3 = x2 * x
        x4 = x2 * x2
        y2 = y * y
        y3 = y2 * y
        y4 = y2 * y2

        j11 = ax1 + 3.0 * ax3 * x2 + 5.0 * ax5 * x4
        j21 = 2.0 * ay1x2 * y * x + 2.0 * ay3x2 * y3 * x + 4.0 * ay1x4 * y * x3
        j22 = ay1 + 3.0 * ay3 * y2 + 5.0 * ay5 * y4 + ay1x2 * x2 + 3.0 * ay3x2 * y2 * x2 + ay1x4 * x4

        det = j11 * j22
        det = np.where(np.abs(det) < 1e-12, np.nan, det)

        dx = rx / j11
        dy = (j11 * ry - rx * j21) / det
        dx = np.nan_to_num(dx, nan=0.0, posinf=0.0, neginf=0.0)
        dy = np.nan_to_num(dy, nan=0.0, posinf=0.0, neginf=0.0)

        x -= dx
        y -= dy

    return x, y


def _scale_pixel_coordinate(value, scale):
    """Pixel-center preserving coordinate scaling."""
    return (float(value) + 0.5) * float(scale) - 0.5


def _scale_rectilinear_K(K, sx, sy):
    """Scale the calibrated rectilinear_output_K to a same-aspect resized image."""
    K = np.asarray(K, dtype=np.float64).copy()
    K_scaled = K.copy()

    K_scaled[0, 0] *= float(sx)
    K_scaled[0, 1] *= float(sx)
    K_scaled[0, 2] = _scale_pixel_coordinate(K[0, 2], sx)

    K_scaled[1, 0] *= float(sy)
    K_scaled[1, 1] *= float(sy)
    K_scaled[1, 2] = _scale_pixel_coordinate(K[1, 2], sy)

    K_scaled[2, :] = [0.0, 0.0, 1.0]
    return K_scaled


def load_gopro_json_model(json_path):
    """Load and validate the D&G-ready GoPro rectilinear JSON model."""
    json_path = Path(json_path)

    if not json_path.exists():
        raise FileNotFoundError(
            f"GoPro D&G-ready JSON camera model was not found:\n{json_path}\n\n"
            "Put gopro_DG_rectilinear_model_lrv.json next to this script, "
            "or edit CAMERA_MODEL_JSON_PATH at the top of the script."
        )

    with open(json_path, "r", encoding="utf-8") as f:
        model = json.load(f)

    required = [
        "width", "height", "ZFOV_deg", "MAPX", "MAPY", "POLY",
        "rectilinear_output_K", "scale_x_px_per_model_unit", "scale_y_px_per_model_unit",
    ]

    missing = [k for k in required if k not in model]
    if missing:
        raise KeyError(
            "The JSON model is missing required fields for the D&G-ready GoPro model: "
            f"{missing}"
        )

    if "source_cx" not in model and "cx" not in model:
        raise KeyError("The JSON model must contain source_cx/source_cy, or old-style cx/cy.")
    if "source_cy" not in model and "cy" not in model:
        raise KeyError("The JSON model must contain source_cx/source_cy, or old-style cx/cy.")

    model["width"] = int(model["width"])
    model["height"] = int(model["height"])
    model["output_width"] = int(model.get("output_width", model["width"]))
    model["output_height"] = int(model.get("output_height", model["height"]))
    model["source_cx"] = float(model.get("source_cx", model.get("cx")))
    model["source_cy"] = float(model.get("source_cy", model.get("cy")))
    model["ZFOV_deg"] = float(model["ZFOV_deg"])

    model["scale_x_px_per_model_unit"] = float(model["scale_x_px_per_model_unit"])
    model["scale_y_px_per_model_unit"] = float(model["scale_y_px_per_model_unit"])
    model["scale_px_per_model_unit"] = float(
        model.get(
            "scale_px_per_model_unit",
            0.5 * (model["scale_x_px_per_model_unit"] + model["scale_y_px_per_model_unit"]),
        )
    )

    model["POLY"] = [float(v) for v in model["POLY"]]
    model["MAPX"] = [float(v) for v in model["MAPX"]]
    model["MAPY"] = [float(v) for v in model["MAPY"]]
    model["rectilinear_output_K"] = np.asarray(model["rectilinear_output_K"], dtype=np.float64)

    if model["rectilinear_output_K"].shape != (3, 3):
        raise ValueError("rectilinear_output_K must be a 3 x 3 matrix.")

    map_direction = str(model.get("map_direction", "inverse")).lower()
    if map_direction != "inverse":
        raise ValueError(
            "This script is locked to the GoPro metadata inverse model. "
            f"Expected map_direction='inverse', got {model.get('map_direction')!r}."
        )

    return model


def scale_gopro_model_to_image(model, image_width, image_height):
    """
    Scale JSON model coordinates to the actual frame size.

    Exact model size is preferred. If the frames are a same-aspect (or close)
    resized copy of the model resolution, the source center, model-unit scale,
    and calibrated rectilinear_output_K are scaled too.
    """
    model_width = int(model["width"])
    model_height = int(model["height"])

    dw = int(image_width) - model_width
    dh = int(image_height) - model_height

    if abs(dw) <= SIZE_TOLERANCE_PX and abs(dh) <= SIZE_TOLERANCE_PX:
        sx = float(image_width) / float(model_width)
        sy = float(image_height) / float(model_height)
        size_message = "exact or tiny size mismatch"
    else:
        model_aspect = model_width / model_height
        image_aspect = float(image_width) / float(image_height)
        aspect_diff = abs(image_aspect - model_aspect)

        if not ALLOW_RESCALED_INPUT or aspect_diff > ASPECT_TOLERANCE:
            raise ValueError(
                f"Frame size is {image_width}x{image_height}, but JSON model expects "
                f"{model_width}x{model_height}.\n"
                f"Aspect difference = {aspect_diff:.6f}.\n"
                "Use a video captured at the same GoPro resolution/FOV as the calibration model, "
                "or a same-aspect resized copy."
            )

        sx = float(image_width) / float(model_width)
        sy = float(image_height) / float(model_height)
        size_message = "same-aspect resized frame"

    source_cx_scaled = _scale_pixel_coordinate(model["source_cx"], sx)
    source_cy_scaled = _scale_pixel_coordinate(model["source_cy"], sy)
    K_scaled = _scale_rectilinear_K(model["rectilinear_output_K"], sx, sy)

    scaled = dict(model)
    scaled["image_width"] = int(image_width)
    scaled["image_height"] = int(image_height)
    scaled["sx"] = float(sx)
    scaled["sy"] = float(sy)
    scaled["source_cx_scaled"] = float(source_cx_scaled)
    scaled["source_cy_scaled"] = float(source_cy_scaled)
    scaled["scale_x_px_per_model_unit"] = float(model["scale_x_px_per_model_unit"]) * sx
    scaled["scale_y_px_per_model_unit"] = float(model["scale_y_px_per_model_unit"]) * sy
    scaled["rectilinear_output_K_scaled"] = K_scaled
    scaled["size_message"] = size_message

    return scaled


def _gopro_source_map_from_rectilinear_output_K(scaled, image_width, image_height):
    """
    Build cv2.remap source coordinates:
        output pixel -> rectilinear_output_K ray -> POLY inverse radius ->
        inverse MAPX/MAPY -> encoded source pixel.
    """
    xs = np.arange(image_width, dtype=np.float64)
    ys = np.arange(image_height, dtype=np.float64)

    u, v = np.meshgrid(xs, ys)

    K = np.asarray(scaled["rectilinear_output_K_scaled"], dtype=np.float64)
    inv_K = np.linalg.inv(K)

    pixel_h = np.stack([u.ravel(), v.ravel(), np.ones(u.size, dtype=np.float64)], axis=0)
    ray = inv_K @ pixel_h
    x = (ray[0] / ray[2]).reshape(u.shape)
    y = (ray[1] / ray[2]).reshape(u.shape)

    rho = np.sqrt(x * x + y * y)
    theta = np.arctan(rho)

    inverse_poly, _ = _prepare_poly_inverse_lookup(scaled["POLY"])
    r_gopro = inverse_poly(theta)

    dir_x = np.divide(x, rho, out=np.zeros_like(x), where=rho > 1e-12)
    dir_y = np.divide(y, rho, out=np.zeros_like(y), where=rho > 1e-12)

    x_model = r_gopro * dir_x
    y_model = r_gopro * dir_y

    x_model, y_model = _gopro_map_inverse(
        x_model,
        y_model,
        scaled["MAPX"],
        scaled["MAPY"],
    )

    map1 = scaled["source_cx_scaled"] + x_model * scaled["scale_x_px_per_model_unit"]
    map2 = scaled["source_cy_scaled"] + y_model * scaled["scale_y_px_per_model_unit"]

    margin = float(VALID_BORDER_MARGIN_PX)
    valid = (
        np.isfinite(map1)
        & np.isfinite(map2)
        & (map1 >= margin)
        & (map1 <= image_width - 1 - margin)
        & (map2 >= margin)
        & (map2 <= image_height - 1 - margin)
    )

    return map1, map2, valid


def build_gopro_json_undistort_map(model, image_width, image_height):
    """
    Build cv2.remap maps for the hybrid GoPro inverse metadata + K model.

    Output image size equals the input frame size. Never auto-zooms or
    invents a K; uses rectilinear_output_K from the JSON model (scaled only
    if the frames differ in size from the calibration model).
    """
    scaled = scale_gopro_model_to_image(model, image_width, image_height)

    map1, map2, valid = _gopro_source_map_from_rectilinear_output_K(
        scaled, image_width, image_height,
    )

    map1 = np.where(valid, map1, -1.0).astype(np.float32)
    map2 = np.where(valid, map2, -1.0).astype(np.float32)

    valid_fraction = float(np.mean(valid))

    return map1, map2, valid_fraction


def _gopro_forward_normalized_from_source_pixels(scaled, source_x, source_y):
    """
    Forward-project raw/source GoPro pixel coordinates to normalized
    rectilinear coordinates (pre-K, i.e. x = X/Z, y = Y/Z of the pinhole ray).

    This is the opposite direction of _gopro_source_map_from_rectilinear_output_K
    (which goes output pixel -> source pixel). It does not depend on any K, so
    it can be used to figure out how "wide" a K/canvas needs to be to show a
    given source pixel without cropping it.
    """
    source_x = np.asarray(source_x, dtype=np.float64)
    source_y = np.asarray(source_y, dtype=np.float64)

    x_enc = (source_x - scaled["source_cx_scaled"]) / scaled["scale_x_px_per_model_unit"]
    y_enc = (source_y - scaled["source_cy_scaled"]) / scaled["scale_y_px_per_model_unit"]

    x_model, y_model = _gopro_map_forward(x_enc, y_enc, scaled["MAPX"], scaled["MAPY"])

    r_model = np.sqrt(x_model * x_model + y_model * y_model)
    theta = _poly_eval(scaled["POLY"], r_model)
    rho = np.tan(theta)

    dir_x = np.divide(x_model, r_model, out=np.zeros_like(x_model), where=r_model > 1e-12)
    dir_y = np.divide(y_model, r_model, out=np.zeros_like(y_model), where=r_model > 1e-12)

    x_rect = rho * dir_x
    y_rect = rho * dir_y

    return x_rect, y_rect


def _estimate_widen_canvas(scaled, image_width, image_height, edge_samples=400, crop_percentile=1.0):
    """
    Compute how wide (and how tall, before any aspect-ratio cropping) the
    canvas needs to be to show the undistorted content at the calibrated
    (unscaled) focal length -- i.e. widening the canvas instead of zooming
    out the K, since rectifying a wide-angle lens naturally produces a wider
    image.
    """
    xs = np.linspace(0.0, image_width - 1, edge_samples)
    ys = np.linspace(0.0, image_height - 1, edge_samples)

    left_right_x = np.concatenate([np.zeros(edge_samples), np.full(edge_samples, image_width - 1)])
    left_right_y = np.concatenate([ys, ys])
    top_bottom_x = np.concatenate([xs, xs])
    top_bottom_y = np.concatenate([np.zeros(edge_samples), np.full(edge_samples, image_height - 1)])

    x_rect_lr, _ = _gopro_forward_normalized_from_source_pixels(scaled, left_right_x, left_right_y)
    _, y_rect_tb = _gopro_forward_normalized_from_source_pixels(scaled, top_bottom_x, top_bottom_y)

    x_rect_lr = x_rect_lr[np.isfinite(x_rect_lr)]
    y_rect_tb = y_rect_tb[np.isfinite(y_rect_tb)]

    keep_pct = 100.0 - float(crop_percentile)
    max_x = float(np.percentile(np.abs(x_rect_lr), keep_pct)) if x_rect_lr.size else 1.0
    max_y = float(np.percentile(np.abs(y_rect_tb), keep_pct)) if y_rect_tb.size else 1.0

    K = np.asarray(scaled["rectilinear_output_K_scaled"], dtype=np.float64)
    fx = float(K[0, 0])
    fy = float(K[1, 1])

    natural_width = int(round(2.0 * fx * max(max_x, 1e-6)))
    natural_height = int(round(2.0 * fy * max(max_y, 1e-6)))

    return natural_width, natural_height


def build_gopro_zhangK_undistort_map_widen(model, image_width, image_height,
                                            target_aspect_ratio=None, crop_percentile=1.0):
    """
    "GoPro inverse metadata + Zhang K" undistortion that widens the output
    canvas instead of zooming out, because rectifying a wide-angle lens
    naturally produces a wider image. The focal length (K) is kept exactly
    as calibrated -- only the canvas is enlarged, so the extra horizontal
    (and vertical) content the wide-angle rectification produces is shown
    instead of being cropped or shrunk away.

    If the resulting "natural" (fully widened) canvas is taller than
    target_aspect_ratio allows, the top and bottom are center-cropped down
    to that ratio. If target_aspect_ratio is None, the input video's own
    width/height ratio is used, so only the widening from undistortion
    changes the frame -- no arbitrary reshaping. crop_percentile still
    allows a small amount of the extreme corners to fall outside the canvas
    (a little side cropping is okay).
    """
    scaled = scale_gopro_model_to_image(model, image_width, image_height)

    natural_width, natural_height = _estimate_widen_canvas(
        scaled, image_width, image_height, crop_percentile=crop_percentile,
    )

    if target_aspect_ratio is None:
        target_aspect_ratio = float(image_width) / float(image_height)

    output_width = natural_width
    output_height = int(round(output_width / float(target_aspect_ratio)))

    if output_height > natural_height:
        # The target ratio wants more height than the widened content
        # naturally has; fall back to the natural height instead of
        # inventing content that was never undistorted.
        output_height = natural_height

    K = np.asarray(scaled["rectilinear_output_K_scaled"], dtype=np.float64).copy()
    K[0, 2] = output_width / 2.0 - 0.5
    K[1, 2] = output_height / 2.0 - 0.5

    scaled = dict(scaled)
    scaled["rectilinear_output_K_scaled"] = K

    map1, map2, valid = _gopro_source_map_from_rectilinear_output_K(
        scaled, output_width, output_height,
    )

    map1 = np.where(valid, map1, -1.0).astype(np.float32)
    map2 = np.where(valid, map2, -1.0).astype(np.float32)

    valid_fraction = float(np.mean(valid))

    return map1, map2, valid_fraction, output_width, output_height


# ============================================================
# Per-video rectification functions
# ============================================================

def rectify_video(input_video_path, output_video_path=None):
    """
    Read a raw GoPro video, undistort ("flatten") every frame using the
    gopro_DG_rectilinear_model_lrv.json camera model, and save the result
    as a new video. Output is the same size as the input, K unchanged, no
    zoom -- the original, faithful rectification.

    Returns the output video path.
    """
    input_video_path = Path(input_video_path)
    if not input_video_path.exists():
        raise FileNotFoundError(f"Input video was not found: {input_video_path}")

    if output_video_path is None:
        output_video_path = input_video_path.with_name(
            f"{input_video_path.stem}_rectified{input_video_path.suffix}"
        )
    output_video_path = Path(output_video_path)

    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_video_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    dprint(f"  Input video: {input_video_path}\t Size: {frame_width} x {frame_height}, FPS: {fps:.3f}, frames: {total_frames}")

    model = load_gopro_json_model(CAMERA_MODEL_JSON_PATH)
    map1, map2, valid_fraction = build_gopro_json_undistort_map(
        model, image_width=frame_width, image_height=frame_height,
    )
    dprint(f"  Valid remap fraction: {valid_fraction:.3f}")

    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (frame_width, frame_height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video writer for: {output_video_path}")

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rectified = cv2.remap(
            frame,
            map1,
            map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        writer.write(rectified)

        frame_idx += 1
        if frame_idx % 100 == 0:
            dprint(f"  Processed {frame_idx} frames...")

    cap.release()
    writer.release()

    dprint(f"Saved rectified video to: {output_video_path}")
    return output_video_path


def rectify_video_widen(input_video_path, output_video_path=None,
                         target_aspect_ratio=None, crop_percentile=1.0,
                         crop_top_px=0, crop_bottom_px=0, crop_left_px=0, crop_right_px=0):
    """
    Undistort a raw GoPro video by widening the output canvas instead of
    zooming out the K. The calibrated focal length is kept as-is, so the
    center of the frame looks exactly like the normal rectification; the
    canvas is just made wider (and, if needed, cropped top/bottom to
    target_aspect_ratio) to fit the extra horizontal content a wide-angle
    undistortion naturally produces.

    target_aspect_ratio: desired output width/height ratio (e.g. 16/9). If
    None, uses the input video's own aspect ratio.
    crop_percentile: percentage of the most extreme border points (corners)
    allowed to fall outside the canvas -- a little side/corner cropping is
    accepted rather than inventing more canvas.
    crop_top_px / crop_bottom_px / crop_left_px / crop_right_px: manual crop
    bars, in pixels, cut from each side of the widened canvas AFTER it is
    built (e.g. to trim leftover black border or unwanted edge content by an
    exact amount). Applied on top of target_aspect_ratio/crop_percentile, not
    instead of them.

    Returns the output video path.
    """
    input_video_path = Path(input_video_path)
    if not input_video_path.exists():
        raise FileNotFoundError(f"Input video was not found: {input_video_path}")

    if output_video_path is None:
        output_video_path = input_video_path.with_name(
            f"{input_video_path.stem}_rectified_widened{input_video_path.suffix}"
        )
    output_video_path = Path(output_video_path)

    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_video_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    dprint(f"  Input video: {input_video_path}\t Size: {frame_width} x {frame_height}, FPS: {fps:.3f}, frames: {total_frames}")

    model = load_gopro_json_model(CAMERA_MODEL_JSON_PATH)
    map1, map2, valid_fraction, output_width, output_height = build_gopro_zhangK_undistort_map_widen(
        model,
        image_width=frame_width,
        image_height=frame_height,
        target_aspect_ratio=target_aspect_ratio,
        crop_percentile=crop_percentile,
    )
    dprint(f"  Output (widened) size: {output_width} x {output_height}\t Valid remap fraction: {valid_fraction:.3f}")

    crop_top_px = max(0, int(crop_top_px))
    crop_bottom_px = max(0, int(crop_bottom_px))
    crop_left_px = max(0, int(crop_left_px))
    crop_right_px = max(0, int(crop_right_px))

    if crop_top_px or crop_bottom_px or crop_left_px or crop_right_px:
        y0 = crop_top_px
        y1 = output_height - crop_bottom_px
        x0 = crop_left_px
        x1 = output_width - crop_right_px

        if y1 <= y0 or x1 <= x0:
            cap.release()
            raise ValueError(
                "Requested crop bars are too large for the "
                f"{output_width}x{output_height} widened canvas: "
                f"top={crop_top_px}, bottom={crop_bottom_px}, "
                f"left={crop_left_px}, right={crop_right_px}"
            )

        map1 = map1[y0:y1, x0:x1]
        map2 = map2[y0:y1, x0:x1]
        output_width = x1 - x0
        output_height = y1 - y0

        dprint(f"  Output (widened, manually cropped) size: {output_width} x {output_height}")

    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (output_width, output_height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video writer for: {output_video_path}")

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rectified = cv2.remap(
            frame,
            map1,
            map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        writer.write(rectified)

        frame_idx += 1
        if frame_idx % 100 == 0:
            dprint(f"  Processed {frame_idx} frames...")

    cap.release()
    writer.release()

    dprint(f"Saved widened rectified video to: {output_video_path}")
    return output_video_path


# ============================================================
# Run configuration + folder/single-video orchestration
# ============================================================

@dataclass
class RunConfig:
    """
    Everything needed to describe one rectification run, in one place
    instead of a dozen loose module-level variables.
    """
    # --- input/output ---
    input_path: Path                      # a single video file OR a folder of videos
    output_folder: Path                   # where the rectified video(s) are saved
    overwrite_existing: bool = False      # re-rectify + overwrite existing outputs, or skip them

    # --- which rectification function to use ---
    # "widen" (default) -> rectify_video_widen; "original" -> rectify_video (untouched, known-good).
    rectify_mode: str = "widen"

    # --- rectify_video_widen-specific parameters (ignored when rectify_mode="original") ---
    #I CHOSE THIS VALUES, I THINK THEY ARE THE BEST
    target_aspect_ratio: Optional[float] = None
    crop_percentile: float = 1.0
    crop_top_px: int = 145
    crop_bottom_px: int = 100
    crop_left_px: int = 300
    crop_right_px: int = 300

    # --- misc / local config ---
    video_extensions: frozenset = field(default_factory=lambda: frozenset({".mp4", ".mov", ".avi", ".mkv"}))
    tqdm_enabled: bool = True
    timing_enabled: bool = True
    stats_enabled: bool = True
    dprint_enabled: bool = False

    @property
    def rectify_function(self) -> Callable:
        if self.rectify_mode == "original":
            return rectify_video
        if self.rectify_mode == "widen":
            return rectify_video_widen
        raise ValueError(f"Unknown rectify_mode: {self.rectify_mode!r} (expected 'widen' or 'original')")


def _apply_global_toggles(config: RunConfig):
    guf.TIMING_ENABLED = config.timing_enabled
    guf.STATS_ENABLED = config.stats_enabled
    guf.DPRINT_ENABLED = config.dprint_enabled


def _rectify_one(config: RunConfig, video_path: Path, output_path: Path):
    if config.rectify_mode == "widen":
        rectify_video_widen(
            video_path,
            output_video_path=output_path,
            target_aspect_ratio=config.target_aspect_ratio,
            crop_percentile=config.crop_percentile,
            crop_top_px=config.crop_top_px,
            crop_bottom_px=config.crop_bottom_px,
            crop_left_px=config.crop_left_px,
            crop_right_px=config.crop_right_px,
        )
    else:
        config.rectify_function(video_path, output_video_path=output_path)


@guf.timeit(label="rectify_pipeline.run")
def run(config: RunConfig):
    """
    Rectify config.input_path (a single video file, or every video in a
    folder) into config.output_folder, according to the rest of config.

    Returns the list of output video paths (both newly created and skipped
    ones that already existed).
    """
    _apply_global_toggles(config)

    input_path = Path(config.input_path)
    output_folder = Path(config.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        video_files = [input_path]
    elif input_path.is_dir():
        video_files = sorted(
            f for f in input_path.iterdir()
            if f.is_file() and f.suffix.lower() in config.video_extensions
        )
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    output_paths = []
    for video_file in tqdm(video_files, desc="Rectifying videos", unit="video", disable=not config.tqdm_enabled):
        output_path = output_folder / video_file.name

        if output_path.exists() and not config.overwrite_existing:
            print(f"Skipping (already exists): {output_path}")
            output_paths.append(output_path)
            continue

        dprint(f"Rectifying: {video_file} -> {output_path}")
        _rectify_one(config, video_file, output_path)
        output_paths.append(output_path)

    guf.print_timing_summary()
    return output_paths


if __name__ == "__main__":
    if len(sys.argv) > 1:
        _input_path = Path(sys.argv[1])
    else:
        _input_path = SCRIPT_DIR / "Input Videos"

    _output_folder = SCRIPT_DIR / "Rect Videos"

    run(RunConfig(
        input_path=_input_path,
        output_folder=_output_folder,
        overwrite_existing=False,
        rectify_mode="widen",
    ))

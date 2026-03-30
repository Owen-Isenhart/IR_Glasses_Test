from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import yaml

from src.camera.ir_capture import CameraConfig, IRCamera
from src.identity.baseline_manager import BaselineManager
from src.identity.scoring import (
    AutoCalConfig,
    AutoCalibrator,
    DistanceSmoother,
    Thresholds,
    classify_distance,
    cosine_distance,
)
from src.identity.state_machine import IdentityStateMachine, StateConfig
from src.io.session_logger import SessionLogger
from src.ui.hud import draw_hud
from src.vision.backends import create_backend


def load_config(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IR privacy glasses diagnostic tool")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--camera-index", type=int, default=None)
    parser.add_argument("--backend", choices=["mediapipe", "dlib"], default=None)
    parser.add_argument("--autocalibrate", action="store_true")
    parser.add_argument("--probe", action="store_true")
    return parser.parse_args()


def capture_burst(camera: IRCamera, n_frames: int) -> List[np.ndarray]:
    frames = []
    for _ in range(n_frames):
        frames.append(camera.read())
    return frames


def choose_backend(current_name: str) -> str:
    return "dlib" if current_name == "mediapipe" else "mediapipe"


def load_thresholds_for_backend(cfg: Dict[str, object], backend_name: str) -> Thresholds:
    thresholds_cfg = cfg.get("thresholds", {})
    defaults_cfg = cfg.get("threshold_defaults", {})
    backend_defaults = defaults_cfg.get(backend_name, {}) if isinstance(defaults_cfg, dict) else {}

    if backend_name == "dlib":
        fallback_match = 0.08
        fallback_blocked = 0.15
    else:
        fallback_match = 0.08
        fallback_blocked = 0.15

    match_max = float(backend_defaults.get("match_max", thresholds_cfg.get("match_max", fallback_match)))
    blocked_max = float(backend_defaults.get("blocked_max", thresholds_cfg.get("blocked_max", fallback_blocked)))
    blocked_max = max(blocked_max, match_max + 1e-4)
    return Thresholds(match_max=match_max, blocked_max=blocked_max)


def fit_frame_to_window(frame, target_w: int, target_h: int, preserve_aspect: bool):
    """Scale frame to target size with optional aspect-ratio preservation."""
    win_w = max(1, int(target_w))
    win_h = max(1, int(target_h))
    h, w = frame.shape[:2]
    if w <= 0 or h <= 0:
        return frame

    if not preserve_aspect:
        return cv2.resize(frame, (win_w, win_h), interpolation=cv2.INTER_LINEAR)

    scale = min(win_w / w, win_h / h)
    out_w = max(1, int(round(w * scale)))
    out_h = max(1, int(round(h * scale)))
    resized = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

    if out_w == win_w and out_h == win_h:
        return resized

    canvas = np.zeros((win_h, win_w, 3), dtype=frame.dtype)
    x0 = (win_w - out_w) // 2
    y0 = (win_h - out_h) // 2
    canvas[y0 : y0 + out_h, x0 : x0 + out_w] = resized
    return canvas


def main() -> int:
    args = parse_args()
    cfg = load_config(Path(args.config))

    camera_cfg = cfg.get("camera", {})
    thresholds_cfg = cfg.get("thresholds", {})
    runtime_cfg = cfg.get("runtime", {})
    ui_cfg = cfg.get("ui", {})

    camera_index = args.camera_index if args.camera_index is not None else int(camera_cfg.get("index", 2))
    backend_name = args.backend or str(runtime_cfg.get("backend", "mediapipe"))
    autocal = bool(args.autocalibrate or runtime_cfg.get("autocalibrate", False))

    camera = IRCamera(
        CameraConfig(
            index=camera_index,
            width=int(camera_cfg.get("width", 640)),
            height=int(camera_cfg.get("height", 480)),
            fps=int(camera_cfg.get("fps", 30)),
            backend_prefer_v4l2=bool(camera_cfg.get("prefer_v4l2", True)),
        )
    )

    camera.open()
    preflight = camera.preflight(sample_frames=int(runtime_cfg.get("preflight_frames", 15)))
    if args.probe:
        print(json.dumps({"camera": camera.metadata(), "preflight": preflight}, indent=2))
        camera.close()
        return 0

    window_name = "IR Privacy Diagnostic"
    window_w = int(ui_cfg.get("window_width", 1280))
    window_h = int(ui_cfg.get("window_height", 900))
    render_w = int(ui_cfg.get("render_width", 1920))
    render_h = int(ui_cfg.get("render_height", 1080))
    preserve_aspect = bool(ui_cfg.get("preserve_aspect", True))
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(window_name, window_w, window_h)

    backend = create_backend(backend_name)
    ok, message = backend.is_available()
    if not ok:
        camera.close()
        raise RuntimeError(f"Backend '{backend_name}' unavailable: {message}")

    baseline_mgr = BaselineManager(Path(cfg.get("paths", {}).get("baseline_dir", "data/baseline")))
    baseline = baseline_mgr.load_embedding(backend.name)

    thresholds = load_thresholds_for_backend(cfg, backend.name)
    autocalibrator = AutoCalibrator(
        AutoCalConfig(
            min_samples=int(thresholds_cfg.get("autocal_min_samples", 25)),
            window=int(thresholds_cfg.get("autocal_window", 120)),
            update_every=int(thresholds_cfg.get("autocal_update_every", 10)),
            match_quantile=float(thresholds_cfg.get("autocal_match_quantile", 0.95)),
            blocked_quantile=float(thresholds_cfg.get("autocal_block_quantile", 0.995)),
            match_margin=float(thresholds_cfg.get("autocal_match_margin", 0.04)),
            blocked_margin=float(thresholds_cfg.get("autocal_block_margin", 0.14)),
            min_gap=float(thresholds_cfg.get("autocal_min_gap", 0.08)),
            max_blocked=float(thresholds_cfg.get("autocal_max_blocked", 0.85)),
        )
    )
    smoother = DistanceSmoother(
        alpha=float(thresholds_cfg.get("ema_alpha", 0.30)),
        rolling=int(thresholds_cfg.get("rolling_window", 5)),
    )
    state_machine = IdentityStateMachine(
        StateConfig(evasion_frames=int(thresholds_cfg.get("evasion_frames", 12)))
    )

    session_name = datetime.now().strftime("session_%Y%m%d_%H%M%S")
    logger = SessionLogger(Path(cfg.get("paths", {}).get("log_dir", "data/logs")), session_name)
    logger.event({
        "event": "session_start",
        "camera": camera.metadata(),
        "preflight": preflight,
        "backend": backend.name,
    })

    if baseline is None:
        print("No baseline found. Press 'c' to capture baseline without glasses.")

    prev_time = time.perf_counter()
    last_distance = None
    last_smooth = None
    banner_message = ""
    banner_until = 0.0

    try:
        while True:
            frame = camera.read()
            obs = backend.observe(frame)

            fps_now = 1.0 / max(time.perf_counter() - prev_time, 1e-6)
            prev_time = time.perf_counter()

            state = "scanning"
            distance_raw = None
            distance_smooth = None

            if baseline is not None and obs.found and obs.embedding is not None:
                if obs.embedding.shape != baseline.shape:
                    logger.event(
                        {
                            "event": "baseline_dim_mismatch",
                            "backend": backend.name,
                            "obs_dim": int(obs.embedding.shape[0]),
                            "baseline_dim": int(baseline.shape[0]),
                        }
                    )
                    baseline = None
                    banner_message = (
                        f"Baseline dimension mismatch for {backend.name}; press 'c' to calibrate"
                    )
                    banner_until = time.time() + 3.0
                    state = state_machine.update(face_found=True, distance_label="blocked")
                else:
                    distance_raw = cosine_distance(obs.embedding, baseline)
                    distance_smooth = smoother.update(distance_raw)
                    label = classify_distance(distance_smooth, thresholds)
                    state = state_machine.update(face_found=True, distance_label=label)
                    last_distance = distance_raw
                    last_smooth = distance_smooth
            elif obs.found:
                state = state_machine.update(face_found=True, distance_label="blocked")
            else:
                state = state_machine.update(face_found=False, distance_label="blocked")

            if autocal and baseline is not None and obs.found and distance_raw is not None:
                updated = autocalibrator.update(distance_raw, thresholds)
                if updated:
                    logger.event(
                        {
                            "event": "autocal_update",
                            "match_max": thresholds.match_max,
                            "blocked_max": thresholds.blocked_max,
                            "samples": autocalibrator.sample_count,
                        }
                    )

            logger.frame(
                state=state,
                distance_raw=distance_raw,
                distance_smooth=distance_smooth,
                fps=fps_now,
                backend=backend.name,
                face_found=obs.found,
            )

            rendered = draw_hud(
                frame=frame,
                state=state,
                distance=last_smooth if last_smooth is not None else last_distance,
                bbox=obs.bbox,
                backend_name=backend.name,
                fps=fps_now,
                autocalibrate=autocal,
                threshold_match=thresholds.match_max,
                threshold_blocked=thresholds.blocked_max,
                autocal_samples=autocalibrator.sample_count,
                banner_message=banner_message if time.time() < banner_until else "",
            )

            display_frame = fit_frame_to_window(
                rendered,
                render_w,
                render_h,
                preserve_aspect=preserve_aspect,
            )
            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("a"):
                autocal = not autocal
                if autocal:
                    autocalibrator.reset()
                logger.event({"event": "toggle_autocalibrate", "enabled": autocal})
                banner_message = f"Auto-calibration {'enabled' if autocal else 'disabled'}"
                banner_until = time.time() + 1.8
            if key == ord("b"):
                candidate = choose_backend(backend.name)
                next_backend = create_backend(candidate)
                ok2, message2 = next_backend.is_available()
                if ok2:
                    backend = next_backend
                    baseline = baseline_mgr.load_embedding(backend.name)
                    thresholds = load_thresholds_for_backend(cfg, backend.name)
                    smoother.reset()
                    autocalibrator.reset()
                    state_machine = IdentityStateMachine(state_machine.cfg)
                    logger.event({"event": "switch_backend", "backend": backend.name})
                    if baseline is None:
                        banner_message = (
                            f"Switched backend to {backend.name}; no baseline for this backend (press 'c')"
                        )
                    else:
                        banner_message = f"Switched backend to {backend.name}"
                    banner_until = time.time() + 1.8
                else:
                    logger.event({"event": "switch_backend_failed", "backend": candidate, "reason": message2})
                    banner_message = f"Backend switch failed: {candidate} unavailable"
                    banner_until = time.time() + 2.5
            if key == ord("c"):
                n = int(runtime_cfg.get("baseline_frames", 25))
                banner_message = "Capturing baseline... hold still without glasses"
                banner_until = time.time() + 2.5
                frames = capture_burst(camera, n)
                new_baseline, snapshot, meta = baseline_mgr.capture_baseline(frames, backend)
                meta.update(
                    {
                        "backend": backend.name,
                        "camera_index": camera_index,
                        "thresholds": {
                            "match_max": thresholds.match_max,
                            "blocked_max": thresholds.blocked_max,
                        },
                    }
                )
                baseline_mgr.save(new_baseline, snapshot, meta, backend_name=backend.name)
                baseline = new_baseline
                smoother.reset()
                autocalibrator.reset()
                state_machine = IdentityStateMachine(state_machine.cfg)
                logger.event({"event": "baseline_captured", **meta})
                banner_message = "Baseline captured"
                banner_until = time.time() + 1.8

    finally:
        logger.event({"event": "session_end"})
        logger.close()
        camera.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

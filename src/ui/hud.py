from __future__ import annotations

from typing import Optional, Tuple

import cv2


STATE_COLOR = {
    "scanning": (0, 215, 255),
    "matched": (0, 180, 0),
    "blocked": (255, 220, 0),
    "evasion": (0, 0, 255),
}


def draw_hud(
    frame,
    state: str,
    distance: Optional[float],
    bbox: Optional[Tuple[int, int, int, int]],
    backend_name: str,
    camera_index: int,
    fps: float,
    autocalibrate: bool,
    threshold_match: float,
    threshold_blocked: float,
    autocal_samples: int,
    banner_message: str = "",
):
    canvas = frame.copy()

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (180, 180, 180), 2)

    color = STATE_COLOR.get(state, (220, 220, 220))
    distance_text = "n/a" if distance is None else f"{distance:.3f}"
    status = state.upper()

    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 78), (20, 20, 20), -1)
    cv2.putText(canvas, f"State: {status}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.putText(
        canvas,
        f"Distance: {distance_text}   Backend: {backend_name}   Camera: {camera_index}",
        (10, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (230, 230, 230),
        1,
    )
    cv2.putText(
        canvas,
        f"FPS: {fps:.1f}   AutoCal: {'ON' if autocalibrate else 'OFF'}   Keys: v=cam 0/2, b=backend, c=baseline, a=autocal, q=quit",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (230, 230, 230),
        1,
    )
    cv2.putText(
        canvas,
        f"Thr(match={threshold_match:.3f}, block={threshold_blocked:.3f})  Samples: {autocal_samples}",
        (10, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (210, 210, 210),
        1,
    )

    if banner_message:
        cv2.rectangle(canvas, (0, canvas.shape[0] - 24), (canvas.shape[1], canvas.shape[0]), (15, 15, 15), -1)
        cv2.putText(
            canvas,
            banner_message,
            (10, canvas.shape[0] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (230, 230, 230),
            1,
        )

    return canvas

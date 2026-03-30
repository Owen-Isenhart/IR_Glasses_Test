from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict


@dataclass
class SessionStats:
    frames: int = 0
    matched: int = 0
    blocked: int = 0
    evasion: int = 0
    scanning: int = 0

    def bump(self, state: str) -> None:
        self.frames += 1
        if state == "matched":
            self.matched += 1
        elif state == "blocked":
            self.blocked += 1
        elif state == "evasion":
            self.evasion += 1
        else:
            self.scanning += 1


@dataclass
class SessionLogger:
    output_dir: Path
    session_name: str
    _fp: object = field(init=False)
    stats: SessionStats = field(default_factory=SessionStats)

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / f"{self.session_name}.jsonl"
        self.summary_path = self.output_dir / f"{self.session_name}_summary.json"
        self._fp = self.log_path.open("w", encoding="utf-8")

    def event(self, payload: Dict[str, object]) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **payload,
        }
        self._fp.write(json.dumps(record) + "\n")
        self._fp.flush()

    def frame(self, state: str, distance_raw, distance_smooth, fps: float, backend: str, face_found: bool) -> None:
        self.stats.bump(state)
        self.event(
            {
                "event": "frame",
                "state": state,
                "distance_raw": distance_raw,
                "distance_smooth": distance_smooth,
                "fps": fps,
                "backend": backend,
                "face_found": face_found,
            }
        )

    def close(self) -> None:
        if self.stats.frames > 0:
            summary = {
                "frames": self.stats.frames,
                "matched_ratio": self.stats.matched / self.stats.frames,
                "blocked_ratio": self.stats.blocked / self.stats.frames,
                "evasion_ratio": self.stats.evasion / self.stats.frames,
                "scanning_ratio": self.stats.scanning / self.stats.frames,
            }
        else:
            summary = {
                "frames": 0,
                "matched_ratio": 0.0,
                "blocked_ratio": 0.0,
                "evasion_ratio": 0.0,
                "scanning_ratio": 0.0,
            }

        with self.summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        self._fp.close()

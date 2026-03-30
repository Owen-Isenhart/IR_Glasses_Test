from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StateConfig:
    evasion_frames: int = 12


class IdentityStateMachine:
    """State machine where sustained no-face becomes successful total evasion."""

    def __init__(self, cfg: StateConfig) -> None:
        self.cfg = cfg
        self._no_face_count = 0
        self._last_state = "scanning"

    def update(self, face_found: bool, distance_label: str) -> str:
        if not face_found:
            self._no_face_count += 1
            if self._no_face_count >= self.cfg.evasion_frames:
                self._last_state = "evasion"
                return "evasion"
            # Keep state stable during brief detector dropouts to avoid HUD flicker.
            if self._last_state in {"matched", "blocked"}:
                return self._last_state
            self._last_state = "blocked"
            return "blocked"

        self._no_face_count = 0
        if distance_label == "matched":
            self._last_state = "matched"
            return "matched"

        self._last_state = "blocked"
        return "blocked"

    @property
    def last_state(self) -> str:
        return self._last_state

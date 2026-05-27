from __future__ import annotations

import time
import winsound


class FallAlarm:
    def __init__(self, cooldown_seconds: float = 2.0) -> None:
        self.cooldown_seconds = cooldown_seconds
        self._last_alert = 0.0

    def reset(self) -> None:
        self._last_alert = 0.0

    def notify_if_needed(self, has_fall: bool) -> None:
        if not has_fall:
            return
        now = time.monotonic()
        if now - self._last_alert < self.cooldown_seconds:
            return
        self._last_alert = now
        winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)


from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from ..video_processor import Frame


class EventAnalysisSession:
    """Stable frame-by-frame session API for event analysis adapters.

    This compatibility session intentionally buffers frames and delegates the heavy
    batch implementation to ``finish``. That preserves current event-window ordering,
    stage timings, and payload shape while providing a future-proof incremental API.
    """

    def __init__(
        self,
        *,
        video_path: str | Path,
        out_dir: str | Path,
        session_id: str,
        finish_fn: Callable[[list[Frame]], dict],
    ) -> None:
        self.video_path = Path(video_path)
        self.out_dir = Path(out_dir)
        self.session_id = session_id
        self._finish_fn = finish_fn
        self._frames: list[Frame] = []

    def process_frame(self, frame: Frame) -> None:
        self._frames.append(frame)

    def finish(self) -> dict:
        return self._finish_fn(list(self._frames))

    @property
    def frames(self) -> list[Frame]:
        return list(self._frames)


__all__ = ["EventAnalysisSession"]

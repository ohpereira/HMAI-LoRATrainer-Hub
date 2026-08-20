"""Best-effort progress reporting for the trusted Hervora orchestrator."""

from __future__ import annotations

import json
import logging
from urllib.request import Request, urlopen

from log_parser import TrainingProgress

logger = logging.getLogger(__name__)


class HervoraProgressReporter:
    def __init__(self, callback_url: str | None) -> None:
        self.callback_url = callback_url

    def send(self, progress: TrainingProgress, stage: str = "training") -> None:
        if not self.callback_url:
            return
        payload = json.dumps({
            "stage": stage,
            "step": max(0, int(progress.step)),
            "total_steps": max(0, int(progress.total_steps)),
            "epoch": max(0, int(progress.epoch)),
            "percent": round(max(0.0, min(100.0, float(progress.percent))), 2),
        }).encode("utf-8")
        try:
            request = Request(
                self.callback_url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=3):
                pass
        except Exception as error:  # Reporting must never interrupt GPU work.
            logger.warning("Hervora progress report failed: %s", error)

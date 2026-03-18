"""
voice_budget/core.py

The core measurement engine. Framework-agnostic.
Measures actual TTFT per turn, maintains rolling statistics,
detects when context growth is causing latency to climb,
and triggers the compression feedback loop.
"""

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional

import numpy as np
import tiktoken


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class TTFTSample:
    turn: int
    ttft_ms: float
    token_count: int
    compressed: bool = False
    compression_strategy: Optional[str] = None


@dataclass
class CompressionEvent:
    turn: int
    strategy: str
    tokens_before: int
    tokens_after: int
    ttft_before_ms: float
    ttft_after_ms: Optional[float] = None   # filled after next turn
    delta_ms: Optional[float] = None
    rolled_back: bool = False


@dataclass
class BudgetStats:
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    jitter_ms: float              # std deviation
    token_count: int
    turn: int
    budget_violated: bool
    compression_events: int
    total_tokens_saved: int


# ── Main engine ───────────────────────────────────────────────────────────────

class TTFTMeasurer:
    """
    Wraps any async LLM call. Measures actual wall-clock TTFT
    (time from call start to first token received).
    Framework-agnostic — works with Pipecat, LiveKit, raw asyncio.
    """

    def __init__(
        self,
        target_ms: float = 800.0,
        window_size: int = 20,
        model: str = "gpt-4o",
    ):
        self.target_ms = target_ms
        self.window_size = window_size
        self.model = model

        self._samples: Deque[TTFTSample] = deque(maxlen=window_size)
        self._compression_events: List[CompressionEvent] = []
        self._turn = 0
        self._total_tokens_saved = 0

        # tiktoken for token counting
        try:
            self._enc = tiktoken.encoding_for_model(model)
        except KeyError:
            self._enc = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, messages: List[Dict]) -> int:
        """Count tokens in a messages list (OpenAI format)."""
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += len(self._enc.encode(content))
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        total += len(self._enc.encode(part.get("text", "")))
            # 4 tokens overhead per message (OpenAI spec)
            total += 4
        total += 2  # reply priming
        return total

    def stats(self) -> Optional[BudgetStats]:
        """Return current rolling statistics. None if fewer than 3 samples."""
        if len(self._samples) < 3:
            return None
        times = [s.ttft_ms for s in self._samples]
        token_count = self._samples[-1].token_count if self._samples else 0
        return BudgetStats(
            p50_ms=float(np.percentile(times, 50)),
            p95_ms=float(np.percentile(times, 95)),
            p99_ms=float(np.percentile(times, 99)),
            mean_ms=float(np.mean(times)),
            jitter_ms=float(np.std(times)),
            token_count=token_count,
            turn=self._turn,
            budget_violated=float(np.percentile(times, 95)) > self.target_ms,
            compression_events=len(self._compression_events),
            total_tokens_saved=self._total_tokens_saved,
        )

    def should_compress(self) -> bool:
        """True when P95 TTFT has exceeded target."""
        s = self.stats()
        if s is None:
            return False
        return s.p95_ms > self.target_ms

    def record_sample(
        self,
        ttft_ms: float,
        token_count: int,
        compressed: bool = False,
        strategy: Optional[str] = None,
    ) -> None:
        self._turn += 1
        sample = TTFTSample(
            turn=self._turn,
            ttft_ms=ttft_ms,
            token_count=token_count,
            compressed=compressed,
            compression_strategy=strategy,
        )
        self._samples.append(sample)

        # Fill in ttft_after for the last compression event
        if self._compression_events and self._compression_events[-1].ttft_after_ms is None:
            ev = self._compression_events[-1]
            ev.ttft_after_ms = ttft_ms
            ev.delta_ms = ev.ttft_before_ms - ttft_ms  # positive = improvement

            # Rollback check: if compression made things worse by >10%
            if ev.delta_ms is not None and ev.delta_ms < -(self.target_ms * 0.10):
                ev.rolled_back = True

    def record_compression(
        self,
        strategy: str,
        tokens_before: int,
        tokens_after: int,
        ttft_before_ms: float,
    ) -> CompressionEvent:
        ev = CompressionEvent(
            turn=self._turn,
            strategy=strategy,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            ttft_before_ms=ttft_before_ms,
        )
        self._compression_events.append(ev)
        self._total_tokens_saved += (tokens_before - tokens_after)
        return ev

    def compression_history(self) -> List[CompressionEvent]:
        return list(self._compression_events)

    def snapshot_report(self) -> Dict[str, Any]:
        """Generate a structured report of all compression decisions."""
        events = self._compression_events
        helpful = [e for e in events if e.delta_ms and e.delta_ms > 0]
        harmful = [e for e in events if e.rolled_back]
        s = self.stats()
        return {
            "total_turns": self._turn,
            "current_p50_ms": s.p50_ms if s else None,
            "current_p95_ms": s.p95_ms if s else None,
            "target_ms": self.target_ms,
            "budget_met": (s.p95_ms <= self.target_ms) if s else None,
            "compression_events": len(events),
            "helpful_compressions": len(helpful),
            "harmful_compressions": len(harmful),
            "avg_ttft_improvement_ms": (
                float(np.mean([e.delta_ms for e in helpful])) if helpful else 0.0
            ),
            "total_tokens_saved": self._total_tokens_saved,
            "strategies_used": list({e.strategy for e in events}),
        }
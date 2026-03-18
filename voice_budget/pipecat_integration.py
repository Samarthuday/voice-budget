"""
voice_budget/pipecat_integration.py

Native Pipecat integration.
Inserts voice-budget as a FrameProcessor in your Pipecat pipeline.

Usage:
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.services.openai import OpenAILLMService
    from voice_budget.pipecat_integration import VoiceBudgetProcessor

    llm = OpenAILLMService(api_key=..., model="gpt-4o")
    budget = VoiceBudgetProcessor(target_ms=800, verbose=True)

    pipeline = Pipeline([
        transport.input(),
        stt,
        context_aggregator.user(),
        budget,          # <-- insert here, before LLM
        llm,
        tts,
        transport.output(),
        context_aggregator.assistant(),
    ])

VoiceBudgetProcessor sits between the context aggregator and the LLM.
It intercepts LLMMessagesAppendFrame, checks if compression is needed,
modifies the context in-place, then passes the frame downstream.
"""

import time
from typing import Optional

from .compressors import BudgetCompressor
from .core import TTFTMeasurer
from .wrapper import compute_effective_target


class VoiceBudgetProcessor:
    """
    Pipecat FrameProcessor that wraps context management.
    Compatible with Pipecat's LLMContext and frame-based pipeline.

    Implements pipecat.processors.frame_processor.FrameProcessor interface.
    Import pipecat types dynamically to keep voice-budget dependency-free
    from pipecat (pipecat is optional).
    """

    def __init__(
        self,
        target_ms: float = 800.0,
        token_budget: int = 2000,
        model: str = "gpt-4o",
        window_size: int = 20,
        use_semantic: bool = True,
        verbose: bool = False,
        on_compression=None,
        on_budget_violation=None,
    ):
        self._target_ms = target_ms
        self._token_budget = token_budget
        self._verbose = verbose
        self._on_compression = on_compression
        self._on_budget_violation = on_budget_violation

        self._measurer = TTFTMeasurer(
            target_ms=target_ms,
            window_size=window_size,
            model=model,
        )
        self._compressor = BudgetCompressor(
            target_tokens=token_budget,
            use_semantic=use_semantic,
            use_summarise=False,  # LLM fn not available at init time for pipecat
        )
        self._turn_start: Optional[float] = None

    # ── Pipecat FrameProcessor interface ──────────────────────────────────────

    async def process_frame(self, frame, direction):
        """
        Intercept context frames. Called by Pipecat pipeline.
        """
        try:
            from pipecat.frames.frames import (
                LLMMessagesAppendFrame,
                LLMMessagesUpdateFrame,
                LLMTextFrame,
            )
        except ImportError:
            # Pipecat not installed — pass through
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, (LLMMessagesAppendFrame, LLMMessagesUpdateFrame)):
            # This is where context is about to be sent to the LLM
            messages = getattr(frame, "messages", None)
            if messages is not None:
                messages = await self._maybe_compress(messages)
                # Mutate in-place — Pipecat reads from the frame object
                if hasattr(frame, "messages"):
                    frame.messages = messages

            self._turn_start = time.perf_counter()

        elif isinstance(frame, LLMTextFrame):
            # First token received — this is TTFT
            if self._turn_start is not None:
                ttft_ms = (time.perf_counter() - self._turn_start) * 1000
                # Get token count from last messages (approximate)
                self._measurer.record_sample(
                    ttft_ms=ttft_ms,
                    token_count=getattr(self, "_last_token_count", 0),
                )
                self._turn_start = None

                if self._verbose:
                    s = self._measurer.stats()
                    if s and self._measurer._turn % 5 == 0:
                        history = self._measurer.compression_history()
                        last_ev = history[-1] if history else None
                        action = ""
                        if last_ev and last_ev.turn == self._measurer._turn:
                            action = f" [{last_ev.strategy}]"
                        print(
                            f"[voice-budget/pipecat] "
                            f"Turn {s.turn} P50={s.p50_ms:.0f}ms "
                            f"P95={s.p95_ms:.0f}ms tokens={s.token_count}"
                            f" compressions={len(history)}{action}"
                        )

        await self.push_frame(frame, direction)

    async def _maybe_compress(self, messages):
        """Check if compression is needed and apply it."""
        current_tokens = self._measurer.count_tokens(messages)
        self._last_token_count = current_tokens

        if not self._measurer.should_compress():
            return messages

        s = self._measurer.stats()
        if self._verbose:
            print(
                f"[voice-budget/pipecat] P95={s.p95_ms:.0f}ms > "
                f"{self._target_ms}ms. Compressing {current_tokens} tokens..."
            )

        if self._on_budget_violation and s:
            try:
                self._on_budget_violation(s)
            except Exception:
                pass

        ttft_before = s.p95_ms if s else 0.0

        effective_target = compute_effective_target(
            current_tokens=current_tokens,
            stats=s,
            target_tokens=self._compressor.target_tokens,
            target_ms=self._target_ms,
        )

        compressed, strategy, removed = await self._compressor.compress(
            messages,
            current_tokens,
            self._measurer.count_tokens,
            effective_target=effective_target,
        )
        tokens_after = self._measurer.count_tokens(compressed)

        ev = self._measurer.record_compression(
            strategy=strategy,
            tokens_before=current_tokens,
            tokens_after=tokens_after,
            ttft_before_ms=ttft_before,
        )

        if self._on_compression:
            try:
                self._on_compression(ev)
            except Exception:
                pass

        if self._verbose:
            print(
                f"[voice-budget/pipecat] {strategy}: "
                f"{current_tokens}→{tokens_after} tokens (saved {removed})"
            )

        self._last_token_count = tokens_after
        return compressed

    def stats(self):
        return self._measurer.stats()

    def compression_history(self):
        return self._measurer.compression_history()

    def report(self):
        return self._measurer.snapshot_report()

    # Pipecat requires push_frame — subclasses provide real implementation
    async def push_frame(self, frame, direction):
        pass  # overridden by Pipecat's FrameProcessor base
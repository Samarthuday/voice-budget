import logging
import time
from collections.abc import AsyncIterator
from typing import Any, Dict, List, Optional

from .compressors import BudgetCompressor
from .core import TTFTMeasurer
from .wrapper import compute_effective_target


logger = logging.getLogger(__name__)


class VoiceBudgetAgent:
    """LiveKit agent wrapper for voice conversation context management.
    
    Integrates with livekit.agents to measure TTFT and compress context.
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
        semantic_threshold: int = 1500,
        summarise_threshold: int = 4000,
    ):
        """Initialize LiveKit voice budget agent.
        
        Args:
            target_ms: TTFT budget in milliseconds (P95 target).
            token_budget: Target token count after compression.
            model: Model name for tiktoken token counting (e.g., "gpt-4o").
            window_size: Rolling window size for P50/P95 stats.
            use_semantic: Enable semantic trimming (requires sentence-transformers).
            verbose: Log compression events.
            on_compression: Callback(CompressionEvent) after compression.
            on_budget_violation: Callback(BudgetStats) when TTFT exceeds budget.
            semantic_threshold: Use SemanticTrim when current_tokens > this value (default 1500).
            summarise_threshold: Use SummariseTail when current_tokens > this value (default 4000).
        """
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
            use_summarise=False,
            semantic_threshold=semantic_threshold,
            summarise_threshold=summarise_threshold,
        )
        self._turn_start: Optional[float] = None
        self._last_token_count: int = 0

    async def process_messages(
        self,
        messages: List[Dict],
        llm_fn,
        **kwargs,
    ) -> Any:
        """Process conversation messages, measure TTFT, and compress if needed.
        
        This is the main entry point for LiveKit agent integration.
        Call before passing messages to the LLM.
        
        Args:
            messages: Conversation history (list of dicts with role/content).
            llm_fn: Async LLM callable that takes messages and returns a response.
            **kwargs: Additional keyword args forwarded to llm_fn.
            
        Returns:
            The LLM response, or a wrapped async stream that records TTFT on
            its first yielded chunk.
        """
        # Check if compression is needed and apply
        messages = await self._maybe_compress(messages)

        # Record turn start time after compression so TTFT reflects LLM latency.
        self._turn_start = time.perf_counter()

        # Call LLM and measure TTFT
        try:
            response = await llm_fn(messages, **kwargs)
        except Exception:
            self._turn_start = None
            logger.exception("[voice-budget/livekit] LLM call failed")
            raise

        if isinstance(response, AsyncIterator):
            return self._wrap_streaming_response(response)

        self._record_ttft_sample()
        return response

    def _record_ttft_sample(self) -> None:
        """Record a single TTFT sample for the active turn."""
        if self._turn_start is not None:
            ttft_ms = (time.perf_counter() - self._turn_start) * 1000
            self._measurer.record_sample(
                ttft_ms=ttft_ms,
                token_count=self._last_token_count,
            )
            self._turn_start = None

            if self._verbose:
                s = self._measurer.stats()
                if s and self._measurer._turn % 5 == 0:
                    history = self._measurer.compression_history()
                    last_ev = history[-1] if history else None
                    action = ""
                    if last_ev and last_ev.turn == self._measurer._turn - 1:
                        action = f" [{last_ev.strategy}]"
                    logger.info(
                        f"[voice-budget/livekit] "
                        f"Turn {s.turn} P50={s.p50_ms:.0f}ms "
                        f"P95={s.p95_ms:.0f}ms tokens={s.token_count} "
                        f"compressions={len(history)}{action}"
                    )

    async def _wrap_streaming_response(self, stream: AsyncIterator[Any]) -> AsyncIterator[Any]:
        """Wrap a streaming response so TTFT is recorded on first chunk."""
        first = True
        try:
            async for chunk in stream:
                if first:
                    first = False
                    self._record_ttft_sample()
                yield chunk
        except Exception:
            if first:
                self._turn_start = None
            raise

        if first:
            self._record_ttft_sample()

    async def _maybe_compress(self, messages: List[Dict]) -> List[Dict]:
        """Check if compression is needed and apply it."""
        current_tokens = self._measurer.count_tokens(messages)
        self._last_token_count = current_tokens

        if not self._measurer.should_compress():
            return messages

        s = self._measurer.stats()
        if self._verbose:
            logger.info(
                f"[voice-budget/livekit] P95={s.p95_ms:.0f}ms > "
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
            logger.info(
                f"[voice-budget/livekit] {strategy}: "
                f"{current_tokens}→{tokens_after} tokens (saved {removed})"
            )

        self._last_token_count = tokens_after
        return compressed

    def stats(self):
        """Return current TTFT stats (BudgetStats)."""
        return self._measurer.stats()

    def compression_history(self):
        """Return list of all compression events."""
        return self._measurer.compression_history()

    def report(self):
        """Return a snapshot report of session stats."""
        return self._measurer.snapshot_report()

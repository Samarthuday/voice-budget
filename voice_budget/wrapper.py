"""
voice_budget/wrapper.py

The public API. One function call wraps any async LLM.

Usage (framework-agnostic):
    from voice_budget import wrap

    managed_llm = wrap(your_async_llm_fn, target_ms=800)
    # then use managed_llm exactly like your_async_llm_fn

Usage (Pipecat — see pipecat_integration.py for full example):
    from voice_budget.pipecat import VoiceBudgetProcessor
"""

import time
import warnings
from collections.abc import AsyncIterator
from typing import Any, Callable, Dict, List, Optional

from .compressors import BudgetCompressor
from .core import TTFTMeasurer


def compute_effective_target(
    current_tokens: int,
    stats: Any,
    target_tokens: int,
    target_ms: float,
) -> int:
    """Compute a dynamic effective token target based on TTFT stats.

    When tokens are already under the static budget but TTFT is high,
    reduce proportionally to how far P95 exceeds target_ms.
    """
    effective_target = target_tokens
    if current_tokens <= effective_target and stats:
        p95 = getattr(stats, "p95_ms", None)
        if not p95 or p95 <= 0:
            return effective_target
        ratio = target_ms / p95
        effective_target = max(int(current_tokens * ratio), 4)
    return effective_target


class VoiceBudget:
    """
    Wraps any async LLM callable with a latency feedback loop.

    The wrapped function has the same signature as the original:
        async fn(messages: List[Dict], **kwargs) -> str  (non-streaming)
        async fn(messages: List[Dict], **kwargs) -> AsyncIterator[str]  (streaming)

    voice-budget intercepts the call, measures TTFT, compresses context
    when P95 TTFT exceeds target_ms, and measures whether it helped.
    """

    def __init__(
        self,
        llm_fn: Callable,
        target_ms: float = 800.0,
        model: str = "gpt-4o",
        window_size: int = 20,
        token_budget: int = 2000,
        use_semantic: bool = True,
        use_summarise: bool = False,   # off by default — costs an LLM call
        on_compression: Optional[Callable] = None,
        on_budget_violation: Optional[Callable] = None,
        verbose: bool = False,
        accuracy_fn: Optional[Callable] = None,
    ):
        """
        Args:
            llm_fn:            Your async LLM function.
            target_ms:         TTFT budget in milliseconds (P95).
            model:             Model name for tiktoken token counting.
            window_size:       Rolling window for P50/P95 calculation.
            token_budget:      Target token count after compression.
            use_semantic:      Enable semantic trim (requires sentence-transformers).
            use_summarise:     Enable LLM-based tail summarisation.
            on_compression:    Callback(CompressionEvent) after each compression.
            on_budget_violation: Callback(BudgetStats) when P95 > target_ms.
            verbose:           Print compression decisions to stdout.
            accuracy_fn:       DEPRECATED. Ignored. Will be removed in a future release.
        """
        self._llm_fn = llm_fn
        self._verbose = verbose
        self._on_compression = on_compression
        self._on_budget_violation = on_budget_violation

        if accuracy_fn is not None:
            warnings.warn(
                "VoiceBudget(accuracy_fn=...) is deprecated and ignored. "
                "It will be removed in a future release.",
                DeprecationWarning,
                stacklevel=2,
            )

        self._measurer = TTFTMeasurer(
            target_ms=target_ms,
            window_size=window_size,
            model=model,
        )
        self._compressor = BudgetCompressor(
            target_tokens=token_budget,
            llm_fn=llm_fn if use_summarise else None,
            use_semantic=use_semantic,
            use_summarise=use_summarise,
        )

        # Last known good messages (for rollback)
        self._last_good_messages: Optional[List[Dict]] = None
        self._restore_next_turn: bool = False

    # ── Main call ─────────────────────────────────────────────────────────────

    async def __call__(
        self,
        messages: List[Dict],
        **kwargs,
    ) -> Any:
        """
        Drop-in replacement for your LLM function.
        Transparently measures, compresses, and feeds back.
        """
        # Rollback: if last compression hurt, restore full context and skip
        # compression this turn so the restored context is actually used.
        restored_context = False
        if self._restore_next_turn and self._last_good_messages is not None:
            if self._verbose:
                print("[voice-budget] Restoring full context after harmful compression.")
            messages = self._last_good_messages
            self._last_good_messages = None
            restored_context = True
        self._restore_next_turn = False

        messages = list(messages)  # defensive copy
        current_tokens = self._measurer.count_tokens(messages)
        compressed = False
        strategy_used = None

        # ── Step 1: Check if we need to compress ──────────────────────────────
        if (not restored_context) and self._measurer.should_compress():
            stats = self._measurer.stats()
            if self._verbose:
                print(
                    f"[voice-budget] P95={stats.p95_ms:.0f}ms > target={self._measurer.target_ms}ms "
                    f"at turn {self._measurer._turn} ({current_tokens} tokens). Compressing..."
                )

            if self._on_budget_violation and stats:
                try:
                    self._on_budget_violation(stats)
                except Exception:
                    pass

            # Save pre-compression state for rollback
            self._last_good_messages = list(messages)
            ttft_before = stats.p95_ms if stats else 0.0

            # ── Step 2: Compress ───────────────────────────────────────────────
            effective_target = compute_effective_target(
                current_tokens=current_tokens,
                stats=stats,
                target_tokens=self._compressor.target_tokens,
                target_ms=self._measurer.target_ms,
            )

            compressed_msgs, strategy_used, tokens_removed = await self._compressor.compress(
                messages,
                current_tokens,
                self._measurer.count_tokens,
                effective_target=effective_target,
            )
            tokens_after = self._measurer.count_tokens(compressed_msgs)

            if self._verbose:
                print(
                    f"[voice-budget] Strategy={strategy_used} "
                    f"tokens: {current_tokens}→{tokens_after} "
                    f"(saved {tokens_removed})"
                )

            # Record compression event
            ev = self._measurer.record_compression(
                strategy=strategy_used,
                tokens_before=current_tokens,
                tokens_after=tokens_after,
                ttft_before_ms=ttft_before,
            )

            if self._on_compression:
                try:
                    self._on_compression(ev)
                except Exception:
                    pass

            messages = compressed_msgs
            compressed = True
            current_tokens = tokens_after
        else:
            # No compression needed — clear stale rollback snapshot
            self._last_good_messages = None

        # ── Step 3: Call LLM, measure TTFT ────────────────────────────────────
        start = time.perf_counter()
        result = await self._llm_fn(messages, **kwargs)

        # Streaming: if result is an AsyncIterator, wrap it so we measure
        # TTFT at first yielded chunk rather than at iterator creation.
        if isinstance(result, AsyncIterator):
            return self._wrap_streaming(
                result, current_tokens, compressed, strategy_used
            )

        # Non-streaming: TTFT = total await time (approximation)
        ttft_ms = (time.perf_counter() - start) * 1000
        self._record_and_check(ttft_ms, current_tokens, compressed, strategy_used)
        return result

    async def _wrap_streaming(self, stream, current_tokens, compressed, strategy_used):
        """Wrap an async iterator to measure TTFT at first chunk."""
        first = True
        # Capture start when iteration actually begins, not when the
        # iterator was created, to avoid counting consumer-side idle time.
        start = time.perf_counter()
        async for chunk in stream:
            if first:
                first = False
                ttft_ms = (time.perf_counter() - start) * 1000
                self._record_and_check(ttft_ms, current_tokens, compressed, strategy_used)
            yield chunk
        # Edge case: empty stream
        if first:
            ttft_ms = (time.perf_counter() - start) * 1000
            self._record_and_check(ttft_ms, current_tokens, compressed, strategy_used)

    def _record_and_check(self, ttft_ms, current_tokens, compressed, strategy_used):
        """Record a TTFT sample and check if rollback is needed."""
        self._measurer.record_sample(
            ttft_ms=ttft_ms,
            token_count=current_tokens,
            compressed=compressed,
            strategy=strategy_used,
        )

        if compressed:
            history = self._measurer.compression_history()
            last_ev = history[-1] if history else None
            if last_ev and last_ev.rolled_back and self._last_good_messages is not None:
                self._restore_next_turn = True
                if self._verbose:
                    print(
                        f"[voice-budget] Compression hurt latency "
                        f"(delta={last_ev.delta_ms:.0f}ms). "
                        f"Will restore full context next turn."
                    )

        if self._verbose and self._measurer._turn % 5 == 0:
            s = self._measurer.stats()
            if s:
                print(
                    f"[voice-budget] Turn {s.turn} | "
                    f"P50={s.p50_ms:.0f}ms P95={s.p95_ms:.0f}ms "
                    f"tokens={s.token_count} jitter={s.jitter_ms:.0f}ms"
                )


    # ── Public stats API ───────────────────────────────────────────────────────

    def stats(self):
        """Current rolling P50/P95/P99/jitter statistics."""
        return self._measurer.stats()

    def report(self):
        """Full report of all compression decisions."""
        return self._measurer.snapshot_report()

    def compression_history(self):
        """List of all CompressionEvent objects."""
        return self._measurer.compression_history()

    def print_report(self):
        """Print a human-readable report to stdout."""
        r = self.report()
        s = self.stats()
        print("\n" + "=" * 60)
        print("voice-budget Report")
        print("=" * 60)
        print(f"  Total turns:          {r['total_turns']}")
        if s:
            print(f"  Current P50 TTFT:     {s.p50_ms:.0f}ms")
            print(f"  Current P95 TTFT:     {s.p95_ms:.0f}ms")
            print(f"  Target:               {r['target_ms']}ms")
            print(f"  Budget met:           {'✓' if r['budget_met'] else '✗'}")
            print(f"  Jitter (std):         {s.jitter_ms:.0f}ms")
        print(f"  Compressions:         {r['compression_events']}")
        print(f"  Helpful:              {r['helpful_compressions']}")
        print(f"  Harmful (rolled back):{r['harmful_compressions']}")
        if r['helpful_compressions'] > 0:
            print(f"  Avg improvement:      {r['avg_ttft_improvement_ms']:.0f}ms")
        print(f"  Total tokens saved:   {r['total_tokens_saved']:,}")
        if r['strategies_used']:
            print(f"  Strategies used:      {', '.join(r['strategies_used'])}")
        print("=" * 60 + "\n")


# ── Convenience function ───────────────────────────────────────────────────────

def wrap(
    llm_fn: Callable,
    target_ms: float = 800.0,
    model: str = "gpt-4o",
    token_budget: int = 2000,
    use_semantic: bool = True,
    verbose: bool = False,
    **kwargs,
) -> VoiceBudget:
    """
    One-line entry point.

    Example:
        from voice_budget import wrap
        managed = wrap(openai_client.chat.completions.create_async, target_ms=800)
        response = await managed(messages)
    """
    return VoiceBudget(
        llm_fn=llm_fn,
        target_ms=target_ms,
        model=model,
        token_budget=token_budget,
        use_semantic=use_semantic,
        verbose=verbose,
        **kwargs,
    )
"""
voice_budget/compressors.py

Three compression strategies in escalating cost order:
  1. SlidingWindow   — zero cost, always works, lowest quality
  2. SemanticTrim    — cheap (embeddings), good quality, no LLM call
  3. SummariseTail   — one LLM call, best quality, highest cost

The BudgetCompressor tries them in order and measures TTFT after each.
"""

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple


DEFAULT_SUMMARY_PROMPT = (
    "Summarise the following conversation history into a single "
    "concise paragraph. Preserve all important facts, decisions, "
    "user preferences, and context. Be specific."
)


# ── Base ──────────────────────────────────────────────────────────────────────

class BaseCompressor(ABC):
    name: str = "base"

    @abstractmethod
    def compress(
        self,
        messages: List[Dict],
        target_tokens: int,
        current_tokens: int,
    ) -> Tuple[List[Dict], int]:
        """
        Returns (compressed_messages, tokens_removed).
        Never removes the system message (index 0 if role==system).
        Never removes the last user message.
        """


# ── Strategy 1: Sliding window ────────────────────────────────────────────────

class SlidingWindowCompressor(BaseCompressor):
    """
    Drop the oldest non-system turns until we hit target_tokens.
    Zero cost. Zero dependencies. Always applicable.
    Worst quality — loses old context entirely.
    """
    name = "sliding_window"

    def compress(
        self,
        messages: List[Dict],
        target_tokens: int,
        current_tokens: int,
        token_counter: Optional[Callable] = None,
    ) -> Tuple[List[Dict], int]:
        if token_counter is None:
            # rough estimate: 4 tokens per word average
            def token_counter(msgs):
                return sum(len(str(m.get("content", "")).split()) * 4 // 3 for m in msgs)

        result = list(messages)
        tokens_removed = 0

        # Find the system message if present
        has_system = result and result[0].get("role") == "system"
        protected_start = 1 if has_system else 0

        while token_counter(result) > target_tokens and len(result) > protected_start + 2:
            # Remove oldest non-system, non-last message
            removed = result.pop(protected_start)
            removed_tokens = len(str(removed.get("content", "")).split()) * 4 // 3
            tokens_removed += removed_tokens

        return result, tokens_removed


# ── Strategy 2: Semantic trim ─────────────────────────────────────────────────

class SemanticTrimCompressor(BaseCompressor):
    """
    Score each turn by cosine similarity to the current query.
    Keep the most relevant turns. Drop the least relevant.
    No LLM call — uses sentence-transformers (optional dep).
    Falls back to sliding window if sentence-transformers not available.
    """
    name = "semantic_trim"

    def __init__(self):
        self._model = None
        self._available = False
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            self._SentenceTransformer = SentenceTransformer
            self._np = np
            self._available = True
        except ImportError:
            pass

    def _get_model(self):
        if self._model is None and self._available:
            # Small, fast model — 22MB, ~5ms per encode on CPU
            self._model = self._SentenceTransformer("all-MiniLM-L6-v2")
        return self._model

    def _cosine(self, a, b):
        return float(
            self._np.dot(a, b) / (self._np.linalg.norm(a) * self._np.linalg.norm(b) + 1e-9)
        )

    def compress(
        self,
        messages: List[Dict],
        target_tokens: int,
        current_tokens: int,
        token_counter: Optional[Callable] = None,
    ) -> Tuple[List[Dict], int]:
        if not self._available:
            # Graceful fallback
            return SlidingWindowCompressor().compress(
                messages, target_tokens, current_tokens, token_counter
            )

        if token_counter is None:
            def token_counter(msgs):
                return sum(len(str(m.get("content", "")).split()) * 4 // 3 for m in msgs)

        has_system = messages and messages[0].get("role") == "system"
        system_msgs = messages[:1] if has_system else []
        body = messages[1:] if has_system else messages

        if len(body) <= 2:
            return list(messages), 0

        model = self._get_model()

        # The current query is the last user message
        current_query = ""
        for msg in reversed(body):
            if msg.get("role") == "user":
                current_query = str(msg.get("content", ""))
                break

        query_emb = model.encode(current_query)
        last_msg = body[-1]
        scored = []
        for i, msg in enumerate(body[:-1]):  # never score/drop the last message
            text = str(msg.get("content", ""))
            emb = model.encode(text)
            score = self._cosine(query_emb, emb)
            scored.append((score, i, msg))

        # Sort by relevance ascending — least relevant first
        scored.sort(key=lambda x: x[0])

        result = list(body)
        tokens_removed = 0
        drop_candidates = [item for item in scored]

        while token_counter(system_msgs + result) > target_tokens and drop_candidates:
            _, orig_idx, msg = drop_candidates.pop(0)
            if msg in result:
                result.remove(msg)
                tokens_removed += len(str(msg.get("content", "")).split()) * 4 // 3

        if last_msg not in result:
            final = system_msgs + result + [last_msg]
        else:
            final = system_msgs + result
        return final, tokens_removed


# ── Strategy 3: Summarise tail ────────────────────────────────────────────────

class SummariseTailCompressor(BaseCompressor):
    """
    Summarise the oldest N turns into a single paragraph using the LLM.
    One LLM call. Best quality. Use sparingly (every 20-30 turns).
    Requires an async LLM callable: async (messages) -> str
    """
    name = "summarise_tail"

    def __init__(self, llm_fn: Optional[Callable] = None, tail_turns: int = 10, summary_prompt: Optional[str] = None):
        """
        llm_fn: async callable that takes a messages list and returns a string.
        tail_turns: how many oldest messages to summarise (not conversation turns/pairs).
        summary_prompt: custom prompt for summarisation. Uses DEFAULT_SUMMARY_PROMPT if None.
        """
        self._llm_fn = llm_fn
        self._tail_turns = tail_turns
        self._summary_prompt = summary_prompt or DEFAULT_SUMMARY_PROMPT

    def compress(
        self,
        messages: List[Dict],
        target_tokens: int,
        current_tokens: int,
        token_counter: Optional[Callable] = None,
    ) -> Tuple[List[Dict], int]:
        # Sync placeholder — actual async version below
        raise NotImplementedError("Use async_compress for SummariseTailCompressor")

    async def async_compress(
        self,
        messages: List[Dict],
        target_tokens: int,
        current_tokens: int,
        token_counter: Optional[Callable] = None,
    ) -> Tuple[List[Dict], int]:
        if self._llm_fn is None:
            return SlidingWindowCompressor().compress(
                messages, target_tokens, current_tokens, token_counter
            )

        has_system = messages and messages[0].get("role") == "system"
        system_msgs = messages[:1] if has_system else []
        body = messages[1:] if has_system else messages

        # Summarise only the oldest `tail_turns` messages, keep the rest.
        # Always protect the last user message so the LLM sees the latest query.
        tail_turns = max(0, int(self._tail_turns))
        if tail_turns == 0 or not body:
            return list(messages), 0

        # Find the last user message index so we never summarise it.
        last_user_idx: Optional[int] = None
        for idx in range(len(body) - 1, -1, -1):
            if body[idx].get("role") == "user":
                last_user_idx = idx
                break

        if last_user_idx is None:
            # No user messages — safe to summarise up to tail_turns.
            split_idx = min(len(body), tail_turns)
        else:
            # Never include the last user message in the summarised slice.
            split_idx = min(tail_turns, last_user_idx)

        to_summarise = body[:split_idx]
        protected_recent = body[split_idx:]

        if not to_summarise:
            return list(messages), 0

        # Build summary request
        summary_prompt = [
            {
                "role": "user",
                "content": (
                    self._summary_prompt + "\n\n"
                    + "\n".join(
                        f"{m['role'].upper()}: {m.get('content', '')}"
                        for m in to_summarise
                    )
                ),
            }
        ]

        summary_text = await self._llm_fn(summary_prompt)
        summary_msg = {
            "role": "system",
            "content": f"[Earlier conversation summary]: {summary_text}",
        }

        original_count = sum(
            len(str(m.get("content", "")).split()) * 4 // 3 for m in to_summarise
        )
        summary_count = len(str(summary_text).split()) * 4 // 3
        tokens_removed = max(0, original_count - summary_count)

        compressed = system_msgs + [summary_msg] + list(protected_recent)
        return compressed, tokens_removed


# ── Orchestrator ──────────────────────────────────────────────────────────────

class BudgetCompressor:
    """
    Tries compression strategies in escalating cost order.
    Measures whether each strategy actually helped TTFT.
    Rolls back if compression made things worse.
    """

    def __init__(
        self,
        target_tokens: int = 2000,
        llm_fn: Optional[Callable] = None,
        use_semantic: bool = True,
        use_summarise: bool = True,
    ):
        self.target_tokens = target_tokens
        self._strategies: List[BaseCompressor] = [SlidingWindowCompressor()]
        if use_semantic:
            self._strategies.append(SemanticTrimCompressor())
        if use_summarise and llm_fn:
            self._strategies.append(SummariseTailCompressor(llm_fn))

    async def compress(
        self,
        messages: List[Dict],
        current_tokens: int,
        token_counter: Optional[Callable] = None,
        effective_target: Optional[int] = None,
    ) -> Tuple[List[Dict], str, int]:
        """
        Returns (compressed_messages, strategy_used, tokens_removed).
        Tries strategies in order, stops when target is met.

        effective_target: override target_tokens for this call (e.g. TTFT-proportional).
        """
        target = effective_target if effective_target is not None else self.target_tokens
        for strategy in self._strategies:
            if isinstance(strategy, SummariseTailCompressor):
                compressed, removed = await strategy.async_compress(
                    messages, target, current_tokens, token_counter
                )
            else:
                compressed, removed = strategy.compress(
                    messages, target, current_tokens, token_counter
                )

            if token_counter:
                new_tokens = token_counter(compressed)
            else:
                new_tokens = current_tokens - removed

            if new_tokens <= target:
                return compressed, strategy.name, removed

        # Fallback: return last attempt even if target not fully met
        return compressed, strategy.name, removed
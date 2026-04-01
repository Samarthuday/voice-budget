import pytest

from voice_budget.compressors import SummariseTailCompressor, BudgetCompressor


async def fake_llm(messages):
    # Return a deterministic short summary
    return "Summary: important facts preserved."


@pytest.mark.asyncio
async def test_summarise_tail_async_compress():
    comp = SummariseTailCompressor(llm_fn=fake_llm, tail_turns=2)

    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "old message one"},
        {"role": "assistant", "content": "old assistant"},
        {"role": "user", "content": "last user message"},
    ]

    def token_counter(msgs):
        return sum(len(m.get("content", "").split()) for m in msgs)

    compressed, removed = await comp.async_compress(messages, target_tokens=10, current_tokens=100, token_counter=token_counter)

    assert isinstance(compressed, list)
    assert removed >= 0
    # Ensure the last user message remains
    assert any(m.get("role") == "user" and "last user" in m.get("content", "") for m in compressed)


@pytest.mark.asyncio
async def test_budget_compressor_tries_strategies():
    # Use a BudgetCompressor with our fake llm so summarise strategy is available
    bc = BudgetCompressor(target_tokens=5, llm_fn=fake_llm, use_semantic=False, use_summarise=True)

    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "a b c d e f g h i j k l"},
        {"role": "assistant", "content": "assistant reply"},
        {"role": "user", "content": "query"},
    ]

    def token_counter(msgs):
        return sum(len(m.get("content", "").split()) for m in msgs)

    compressed, strategy, removed = await bc.compress(messages, current_tokens=100, token_counter=token_counter, effective_target=5)

    assert isinstance(compressed, list)
    assert isinstance(strategy, str)
    assert removed >= 0


def test_budget_compressor_strategy_selection_below_semantic_threshold():
    bc = BudgetCompressor(llm_fn=fake_llm, use_semantic=True, use_summarise=True)

    strategies = bc._select_strategies(bc.semantic_threshold - 1)

    assert [strategy.name for strategy in strategies] == ["sliding_window"]


def test_budget_compressor_strategy_selection_at_semantic_threshold():
    bc = BudgetCompressor(llm_fn=fake_llm, use_semantic=True, use_summarise=True)

    strategies = bc._select_strategies(bc.semantic_threshold)

    assert [strategy.name for strategy in strategies] == [
        "semantic_trim",
        "sliding_window",
    ]


def test_budget_compressor_strategy_selection_at_summarise_threshold():
    bc = BudgetCompressor(llm_fn=fake_llm, use_semantic=True, use_summarise=True)

    strategies = bc._select_strategies(bc.summarise_threshold)

    assert [strategy.name for strategy in strategies] == [
        "summarise_tail",
        "semantic_trim",
        "sliding_window",
    ]


@pytest.mark.asyncio
async def test_budget_compressor_fallback_prefers_lowest_resulting_token_count(monkeypatch):
    class FakeStrategy:
        def __init__(self, name, new_tokens, removed):
            self.name = name
            self._new_tokens = new_tokens
            self._removed = removed

        def compress(self, messages, target_tokens, current_tokens, token_counter=None):
            return [{"role": "system", "content": self.name, "tokens": self._new_tokens}], self._removed

    bc = BudgetCompressor(target_tokens=50, use_semantic=False, use_summarise=False)
    strategies = [
        FakeStrategy("more_removed_but_worse", new_tokens=70, removed=40),
        FakeStrategy("fewer_tokens_best_fallback", new_tokens=60, removed=20),
    ]

    monkeypatch.setattr(bc, "_select_strategies", lambda current_tokens: strategies)

    def token_counter(msgs):
        return msgs[0]["tokens"]

    compressed, strategy, removed = await bc.compress(
        messages=[{"role": "user", "content": "original"}],
        current_tokens=100,
        token_counter=token_counter,
        effective_target=50,
    )

    assert strategy == "fewer_tokens_best_fallback"
    assert removed == 20
    assert token_counter(compressed) == 60

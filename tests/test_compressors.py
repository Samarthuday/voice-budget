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

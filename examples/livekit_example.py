import asyncio
import os
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


async def example_livekit_agent():
    """
    Template: integrate VoiceBudgetAgent with LiveKit agent pipeline.
    
    In a real LiveKit agent, you would:
    1. Create a VoiceBudgetAgent instance
    2. Call process_messages() before each LLM call
    3. Monitor stats and compression events
    """
    from voice_budget import VoiceBudgetAgent

    # Initialize the budget agent
    budget = VoiceBudgetAgent(
        target_ms=800,           # TTFT budget in milliseconds
        token_budget=2000,       # max tokens after compression
        model="gpt-4o",          # for tiktoken token counting
        window_size=20,          # rolling window for P50/P95
        use_semantic=True,       # semantic trim (requires sentence-transformers)
        verbose=True,            # log compression decisions
        on_compression=on_compression_callback,
        on_budget_violation=on_budget_violation_callback,
    )

    # Example conversation history (what you'd accumulate in a real agent)
    messages = [
        {
            "role": "system",
            "content": "You are a helpful voice assistant powered by LiveKit and voice-budget.",
        }
    ]

    # Simulate a few turns of conversation
    print("\n=== LiveKit + voice-budget Example ===\n")

    for turn in range(1, 6):
        # Simulate user input
        user_text = f"Turn {turn}: " + " ".join([f"word{i}" for i in range(40)])
        messages.append({"role": "user", "content": user_text})
        print(f"\n[Turn {turn}] User: {user_text[:60]}...")

        # Process messages with voice-budget (measure TTFT + compress if needed)
        try:
            response = await budget.process_messages(
                messages=messages,
                llm_fn=simulated_llm,
            )
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            response = "Error in LLM processing."

        messages.append({"role": "assistant", "content": response})
        print(f"[Turn {turn}] Assistant: {response}")

    # Print final report
    print("\n=== Session Report ===\n")
    report = budget.report()
    if report:
        print(report)

    stats = budget.stats()
    if stats:
        print("\nFinal stats:")
        print(f"  P50 TTFT: {stats.p50_ms:.0f}ms")
        print(f"  P95 TTFT: {stats.p95_ms:.0f}ms")
        print(f"  Jitter:   {stats.jitter_ms:.0f}ms")
        print(f"  Tokens:   {stats.token_count}")


def on_compression_callback(event):
    """Called after a compression event."""
    logger.info(
        f"[Compression] Strategy: {event.strategy}, "
        f"Removed: {event.tokens_before - event.tokens_after} tokens, "
        f"Turn: {event.turn}"
    )


def on_budget_violation_callback(stats):
    """Called when TTFT exceeds budget (P95 > target_ms)."""
    logger.warning(
        f"[Budget Violation] P95 TTFT ({stats.p95_ms:.0f}ms) "
        f"exceeds target ({stats.target_ms:.0f}ms). "
        f"Turn {stats.turn} with {stats.token_count} tokens."
    )


async def simulated_llm(messages: List[Dict], **kwargs) -> str:
    """
    Mock LLM that simulates latency growing with message count.
    Replace with your actual LLM call (e.g., OpenAI API, LLaMA, etc.)
    """
    # Simulate TTFT growing with context size
    latency_ms = 100 + (len(messages) * 5)  # 100ms baseline + 5ms per message
    await asyncio.sleep(latency_ms / 1000.0)
    return f"Response to turn {len(messages) // 2}."


async def example_with_real_openai():
    """
    Example with real OpenAI API (requires OPENAI_API_KEY).
    Uncomment and modify as needed.
    """
    from voice_budget import VoiceBudgetAgent

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set. Skipping real OpenAI example.")
        return

    # Set up OpenAI client
    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=api_key)
    except ImportError:
        print("openai package not installed. Install with: pip install openai")
        return

    async def real_llm(messages, **kwargs):
        """Call real OpenAI API."""
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=100,
            temperature=0.7,
        )
        return response.choices[0].message.content

    budget = VoiceBudgetAgent(
        target_ms=500,
        token_budget=1500,
        model="gpt-4o",
        use_semantic=True,
        verbose=True,
    )

    messages = [
        {"role": "system", "content": "You are a helpful voice assistant."}
    ]

    print("\n=== Real OpenAI Example ===\n")

    for turn in range(1, 4):
        user_text = f"Turn {turn}: Tell me about voice agents in {20 + turn} words."
        messages.append({"role": "user", "content": user_text})
        print(f"\n[Turn {turn}] User: {user_text}")

        response = await budget.process_messages(
            messages=messages,
            llm_fn=real_llm,
        )
        messages.append({"role": "assistant", "content": response})
        print(f"[Turn {turn}] Assistant: {response[:100]}...")

    print("\n=== Report ===")
    print(budget.report())


if __name__ == "__main__":
    # Run the simulated example
    asyncio.run(example_livekit_agent())

    # Uncomment to run with real OpenAI (requires API key):
    # asyncio.run(example_with_real_openai())

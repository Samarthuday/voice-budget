"""
examples/test_real_openai.py

Test voice-budget with a real OpenAI API call.

Usage:
    # Set your API key first:
    #   Windows:  set OPENAI_API_KEY=sk-...
    #   Linux:    export OPENAI_API_KEY=sk-...

    python examples/test_real_openai.py
"""

import asyncio
import os
import sys
import time


async def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: Set OPENAI_API_KEY environment variable first.")
        print("  Windows:  set OPENAI_API_KEY=sk-...")
        print("  Linux:    export OPENAI_API_KEY=sk-...")
        sys.exit(1)

    try:
        from openai import AsyncOpenAI
    except ImportError:
        print("Error: pip install openai")
        sys.exit(1)

    from voice_budget import wrap

    client = AsyncOpenAI(api_key=api_key)

    async def llm(messages, **kwargs):
        resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=100,
        )
        return resp.choices[0].message.content

    managed = wrap(
        llm,
        target_ms=1000,
        token_budget=1500,
        use_semantic=False,
        verbose=True,
    )

    messages = [
        {"role": "system", "content": "You are a concise voice assistant. Keep replies under 2 sentences."}
    ]

    topics = [
        "What is Python?",
        "How does async/await work?",
        "What is TTFT in voice agents?",
        "Explain context window limits.",
        "What causes latency in LLM calls?",
        "How to reduce voice agent latency?",
        "What is tiktoken?",
        "Explain P95 latency.",
        "What is a sliding window?",
        "Summarise what we discussed.",
    ]

    print(f"\nTesting voice-budget with real OpenAI API ({len(topics)} turns)\n")
    print(f"  {'Turn':>4}  {'TTFT':>8}  {'Tokens':>7}  Response")
    print(f"  {'----':>4}  {'----':>8}  {'------':>7}  --------")

    for i, topic in enumerate(topics):
        messages.append({"role": "user", "content": topic})

        t0 = time.perf_counter()
        response = await managed(messages)
        ttft_ms = (time.perf_counter() - t0) * 1000

        messages.append({"role": "assistant", "content": response})

        tokens = managed._measurer.count_tokens(messages)
        print(f"  {i+1:>4}  {ttft_ms:>6.0f}ms  {tokens:>7}  {response[:70]}...")

    print()
    managed.print_report()

    history = managed.compression_history()
    if history:
        print(f"\nCompression events ({len(history)}):")
        for ev in history:
            print(f"  Turn {ev.turn}: {ev.strategy} "
                  f"({ev.tokens_before}→{ev.tokens_after} tokens, "
                  f"delta={ev.delta_ms:.0f}ms)" if ev.delta_ms else
                  f"  Turn {ev.turn}: {ev.strategy} "
                  f"({ev.tokens_before}→{ev.tokens_after} tokens, pending)")
    else:
        print("\nNo compressions triggered (context stayed within budget).")


if __name__ == "__main__":
    asyncio.run(main())

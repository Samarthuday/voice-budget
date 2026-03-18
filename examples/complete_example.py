"""
examples/complete_example.py

Three complete examples showing voice-budget in different setups.
Run any example:
    python examples/complete_example.py raw
    python examples/complete_example.py pipecat
    python examples/complete_example.py openai
"""

import asyncio
import random
import sys
import time

# ─────────────────────────────────────────────────────────────────────────────
# Example 1: Raw asyncio — any LLM, any pipeline
# ─────────────────────────────────────────────────────────────────────────────

async def example_raw():
    """
    Framework-agnostic. Works with any async LLM function.
    Replace `simulated_llm` with your actual LLM call.
    """
    from voice_budget import wrap

    # YOUR LLM FUNCTION — replace with real implementation
    async def simulated_llm(messages, **kwargs):
        """Simulate context accumulation: TTFT grows with message count."""
        await asyncio.sleep(0.1 + len(messages) * 0.04 + random.uniform(0, 0.1))
        return "Simulated response."

    # Wrap it — this is the only change to your existing code
    llm = wrap(
        simulated_llm,
        target_ms=800,          # your TTFT budget
        token_budget=1500,      # compress when context exceeds this
        use_semantic=False,     # True if sentence-transformers installed
        verbose=True,           # print what voice-budget is doing
    )

    # Your normal voice loop — unchanged
    messages = [
        {"role": "system", "content": "You are a helpful voice assistant."}
    ]

    print("Running 25-turn voice conversation...\n")
    for i in range(25):
        # Simulate user speech
        user_text = f"Turn {i+1}: " + " ".join([f"word{j}" for j in range(30)])
        messages.append({"role": "user", "content": user_text})

        # Call LLM through voice-budget — identical to calling simulated_llm directly
        t0 = time.perf_counter()
        response = await llm(messages)
        actual_ttft = (time.perf_counter() - t0) * 1000

        messages.append({"role": "assistant", "content": response})

        s = llm.stats()
        if s:
            print(f"Turn {i+1:>2}: actual={actual_ttft:>5.0f}ms | "
                  f"P95={s.p95_ms:>5.0f}ms | tokens={s.token_count}")

    # Final report
    llm.print_report()


# ─────────────────────────────────────────────────────────────────────────────
# Example 2: OpenAI with real API
# ─────────────────────────────────────────────────────────────────────────────

async def example_openai():
    """
    Real OpenAI GPT-4o integration.
    Requires: pip install openai && export OPENAI_API_KEY=sk-...
    """
    import os
    from voice_budget import wrap

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Set OPENAI_API_KEY environment variable first.")
        return

    try:
        from openai import AsyncOpenAI
    except ImportError:
        print("pip install openai")
        return

    client = AsyncOpenAI(api_key=api_key)

    async def openai_llm(messages, **kwargs):
        resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=100,
        )
        return resp.choices[0].message.content

    llm = wrap(
        openai_llm,
        target_ms=1000,
        token_budget=2000,
        use_semantic=True,
        verbose=True,
    )

    messages = [{"role": "system", "content": "You are a concise voice assistant."}]
    topics = [
        "the weather", "Python programming", "cooking recipes",
        "travel destinations", "machine learning", "music recommendations",
        "exercise tips", "book suggestions", "history facts", "science news",
    ]

    print("Running 10-turn OpenAI conversation with voice-budget...\n")
    for i, topic in enumerate(topics):
        messages.append({"role": "user", "content": f"Tell me briefly about {topic}."})
        response = await llm(messages)
        messages.append({"role": "assistant", "content": response})
        print(f"Turn {i+1}: {response[:60]}...")

    llm.print_report()


# ─────────────────────────────────────────────────────────────────────────────
# Example 3: Pipecat pipeline (conceptual — shows integration pattern)
# ─────────────────────────────────────────────────────────────────────────────

def example_pipecat_code():
    """
    Print the Pipecat integration code.
    Requires a real Pipecat setup to run.
    """
    code = '''
# Complete Pipecat integration with voice-budget

import asyncio
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.elevenlabs import ElevenLabsTTSService
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport

from voice_budget.pipecat_integration import VoiceBudgetProcessor

async def main():
    transport = DailyTransport(
        room_url, token, "Voice Agent",
        DailyParams(audio_out_enabled=True, vad_analyzer=SileroVADAnalyzer())
    )

    stt = DeepgramSTTService(api_key=DEEPGRAM_API_KEY)
    llm = OpenAILLMService(api_key=OPENAI_API_KEY, model="gpt-4o")
    tts = ElevenLabsTTSService(api_key=ELEVENLABS_API_KEY, voice_id=VOICE_ID)

    # voice-budget sits between context aggregator and LLM
    budget = VoiceBudgetProcessor(
        target_ms=800,
        token_budget=2000,
        verbose=True,
        on_budget_violation=lambda stats: print(
            f"Budget violated! P95={stats.p95_ms:.0f}ms"
        ),
    )

    context = OpenAILLMContext([
        {"role": "system", "content": "You are a helpful voice assistant."}
    ])
    context_aggregator = llm.create_context_aggregator(context)

    pipeline = Pipeline([
        transport.input(),
        stt,
        context_aggregator.user(),
        budget,          # ← voice-budget right here
        llm,
        tts,
        transport.output(),
        context_aggregator.assistant(),
    ])

    task = PipelineTask(pipeline, PipelineParams(allow_interruptions=True))
    runner = PipelineRunner()
    await runner.run(task)

asyncio.run(main())
'''
    print(code)


# ─────────────────────────────────────────────────────────────────────────────
# Example 4: Callbacks and monitoring
# ─────────────────────────────────────────────────────────────────────────────

async def example_callbacks():
    """Show how to use callbacks for monitoring and alerting."""
    from voice_budget import VoiceBudget
    from voice_budget.core import CompressionEvent, BudgetStats

    compression_log = []
    violations = []

    def on_compression(event: CompressionEvent):
        compression_log.append({
            "turn": event.turn,
            "strategy": event.strategy,
            "tokens_before": event.tokens_before,
            "tokens_after": event.tokens_after,
        })
        print(f"  [COMPRESSED] Turn {event.turn}: {event.strategy} "
              f"({event.tokens_before}→{event.tokens_after} tokens)")

    def on_violation(stats: BudgetStats):
        violations.append(stats)
        print(f"  [VIOLATION]  P95={stats.p95_ms:.0f}ms > target "
              f"at turn {stats.turn} ({stats.token_count} tokens)")

    async def simulated_llm(messages, **kwargs):
        await asyncio.sleep(0.08 + len(messages) * 0.05)
        return "OK."

    budget = VoiceBudget(
        llm_fn=simulated_llm,
        target_ms=600,
        token_budget=400,
        window_size=6,
        use_semantic=False,
        on_compression=on_compression,
        on_budget_violation=on_violation,
        verbose=False,
    )

    messages = [{"role": "system", "content": "Assistant."}]
    print("Running with callbacks...\n")
    for i in range(20):
        messages.append({"role": "user", "content": f"Turn {i}: " + "x " * 25})
        await budget(messages)
        messages.append({"role": "assistant", "content": "OK."})

    print(f"\n  Total compressions: {len(compression_log)}")
    print(f"  Total violations:   {len(violations)}")
    budget.print_report()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    examples = {
        "raw":       example_raw,
        "openai":    example_openai,
        "pipecat":   lambda: print(example_pipecat_code()) or asyncio.sleep(0),
        "callbacks": example_callbacks,
    }

    cmd = sys.argv[1] if len(sys.argv) > 1 else "raw"
    if cmd not in examples:
        print(f"Usage: python examples/complete_example.py [{' | '.join(examples)}]")
        sys.exit(1)

    fn = examples[cmd]
    if asyncio.iscoroutinefunction(fn):
        asyncio.run(fn())
    else:
        fn()
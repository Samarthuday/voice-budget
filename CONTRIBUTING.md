# Contributing to voice-budget

## Setup

```bash
git clone https://github.com/YOUR_HANDLE/voice-budget
cd voice-budget
pip install -e ".[dev,semantic]"
```

## Running tests

```bash
pytest tests/ -v
```

## Running the demo

```bash
voice-budget demo --turns 20 --target-ms 800 --verbose
```

## Adding a compression strategy

1. Create a new class in `voice_budget/compressors.py` inheriting `BaseCompressor`
2. Implement the `compress(messages, target_tokens, current_tokens)` method
3. Add it to `BudgetCompressor.__init__` in the strategy list
4. Write tests in `tests/test_voice_budget.py`
5. Document it in README.md

## Adding a framework integration

Create `voice_budget/{framework}_integration.py` following the pattern in
`pipecat_integration.py`. The integration should:
- Be framework-specific (import guarded so voice-budget stays lightweight)
- Intercept messages before they reach the LLM
- Measure actual TTFT by timing first-token arrival
- Use `TTFTMeasurer` and `BudgetCompressor` from the core

## Publishing a new version

```bash
# Update version in voice_budget/__init__.py and pyproject.toml
git tag v0.2.0
git push origin v0.2.0
# GitHub Actions will publish to PyPI automatically
```

## Code style

- `ruff check voice_budget/` must pass
- Type hints on all public functions
- Docstrings on all public classes and methods
- No external dependencies in core (numpy + tiktoken only)
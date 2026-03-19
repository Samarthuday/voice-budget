# Contributing to voice-budget

## Setup

```bash
git clone https://github.com/Samarthre/voice-budget
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

Follow these steps to publish a release which will trigger CI to publish the package to PyPI:

1. Update versions

- Update `pyproject.toml`'s `version` field to the new version (MAJOR.MINOR.PATCH).
- Update `voice_budget/__init__.py` `__version__` to the same new version.

2. Run tests and lint locally

```bash
pip install -e ".[dev]"
pytest tests/ -v
ruff check voice_budget/ || true
```

3. Commit and push

```bash
git add pyproject.toml voice_budget/__init__.py
git commit -m "chore(release): bump version X.Y.Z"
git push origin HEAD
```

4. Create an annotated tag and push it

```bash
# Annotated tag recommended
git tag -a vX.Y.Z -m "Release vX.Y.Z"
# Push tag to origin
git push origin vX.Y.Z
```

5. Ensure GitHub has the `PYPI_API_TOKEN` set in repository Settings → Secrets → Actions. The CI will publish the package automatically when it sees a tag pushed that starts with `v`.

Notes & troubleshooting

- PyPI does not allow re-uploading the same version. If `vX.Y.Z` is already published, bump the patch version (e.g., `vX.Y.Z+1`).
- If you need to move an existing tag (not recommended), coordinate with maintainers and force-push carefully:

```bash
git tag -f vX.Y.Z
git push -f origin vX.Y.Z
```

- For non-interactive publish from CI, ensure the `PYPI_API_TOKEN` secret is available and the workflow uses it as `secrets.PYPI_API_TOKEN`.

## Code style

- `ruff check voice_budget/` must pass
- Type hints on all public functions
- Docstrings on all public classes and methods
- No external dependencies in core (numpy + tiktoken only)

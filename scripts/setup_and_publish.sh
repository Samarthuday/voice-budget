#!/usr/bin/env bash
# scripts/setup_and_publish.sh
# Run this once to set up your environment and publish to PyPI.
# Usage:
#   chmod +x scripts/setup_and_publish.sh
#   ./scripts/setup_and_publish.sh

set -e

echo "=== voice-budget setup ==="

# 1. Install in editable mode with all extras
echo "[1/5] Installing dependencies..."
pip install -e ".[dev,semantic]"

# 2. Run tests
echo "[2/5] Running tests..."
pytest tests/ -v --tb=short
echo "✓ All tests passed"

# 3. Run demo smoke test
echo "[3/5] Running demo smoke test..."
voice-budget demo --turns 10 --target-ms 800
echo "✓ Demo passed"

# 4. Build distribution
echo "[4/5] Building distribution..."
pip install hatch
hatch build
echo "✓ Built dist/"
ls -la dist/

# 5. Publish to PyPI
echo "[5/5] Publishing to PyPI..."
echo "You need a PyPI account and API token."
echo "Get one at: https://pypi.org/manage/account/token/"
echo ""
read -p "Enter your PyPI API token (or press Enter to skip): " PYPI_TOKEN

if [ -n "$PYPI_TOKEN" ]; then
    pip install twine
    TWINE_PASSWORD="$PYPI_TOKEN" TWINE_USERNAME="__token__" twine upload dist/*
    echo ""
    echo "✓ Published! Install with: pip install voice-budget"
    echo "  PyPI page: https://pypi.org/project/voice-budget/"
else
    echo "Skipped PyPI publish. Run manually with:"
    echo "  twine upload dist/*"
fi

echo ""
echo "=== Next steps ==="
echo "1. Create GitHub repo: https://github.com/new"
echo "   Name: voice-budget"
echo "   Description: The only voice agent context manager with a TTFT feedback loop"
echo ""
echo "2. Push code:"
echo "   git init && git add . && git commit -m 'feat: initial release v0.1.0'"
echo "   git remote add origin https://github.com/Samarthuday/voice-budget"
echo "   git push -u origin main"
echo ""
echo "3. Add PYPI_API_TOKEN to GitHub Secrets for auto-publish on tags"
echo ""
echo "4. Post on HN as Show HN:"
echo "   Title: Show HN: voice-budget – voice agent context manager with TTFT feedback loop"
echo "   URL: https://github.com/Samarthuday/voice-budget"
echo ""
echo "5. Post in communities:"
echo "   - Pipecat Discord: https://discord.gg/pipecat"  
echo "   - LiveKit Slack: https://livekit.io/join-slack"
echo "   - r/LocalLLaMA"
echo "   - HuggingFace forums"
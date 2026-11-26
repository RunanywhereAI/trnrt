#!/usr/bin/env bash
# The virtualenv for benchmarks/accuracy/gsm8k.sh, as InferenceX's infx/bench/eval/lm_eval.install builds it: lm-eval[api] and
# huggingface-hub >= 1.5, < 2, then lm-evaluation-harness at commit b315ef3b without dependencies; plus what InferenceX's
# serving containers already carry and a clean venv lacks: CPU torch (lm_eval.models.utils imports it) and transformers 5,
# each pinned to the version the published accuracy runs used (torch 2.15.0+cpu, transformers 5.24.1).
#
#   [VENV=~/.venvs/trnrt-lm-eval] benchmarks/accuracy/install_lm_eval.sh
set -euo pipefail
VENV=${VENV:-$HOME/.venvs/trnrt-lm-eval}
[ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
"$VENV/bin/pip" install -q --no-cache-dir "lm-eval[api]" "huggingface-hub>=1.5,<2"
"$VENV/bin/pip" install -q --no-cache-dir --no-deps --force-reinstall \
  "git+https://github.com/EleutherAI/lm-evaluation-harness.git@b315ef3b05176acc9732bb7fdec116abe1ecc476"
"$VENV/bin/pip" install -q --no-cache-dir torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
"$VENV/bin/pip" install -q --no-cache-dir transformers==5.24.1
"$VENV/bin/python" -c "import importlib.metadata as m, torch, transformers, lm_eval.models.openai_completions; \
print('lm_eval', m.version('lm_eval'), 'torch', torch.__version__, 'transformers', transformers.__version__)"

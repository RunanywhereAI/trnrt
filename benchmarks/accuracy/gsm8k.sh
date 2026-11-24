#!/usr/bin/env bash
# GSM8K by InferenceX's own eval: lm-eval at EleutherAI/lm-evaluation-harness b315ef3b with InferenceX's gsm8k.yaml (5-shot,
# exact match, the 1,319 test problems, greedy) and its lm-eval hooks (lm_eval_patch/sitecustomize.py: the answer is
# `content`, or `reasoning_content` when content is empty), through the server's /v1/chat/completions with the chat
# template applied by the server. The same command as InferenceX's infx/bench/eval/lm_eval.run.
#
#   [VENV=DIR] benchmarks/accuracy/gsm8k.sh URL OUT_DIR [CONCURRENCY]
#
# URL: the router of `trnrt serve` (or benchmarks/serve.sh); CONCURRENCY: requests at once (default 32). OUT_DIR: a new
# directory for lm-eval's results and samples. VENV: lm-eval's virtualenv (default ~/.venvs/trnrt-lm-eval; built by
# benchmarks/accuracy/install_lm_eval.sh). MAX_MODEL_LEN (default 1048576): the eval's context; the output budget is
# MAX_MODEL_LEN - 4,096, at most 16,384 tokens (InferenceX's rule).
#
# Refuses a server in the benchmark mode (synthetic acceptance): its drafts are not verified, so its answers are not the
# model's. Speculative builds are checked with real acceptance, the only mode a quality number may come from.
# Pass bar used for this repository's builds: strict match >= 94% (H100 with MTP scores 97.42% on the same eval).
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
[ $# -ge 2 ] || { echo "usage: $0 URL OUT_DIR [CONCURRENCY]" >&2; exit 2; }
url=${1%/} out=$2 c=${3:-32}
VENV=${VENV:-$HOME/.venvs/trnrt-lm-eval}
ctx=${MAX_MODEL_LEN:-1048576}
mt=$(( ctx > 4096 ? ctx - 4096 : ctx / 2 ))
(( mt > 16384 )) && mt=16384
[ ! -e "$out" ] || { echo "$out exists: give each run a new directory" >&2; exit 2; }
[ -x "$VENV/bin/python" ] || { echo "no lm-eval venv at $VENV: run benchmarks/accuracy/install_lm_eval.sh" >&2; exit 2; }
python3 -I "$here/acceptance_mode.py" --expect real --url "$url" || exit 2
model=$(curl -fsS "$url/v1/models" | python3 -I -c 'import json, sys; print(json.load(sys.stdin)["data"][0]["id"])')

export OPENAI_API_KEY=${OPENAI_API_KEY:-EMPTY}
export PYTHONPATH="$here/lm_eval_patch${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$out"
echo "GSM8K: $url, model $model, concurrency $c, context $ctx, max output tokens $mt -> $out"
(cd "$here" && "$VENV/bin/python" -m lm_eval --model local-chat-completions --apply_chat_template --tasks gsm8k.yaml \
  --output_path "$out" --log_samples \
  --model_args "model=$model,base_url=$url/v1/chat/completions,api_key=$OPENAI_API_KEY,eos_string=</s>,max_retries=5,num_concurrent=$c,timeout=1800,tokenized_requests=False,max_length=$ctx" \
  --gen_kwargs "max_tokens=$mt,temperature=0,top_p=1") 2>&1 | tee "$out/lm_eval.log"
"$VENV/bin/python" - "$out" <<'PY'
import glob, json, sys
fs = sorted(glob.glob(f"{sys.argv[1]}/**/results*.json", recursive=True))
r = json.load(open(fs[-1]))["results"]["gsm8k"]
strict, flexible = r.get("exact_match,strict-match"), r.get("exact_match,flexible-extract")
print(f"GSM8K strict-match {strict:.4f} flexible-extract {flexible:.4f} ({'PASS' if strict >= 0.94 else 'FAIL'} at 0.94) {fs[-1]}")
PY

#!/bin/bash
# Install the KVarN KV-cache port into this repo's vLLM 0.27.1 venv:
# copies the new modules into site-packages/vllm and applies the upstream hunks.
# usage: bash kvarn/install.sh            (idempotent-ish: re-copying files is fine;
#        the patch is applied with --forward so a second run is a no-op)
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
PY=${PY:-$REPO/venv/bin/python}
SP=$("$PY" -c 'import vllm, os; print(os.path.dirname(vllm.__file__))' 2>/dev/null | tail -n1)
[ -n "$SP" ] && [ -d "$SP" ] || { echo "cannot import vllm with $PY (README: Setup)"; exit 1; }
cp -r "$HERE/files/vllm/." "$SP/"
patch -p1 -N -r /dev/null -d "$SP" < "$HERE/kvarn-0.27.1.patch" || true
# V2-runner port: lets SPEC=dflash2 run with CTX=huge (KVarN KV + prefix caching, 240k).
# Depends on hunks from both the patches/ set and kvarn-0.27.1.patch, hence applied last.
patch -p1 -N -r /dev/null -d "$SP" < "$HERE/kvarn-v2-runner.patch" || true
find "$SP" -type d -name __pycache__ -path "*kvarn*" -prune -exec rm -rf {} + 2>/dev/null || true
"$PY" - <<'PY'
from typing import get_args
from vllm.config.cache import CacheDType
assert "kvarn_k4v2_g128" in get_args(CacheDType)
from vllm.v1.attention.backends.registry import AttentionBackendEnum
print("KVarN backend:", AttentionBackendEnum.KVARN.get_class().get_name())
from vllm.model_executor.layers.quantization.kvarn.config import KVarNConfig
c = KVarNConfig.from_cache_dtype("kvarn_k4v2_g128", 256)
print("tile bytes", c.tile_bytes, "-> per token per head", c.tile_bytes_aligned // c.group, "B (fp8: 256 B)")
import inspect
from vllm.v1.worker.gpu.model_states import mamba_hybrid
assert "port(kvarn-v2)" in inspect.getsource(mamba_hybrid), "kvarn-v2-runner.patch missing"
print("kvarn-v2 runner port present (DFlash2 + CTX=huge)")
PY
echo "kvarn installed"

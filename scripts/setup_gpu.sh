#!/usr/bin/env bash
# One-shot setup on a rented GPU box (RunPod / Vast / Lambda, PyTorch + CUDA image).
#
#   git clone https://github.com/ziangni-sys/silent-dissent && cd silent-dissent
#   bash scripts/setup_gpu.sh [configs/qwen35_4b.yaml]
#
# Installs the package, the fast Qwen3.5 kernels, downloads the model, the
# official Jacobian lens and the datasets, runs the tests and the pre-flight
# check. Safe to re-run (downloads are cached).
set -euo pipefail
CONFIG=${1:-configs/qwen35_4b.yaml}

# Keep HF downloads on the persistent volume when there is one (RunPod: /workspace).
if [ -d /workspace ] && [ -z "${HF_HOME:-}" ]; then
  export HF_HOME=/workspace/hf_cache
  grep -q "HF_HOME=" ~/.bashrc 2>/dev/null || echo "export HF_HOME=/workspace/hf_cache" >> ~/.bashrc
fi

# Containers often expose every host core (nproc) but only a slice of CPU quota. torch then
# starts one thread per host core, gets throttled, and small CPU ops (the tests) crawl for hours.
if [ -z "${OMP_NUM_THREADS:-}" ]; then
  quota=$(cat /sys/fs/cgroup/cpu.max 2>/dev/null | awk '$1 != "max" {print int($1 / $2)}' || true)
  [ -z "$quota" ] && quota=$(awk 'NR == 1 {q = $1} NR == 2 && q > 0 {print int(q / $1)}' \
    /sys/fs/cgroup/cpu/cpu.cfs_quota_us /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null || true)
  n=${quota:-$(nproc)}; [ "$n" -gt 16 ] && n=16; [ "$n" -lt 1 ] && n=1
  export OMP_NUM_THREADS=$n
  grep -q "OMP_NUM_THREADS=" ~/.bashrc 2>/dev/null || echo "export OMP_NUM_THREADS=$n" >> ~/.bashrc
fi
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

pip install -q -e ".[dev]" "transformers>=5.5" huggingface_hub
# Qwen3.5 linear-attention layers fall back to slow reference PyTorch code without these.
pip install -q flash-linear-attention || echo "WARN: flash-linear-attention not installed (slower, still correct)"
pip install -q causal-conv1d --no-build-isolation || echo "WARN: causal-conv1d not installed (slower, still correct)"

python - "$CONFIG" <<'PY'
import sys
from huggingface_hub import hf_hub_download, snapshot_download
from silent_dissent.config import load_config
cfg = load_config(sys.argv[1])
# original/: Meta's consolidated checkpoint (Llama), a second copy of the weights transformers never reads
print("model:", snapshot_download(cfg["model"]["name"], ignore_patterns=["original/*"]))
kw = cfg["lens"].get("kwargs", {})
if "repo" in kw:
    print("lens:", hf_hub_download(kw["repo"], kw["filename"], revision=kw.get("revision")))
PY

pytest -q
python scripts/check_model.py --config "$CONFIG" --limit 50

#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# Native FedScale launcher with optional Sparrow/DCPU support.
#
# Leaves run-albert.sh and run-model-sweep.sh unchanged.
#
# Usage:
#   ./run-sparrow-fedscale.sh [all|aggregator|executor]
#
# Normal PyTorch CPU:
#   USE_DCPU=False ./run-sparrow-fedscale.sh ...
#
# Sparrow:
#   USE_DCPU=True ./run-sparrow-fedscale.sh ...
# ============================================================

ROLE="${1:-all}"

case "${ROLE}" in
    all|aggregator|executor) ;;
    *)
        echo "Usage: $0 [all|aggregator|executor]" >&2
        exit 2
        ;;
esac

PROJECT_DIR="${PROJECT_DIR:-${HOME}/fedscale-lora}"
FEDSCALE_DIR="${FEDSCALE_DIR:-${HOME}/FedScale}"
BASE_RUNNER="${BASE_RUNNER:-${PROJECT_DIR}/run-albert.sh}"

USE_DCPU="${USE_DCPU:-False}"

DCPU_FRONTEND_SOCK="${DCPU_FRONTEND_SOCK:-/tmp/dgemm-fe.sock}"
DCPU_TRACE_MM="${DCPU_TRACE_MM:-1}"

NUM_PARTICIPANTS="${NUM_PARTICIPANTS:-1}"
NUM_EXECUTORS="${NUM_EXECUTORS:-1}"
ROUNDS="${ROUNDS:-1}"
LOCAL_STEPS="${LOCAL_STEPS:-1}"
CPU_COUNTS="${CPU_COUNTS:-8}"
BATCH_SIZE="${BATCH_SIZE:-4}"

METHOD="${METHOD:-lora}"
MODEL="${MODEL:-meta-llama/Llama-3.2-1B}"
TOPK_RATIO="${TOPK_RATIO:-1.0}"

TEST_RATIO="${TEST_RATIO:-0.1}"
TEST_BSZ="${TEST_BSZ:-2}"
EVAL_INTERVAL="${EVAL_INTERVAL:-999999}"
SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-0}"

PS_IP="${PS_IP:-127.0.0.1}"
PS_PORT="${PS_PORT:-20010}"

RUN_NAME="${RUN_NAME:-fedscale_llama_1b_${METHOD}}"
RESULTS="${RESULTS:-${PROJECT_DIR}/results/${RUN_NAME}-$(date -u +%Y%m%dT%H%M%SZ)}"

VENV_PYTHON="${VENV_PYTHON:-/users/rachs/.venv/qlora/bin/python}"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

[[ -d "${FEDSCALE_DIR}/fedscale" ]] || \
    die "FedScale checkout not found: ${FEDSCALE_DIR}"

[[ -f "${BASE_RUNNER}" ]] || \
    die "Base FedScale runner not found: ${BASE_RUNNER}"

[[ -x "${VENV_PYTHON}" ]] || \
    die "FedScale virtualenv Python not found: ${VENV_PYTHON}"

# Only require Sparrow when DCPU is actually enabled.
if [[ "${USE_DCPU}" == "True" ]] && \
   [[ "${ROLE}" == "executor" || "${ROLE}" == "all" ]]; then
    [[ -S "${DCPU_FRONTEND_SOCK}" ]] || \
        die "Sparrow frontend socket not found: ${DCPU_FRONTEND_SOCK}"
fi

mkdir -p "${RESULTS}"

echo "============================================================"
echo "FedScale launcher"
echo "============================================================"
echo "Role             : ${ROLE}"
echo "FedScale         : ${FEDSCALE_DIR}"
echo "Python           : ${VENV_PYTHON}"
echo "Use DCPU         : ${USE_DCPU}"
echo "Model            : ${MODEL}"
echo "Method           : ${METHOD}"
echo "Rounds           : ${ROUNDS}"
echo "Local steps      : ${LOCAL_STEPS}"
echo "Participants     : ${NUM_PARTICIPANTS}"
echo "Executors        : ${NUM_EXECUTORS}"
echo "CPU threads      : ${CPU_COUNTS}"
echo "Batch size       : ${BATCH_SIZE}"
echo "Aggregator IP    : ${PS_IP}"
echo "Aggregator port  : ${PS_PORT}"
echo "Results          : ${RESULTS}"

if [[ "${USE_DCPU}" == "True" ]]; then
    echo "Frontend socket  : ${DCPU_FRONTEND_SOCK}"
fi

echo "============================================================"

# Verify Python. Only import torch_dcpu when requested.
USE_DCPU="${USE_DCPU}" "${VENV_PYTHON}" - <<'PY'
import os
import torch

print("torch:", torch.__version__)

if os.environ.get("USE_DCPU", "False") == "True":
    import torch_dcpu
    print("torch_dcpu: OK")
else:
    print("torch_dcpu: disabled")
PY

# ------------------------------------------------------------
# Create temporary native copy of run-albert.sh.
# ------------------------------------------------------------

TMP_RUNNER="$(mktemp /tmp/run-albert-native.XXXXXX.sh)"
trap 'rm -f "${TMP_RUNNER}"' EXIT

cp "${BASE_RUNNER}" "${TMP_RUNNER}"

"${VENV_PYTHON}" - "${TMP_RUNNER}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()

changes = [
    (
        "export FEDSCALE_HOME=/opt/FedScale\n"
        "export PYTHONPATH=/opt/FedScale\n",
        'FEDSCALE_HOME="${FEDSCALE_HOME:-/opt/FedScale}"\n'
        'export FEDSCALE_HOME\n'
        'export PYTHONPATH="${PYTHONPATH:-${FEDSCALE_HOME}}"\n'
    ),
    (
        "    --device_conf_file /opt/FedScale/benchmark/dataset/data/device_info/client_device_capacity\n",
        '    --device_conf_file "${FEDSCALE_HOME}/benchmark/dataset/data/device_info/client_device_capacity"\n'
    ),
    (
        "    --data_dir /opt/FedScale/benchmark/dataset/data/blog\n",
        '    --data_dir "${FEDSCALE_HOME}/benchmark/dataset/data/blog"\n'
    ),
    (
        "    --use_cuda False\n",
        '    --use_cuda False\n'
        '    --use_dcpu "${USE_DCPU:-False}"\n'
    ),
]

for old, new in changes:
    if old not in text:
        raise SystemExit(
            "ERROR: run-albert.sh no longer contains an expected block:\n"
            + old
        )
    text = text.replace(old, new, 1)

path.write_text(text)
PY

chmod +x "${TMP_RUNNER}"

# ------------------------------------------------------------
# Native FedScale environment
# ------------------------------------------------------------

export FEDSCALE_HOME="${FEDSCALE_DIR}"
export PYTHONPATH="${FEDSCALE_DIR}"
export PATH="$(dirname "${VENV_PYTHON}"):${PATH}"
export USE_DCPU

if [[ "${USE_DCPU}" == "True" ]]; then
    export DCPU_FRONTEND_SOCK
    export DCPU_TRACE_MM
else
    unset DCPU_FRONTEND_SOCK || true
    unset DCPU_TRACE_MM || true
fi

export OMP_NUM_THREADS="${CPU_COUNTS}"
export MKL_NUM_THREADS="${CPU_COUNTS}"
export OPENBLAS_NUM_THREADS="${CPU_COUNTS}"
export NUMEXPR_NUM_THREADS="${CPU_COUNTS}"

export NUM_PARTICIPANTS
export NUM_EXECUTORS
export ROUNDS
export LOCAL_STEPS
export BATCH_SIZE

export METHOD
export MODEL
export TOPK_RATIO

export TEST_RATIO
export TEST_BSZ
export EVAL_INTERVAL
export SAVE_CHECKPOINT

export PS_IP
export PS_PORT
export RUN_NAME
export RESULTS

export GEMM_TRACE_DIR="${RESULTS}"
export GEMM_TRACE_METHOD="${METHOD}"
export OPERATOR_PROFILE_DIR="${RESULTS}"

echo
echo "Launching native FedScale with USE_DCPU=${USE_DCPU}"
echo

cd "${FEDSCALE_DIR}"
bash "${TMP_RUNNER}" "${ROLE}"

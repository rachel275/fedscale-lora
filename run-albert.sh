#!/usr/bin/env bash
set -euo pipefail

export FEDSCALE_HOME=/opt/FedScale
export PYTHONPATH=/opt/FedScale

# ------------------------------------------------------------
# Experiment parameters
# ------------------------------------------------------------

NUM_EXECUTORS="${NUM_EXECUTORS:-1}"
NUM_PARTICIPANTS="${NUM_PARTICIPANTS:-1}"
ROUNDS="${ROUNDS:-2}"
LOCAL_STEPS="${LOCAL_STEPS:-1}"

METHOD="${METHOD:-full}"
MODEL="${MODEL:-albert-base-v2}"

RUN_NAME="${RUN_NAME:-albert_full}"
RESULTS="${RESULTS:-/workspace/results}"

TOPK_RATIO="${TOPK_RATIO:-0.01}"
EVAL_INTERVAL="${EVAL_INTERVAL:-999999}"

SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-0}"

TEST_RATIO="${TEST_RATIO:-0.01}"
TEST_BSZ="${TEST_BSZ:-32}"

# Use a configurable port so separate/stale runs do not collide.
PS_PORT="${PS_PORT:-20010}"

# ------------------------------------------------------------
# CPU settings
# ------------------------------------------------------------

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

mkdir -p "${RESULTS}"

# ------------------------------------------------------------
# Common FedScale arguments
# ------------------------------------------------------------

COMMON_ARGS=(
    --job_name "${RUN_NAME}"
    --log_path "${RESULTS}"

    --ps_ip 127.0.0.1
    --ps_port "${PS_PORT}"

    --experiment_mode simulation

    --num_executors "${NUM_EXECUTORS}"
    --num_participants "${NUM_PARTICIPANTS}"

    --method "${METHOD}"
    --topk_ratio "${TOPK_RATIO}"

    --test_ratio "${TEST_RATIO}"
    --test_bsz "${TEST_BSZ}"
    
    --task nlp

    --data_set blog
    --data_dir /opt/FedScale/benchmark/dataset/data/blog

    --model "${MODEL}"

    --rounds "${ROUNDS}"
    --eval_interval "${EVAL_INTERVAL}"

    --filter_less 21

    --num_loaders 0
    --local_steps "${LOCAL_STEPS}"

    --learning_rate 4e-5
    --min_learning_rate 1e-5

    --batch_size 2

    --use_cuda False
)

# ------------------------------------------------------------
# Optional checkpoint flag
#
# This assumes config_parser.py uses:
#
# parser.add_argument("--save_checkpoint", action="store_true")
# ------------------------------------------------------------

if [ "${SAVE_CHECKPOINT}" = "1" ]; then
    COMMON_ARGS+=(--save_checkpoint)
fi


# ------------------------------------------------------------
# Process management
# ------------------------------------------------------------

AGG_PID=""
EXECUTOR_PIDS=()

cleanup() {
    local exit_code=$?

    echo
    echo "Stopping FedScale..."

    # Stop executors first.
    for pid in "${EXECUTOR_PIDS[@]:-}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
        fi
    done

    # Then stop aggregator if still alive.
    if [ -n "${AGG_PID}" ] && kill -0 "${AGG_PID}" 2>/dev/null; then
        kill "${AGG_PID}" 2>/dev/null || true
    fi

    wait 2>/dev/null || true

    exit "${exit_code}"
}

trap cleanup EXIT INT TERM


# ------------------------------------------------------------
# Start aggregator
# ------------------------------------------------------------

echo "========================================"
echo "Starting Aggregator"
echo "========================================"

python -u \
    fedscale/cloud/aggregation/aggregator.py \
    "${COMMON_ARGS[@]}" \
    --this_rank 0 \
    > "${RESULTS}/aggregator.log" 2>&1 &

AGG_PID=$!

echo "Aggregator PID: ${AGG_PID}"


# ------------------------------------------------------------
# Wait for aggregator port
# ------------------------------------------------------------

echo "Waiting for aggregator on port ${PS_PORT}..."

aggregator_ready=0

for _ in $(seq 1 60); do

    # Aggregator died before becoming ready.
    if ! kill -0 "${AGG_PID}" 2>/dev/null; then
        echo "ERROR: Aggregator exited before becoming ready."
        tail -50 "${RESULTS}/aggregator.log" || true
        exit 1
    fi

    # Bash TCP connectivity check.
    if (
        echo > "/dev/tcp/127.0.0.1/${PS_PORT}"
    ) >/dev/null 2>&1; then

        aggregator_ready=1
        break

    fi

    sleep 1
done

if [ "${aggregator_ready}" -ne 1 ]; then
    echo "ERROR: Aggregator did not open port ${PS_PORT}."
    tail -50 "${RESULTS}/aggregator.log" || true
    exit 1
fi

echo "Aggregator is listening."


# ------------------------------------------------------------
# Start executors
# ------------------------------------------------------------

echo "========================================"
echo "Starting ${NUM_EXECUTORS} Executors"
echo "========================================"

for RANK in $(seq 1 "${NUM_EXECUTORS}"); do

    python -u \
        fedscale/cloud/execution/executor.py \
        "${COMMON_ARGS[@]}" \
        --this_rank "${RANK}" \
        > "${RESULTS}/executor-${RANK}.log" 2>&1 &

    EXECUTOR_PIDS+=("$!")

    echo "Executor ${RANK} PID: ${EXECUTOR_PIDS[-1]}"
done


# ------------------------------------------------------------
# Experiment information
# ------------------------------------------------------------

echo
echo "========================================"
echo "FedScale experiment running"
echo "========================================"
echo "Run name      : ${RUN_NAME}"
echo "Model         : ${MODEL}"
echo "Dataset       : blog"
echo "Method        : ${METHOD}"
echo "Top-K ratio   : ${TOPK_RATIO}"
echo "Participants  : ${NUM_PARTICIPANTS}"
echo "Executors     : ${NUM_EXECUTORS}"
echo "Rounds        : ${ROUNDS}"
echo "Local steps   : ${LOCAL_STEPS}"
echo "Eval interval : ${EVAL_INTERVAL}"
echo "Checkpointing : ${SAVE_CHECKPOINT}"
echo "Port          : ${PS_PORT}"
echo
echo "Aggregator log:"
echo "  tail -f ${RESULTS}/aggregator.log"
echo
echo "Executor log:"
echo "  tail -f ${RESULTS}/executor-1.log"
echo


# ------------------------------------------------------------
# Wait specifically for aggregator completion
# ------------------------------------------------------------

set +e
wait "${AGG_PID}"
AGG_STATUS=$?
set -e

if [ "${AGG_STATUS}" -ne 0 ]; then
    echo
    echo "ERROR: Aggregator exited with status ${AGG_STATUS}."
    tail -100 "${RESULTS}/aggregator.log" || true
    exit "${AGG_STATUS}"
fi


# ------------------------------------------------------------
# Give executors a chance to process terminate event
# ------------------------------------------------------------

for pid in "${EXECUTOR_PIDS[@]}"; do

    for _ in $(seq 1 10); do

        if ! kill -0 "${pid}" 2>/dev/null; then
            break
        fi

        sleep 1

    done

    # Kill executor if it failed to terminate cleanly.
    if kill -0 "${pid}" 2>/dev/null; then
        echo "WARNING: Executor PID ${pid} did not terminate; stopping it."
        kill "${pid}" 2>/dev/null || true
    fi

    wait "${pid}" 2>/dev/null || true
done


# ------------------------------------------------------------
# Basic run validation
# ------------------------------------------------------------

if ! grep -q "Training of (CLIENT: .* completes" \
    "${RESULTS}"/executor-*.log; then

    echo
    echo "ERROR: No successful client training was recorded."
    exit 1
fi

if ! find "${RESULTS}" \
    -maxdepth 1 \
    -name 'communication-executor-*.jsonl' \
    -size +0c \
    -print -quit \
    | grep -q .; then

    echo
    echo "ERROR: No communication records were generated."
    exit 1
fi

if ! find "${RESULTS}" \
    -maxdepth 1 \
    -name 'gemm-executor-*.jsonl' \
    -size +0c \
    -print -quit \
    | grep -q .; then

    echo
    echo "ERROR: No GEMM records were generated."
    exit 1
fi

if ! find "${RESULTS}" \
    -maxdepth 1 \
    -name 'client-metrics-executor-*.jsonl' \
    -size +0c \
    -print -quit \
    | grep -q .; then

    echo
    echo "ERROR: No client metrics records were generated."
    exit 1
fi

echo
echo "========================================"
echo "FedScale experiment completed successfully"
echo "========================================"


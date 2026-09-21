#!/usr/bin/env bash
set -euo pipefail

export FEDSCALE_HOME=/opt/FedScale
export PYTHONPATH=/opt/FedScale

# ------------------------------------------------------------
# Role
#
# Usage:
#   ./run-albert.sh all
#   ./run-albert.sh aggregator
#   ./run-albert.sh executor
#
# Default: all
# ------------------------------------------------------------

ROLE="${1:-all}"

case "${ROLE}" in
    all|aggregator|executor)
        ;;
    *)
        echo "Usage: $0 [all|aggregator|executor]"
        exit 1
        ;;
esac


# ------------------------------------------------------------
# Experiment parameters
# ------------------------------------------------------------

NUM_EXECUTORS="${NUM_EXECUTORS:-1}"
NUM_PARTICIPANTS="${NUM_PARTICIPANTS:-1}"
ROUNDS="${ROUNDS:-2}"
LOCAL_STEPS="${LOCAL_STEPS:-1}"

# Supported methods include full, lora, qlora, and topk.
# METHOD is supplied by run-model-sweep.sh for sweep runs.
METHOD="${METHOD:-full}"
MODEL="${MODEL:-albert-base-v2}"

RUN_NAME="${RUN_NAME:-albert_full}"
RESULTS="${RESULTS:-/workspace/results}"

# Operator profiling output. The sweep script normally sets this to
# /workspace/results, which is mounted to the host run directory.
export OPERATOR_PROFILE_DIR="${OPERATOR_PROFILE_DIR:-${RESULTS}}"

TOPK_RATIO="${TOPK_RATIO:-0.01}"
EVAL_INTERVAL="${EVAL_INTERVAL:-999999}"

SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-0}"

TEST_RATIO="${TEST_RATIO:-0.01}"
TEST_BSZ="${TEST_BSZ:-32}"

BATCH_SIZE="${BATCH_SIZE:-2}"

# Address executors use to reach the aggregator.
#
# Same-node:
#   PS_IP=127.0.0.1
#
# Two-node:
#   PS_IP=<aggregator node IP>
#
PS_IP="${PS_IP:-127.0.0.1}"
PS_PORT="${PS_PORT:-20010}"

# ------------------------------------------------------------
# Validate training method
# ------------------------------------------------------------

case "${METHOD}" in
    full|lora|qlora|topk)
        ;;
    *)
        echo "ERROR: Unsupported METHOD='${METHOD}'."
        echo "Supported methods: full, lora, qlora, topk"
        exit 1
        ;;
esac


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

    --ps_ip "${PS_IP}"
    --ps_port "${PS_PORT}"

    --experiment_mode simulation
    --simulate_aggregation True

    --device_conf_file /opt/FedScale/benchmark/dataset/data/device_info/client_device_capacity
#    --device_avail_file /opt/FedScale/benchmark/dataset/data/device_info/client_behave_trace

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

    --batch_size "${BATCH_SIZE}"

    --use_cuda False
)


# ------------------------------------------------------------
# Optional checkpoint flag
# ------------------------------------------------------------

if [ "${SAVE_CHECKPOINT}" = "1" ]; then
    COMMON_ARGS+=(--save_checkpoint True)
else
    COMMON_ARGS+=(--save_checkpoint False)
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

    # Each executor is launched in its own session/process group.
    # Kill the whole group so multiprocessing children do not survive
    # if an executor is interrupted or fails.
    for pid in "${EXECUTOR_PIDS[@]:-}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            echo "Stopping executor process group ${pid}..."
            kill -TERM -- "-${pid}" 2>/dev/null || true
        fi
    done

    sleep 2

    # Force-kill anything in an executor group that ignored SIGTERM.
    for pid in "${EXECUTOR_PIDS[@]:-}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            kill -KILL -- "-${pid}" 2>/dev/null || true
        fi
    done

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

if [[ "${ROLE}" == "aggregator" || "${ROLE}" == "all" ]]; then

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


    # --------------------------------------------------------
    # Wait for locally started aggregator
    # --------------------------------------------------------

    echo "Waiting for aggregator on port ${PS_PORT}..."

    aggregator_ready=0

    for _ in $(seq 1 60); do

        if ! kill -0 "${AGG_PID}" 2>/dev/null; then
            echo "ERROR: Aggregator exited before becoming ready."
            tail -50 "${RESULTS}/aggregator.log" || true
            exit 1
        fi

        # The aggregator listens locally, regardless of which IP
        # remote executors use to reach this node.
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
fi


# ------------------------------------------------------------
# Executor-only mode: verify remote aggregator is reachable
# ------------------------------------------------------------

if [[ "${ROLE}" == "executor" ]]; then

    echo "Checking aggregator at ${PS_IP}:${PS_PORT}..."

    aggregator_ready=1

    #for _ in $(seq 1 60); do

     #   if (
      #      echo > "/dev/tcp/${PS_IP}/${PS_PORT}"
       # ) >/dev/null 2>&1; then

        #    aggregator_ready=1
         #   break
      #  fi

       # sleep 1
#  done

    if [ "${aggregator_ready}" -ne 1 ]; then
        echo "ERROR: Cannot reach aggregator at ${PS_IP}:${PS_PORT}."
        exit 1
    fi

    echo "Aggregator is reachable."
fi


# ------------------------------------------------------------
# Start executors
# ------------------------------------------------------------

if [[ "${ROLE}" == "executor" || "${ROLE}" == "all" ]]; then

    echo "========================================"
    echo "Starting ${NUM_EXECUTORS} Executors"
    echo "========================================"

    for RANK in $(seq 1 "${NUM_EXECUTORS}"); do

        setsid python -u \
            fedscale/cloud/execution/executor.py \
            "${COMMON_ARGS[@]}" \
            --this_rank "${RANK}" \
            > "${RESULTS}/executor-${RANK}.log" 2>&1 &

        EXECUTOR_PIDS+=("$!")

        echo "Executor ${RANK} PID: ${EXECUTOR_PIDS[-1]}"
    done
fi

# ------------------------------------------------------------
# Experiment information
# ------------------------------------------------------------

echo
echo "========================================"
echo "FedScale experiment running"
echo "========================================"
echo "Role          : ${ROLE}"
echo "Run name      : ${RUN_NAME}"
echo "Model         : ${MODEL}"
echo "Dataset       : blog"
echo "Method        : ${METHOD}"
echo "Top-K ratio   : ${TOPK_RATIO}"
echo "Profile dir   : ${OPERATOR_PROFILE_DIR}"
echo "Participants  : ${NUM_PARTICIPANTS}"
echo "Executors     : ${NUM_EXECUTORS}"
echo "Rounds        : ${ROUNDS}"
echo "Local steps   : ${LOCAL_STEPS}"
echo "Eval interval : ${EVAL_INTERVAL}"
echo "Checkpointing : ${SAVE_CHECKPOINT}"
echo "Aggregator IP : ${PS_IP}"
echo "Port          : ${PS_PORT}"
echo

if [[ "${ROLE}" == "aggregator" || "${ROLE}" == "all" ]]; then
    echo "Aggregator log:"
    echo "  tail -f ${RESULTS}/aggregator.log"
    echo
fi

if [[ "${ROLE}" == "executor" || "${ROLE}" == "all" ]]; then
    echo "Executor log:"
    echo "  tail -f ${RESULTS}/executor-1.log"
    echo
fi


# ------------------------------------------------------------
# Aggregator/all mode: wait for aggregator completion
# ------------------------------------------------------------

if [[ "${ROLE}" == "aggregator" || "${ROLE}" == "all" ]]; then

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
fi


# ------------------------------------------------------------
# Executor mode: wait for executors
#
# In executor-only mode, they should terminate after receiving
# the aggregator's terminate event.
# ------------------------------------------------------------

if [[ "${ROLE}" == "executor" ]]; then

    for pid in "${EXECUTOR_PIDS[@]}"; do
        wait "${pid}" || true
    done
fi


# ------------------------------------------------------------
# All mode: give local executors time to terminate
# ------------------------------------------------------------

if [[ "${ROLE}" == "all" ]]; then

    for pid in "${EXECUTOR_PIDS[@]}"; do

        for _ in $(seq 1 10); do

            if ! kill -0 "${pid}" 2>/dev/null; then
                break
            fi

            sleep 1
        done

        if kill -0 "${pid}" 2>/dev/null; then
            echo "WARNING: Executor PID ${pid} did not terminate; stopping it."
            kill "${pid}" 2>/dev/null || true
        fi

        wait "${pid}" 2>/dev/null || true
    done
fi


# ------------------------------------------------------------
# Validation
#
# Aggregator-only mode cannot validate local executor files,
# because those are produced on another physical node.
# ------------------------------------------------------------

if [[ "${ROLE}" == "executor" || "${ROLE}" == "all" ]]; then

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
fi

if [[ "${ROLE}" == "executor" || "${ROLE}" == "all" ]]; then
    echo
    echo "Operator profiler outputs:"
    find "${OPERATOR_PROFILE_DIR}" \
        -maxdepth 2 \
        \( -name 'operator-profile.txt' \
           -o -name 'all-operators.txt' \
           -o -name 'matrix-operators.txt' \
           -o -name 'operator-trace.json' \) \
        -print 2>/dev/null || true
fi


echo
echo "========================================"
echo "FedScale ${ROLE} run completed"
echo "========================================"


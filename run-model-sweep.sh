#!/usr/bin/env bash
set -euo pipefail


# ------------------------------------------------------------
# Distributed role
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

AGGREGATOR_IP="${AGGREGATOR_IP:-127.0.0.1}"


# ============================================================
# Multi-model federated training sweep
#
# Methods:
#   - Full fine-tuning
#   - LoRA
#   - QLoRA
#   - Top-K 10%
#   - Top-K 1%
#
# Measurements:
#   - Communication
#   - GEMM workload
#   - Model quality
#
# Sweep-level resilience:
#   - DONE marker per completed run
#   - progress.log
#   - sweep-config.txt
#   - resumable by reusing EXPERIMENT_DIR
# ============================================================


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

PROJECT_DIR="${PROJECT_DIR:-/users/rachs/fedscale-lora}"
FEDSCALE_DIR="${FEDSCALE_DIR:-/users/rachs/FedScale}"
IMAGE="${IMAGE:-fedscale-lora:torch113}"


# ------------------------------------------------------------
# Experiment configuration
# ------------------------------------------------------------

NUM_PARTICIPANTS="${NUM_PARTICIPANTS:-1}"
NUM_EXECUTORS="${NUM_EXECUTORS:-1}"

ROUNDS="${ROUNDS:-1}"
LOCAL_STEPS="${LOCAL_STEPS:-1}"

CPU_COUNTS="${CPU_COUNTS:-8}"

BATCH_SIZE=(4)

# For debugging:
#   EVAL_INTERVAL=999999
#
# For quality/convergence experiments:
#   EVAL_INTERVAL=5
EVAL_INTERVAL="${EVAL_INTERVAL:-5}"

# Whether FedScale should save model checkpoints.
#
# Expected values:
#   0 = disabled
#   1 = enabled
SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-1}"

TEST_RATIO="${TEST_RATIO:-0.1}"
TEST_BSZ="${TEST_BSZ:-2}"

PORT_COUNTER=20010
# ------------------------------------------------------------
# Models
# ------------------------------------------------------------

MODELS=(
    #"albert-base-v2"
    #"bert-base-uncased"
    #"distilbert-base-uncased"
    #"bert-large-uncased"

    # Decoder-only causal language models
    #"meta-llama/Llama-3.1-8B"
    "meta-llama/Llama-3.2-1B"
)

# ------------------------------------------------------------
# Experiment methods
#
# Format:
#
#   method:topk_ratio
#
# topk_ratio is ignored for full and lora.
# ------------------------------------------------------------

EXPERIMENTS=(
    # "full:1.0"
    "lora:1.0"
    "qlora:1.0"
    # "topk:0.10"
    # "topk:0.01"
)


# ------------------------------------------------------------
# Experiment directory
#
# To resume an existing sweep:
#
# EXPERIMENT_DIR=/users/rachs/fedscale-lora/results/model-sweep-... \
# ./run-model-sweep.sh
# ------------------------------------------------------------

if [ -z "${EXPERIMENT_DIR:-}" ]; then

    EXPERIMENT_ID="$(date -u +%Y%m%dT%H%M%SZ)"

    EXPERIMENT_DIR="${PROJECT_DIR}/results/model-sweep-${EXPERIMENT_ID}"

fi

mkdir -p "${EXPERIMENT_DIR}"
mkdir -p "${PROJECT_DIR}/huggingface-cache"

PROGRESS_LOG="${EXPERIMENT_DIR}/progress.log"
CONFIG_FILE="${EXPERIMENT_DIR}/sweep-config.txt"


# ------------------------------------------------------------
# Record sweep configuration
# ------------------------------------------------------------

if [ ! -f "${CONFIG_FILE}" ]; then

    cat > "${CONFIG_FILE}" <<EOF
PROJECT_DIR=${PROJECT_DIR}
FEDSCALE_DIR=${FEDSCALE_DIR}
IMAGE=${IMAGE}
ROLE=${ROLE}
AGGREGATOR_IP=${AGGREGATOR_IP}

NUM_PARTICIPANTS=${NUM_PARTICIPANTS}
NUM_EXECUTORS=${NUM_EXECUTORS}

ROUNDS=${ROUNDS}
LOCAL_STEPS=${LOCAL_STEPS}

CPU_COUNTS=$(printf '%s ' "${CPU_COUNTS}")

EVAL_INTERVAL=${EVAL_INTERVAL}
#SAVE_CHECKPOINT=${SAVE_CHECKPOINT}

MODELS:
$(printf '  %s\n' "${MODELS[@]}")

EXPERIMENTS:
$(printf '  %s\n' "${EXPERIMENTS[@]}")
EOF

fi


# ------------------------------------------------------------
# Initial sweep information
# ------------------------------------------------------------

echo
echo "============================================================"
echo "FedScale Transformer Communication / Compute Sweep"
echo "============================================================"
echo "Role             : ${ROLE}"
echo "Aggregator IP    : ${AGGREGATOR_IP}"
echo "Participants     : ${NUM_PARTICIPANTS}"
echo "Executors        : ${NUM_EXECUTORS}"
echo "Rounds           : ${ROUNDS}"
echo "Local steps      : ${LOCAL_STEPS}"
echo "CPU counts       : ${CPU_COUNTS}"
echo "Eval interval    : ${EVAL_INTERVAL}"
echo "Save checkpoints : ${SAVE_CHECKPOINT}"
echo "Output           : ${EXPERIMENT_DIR}"
echo "============================================================"


# ------------------------------------------------------------
# Convert Hugging Face model names into short names
# ------------------------------------------------------------

model_short_name() {

    local model="$1"

    case "${model}" in

        "albert-base-v2")
            echo "albert_base"
            ;;

        "bert-base-uncased")
            echo "bert_base"
            ;;

        "distilbert-base-uncased")
            echo "distilbert_base"
            ;;

        "bert-large-uncased")
            echo "bert_large"
            ;;

        "meta-llama/Llama-3.2-1B")
            echo "llama_1b"
            ;;

        "meta-llama/Llama-3.1-8B")
            echo "llama_8b"
            ;;

        *)
            echo "${model}" \
                | sed 's|/|_|g' \
                | sed 's|-|_|g'
            ;;

    esac
}


# ------------------------------------------------------------
# Run experiments
# ------------------------------------------------------------

for model in "${MODELS[@]}"; do

    short_model="$(
        model_short_name "${model}"
    )"

    for BATCH_S in "${BATCH_SIZE[@]}"; do

        for experiment in "${EXPERIMENTS[@]}"; do

        IFS=':' read -r method topk_ratio \
            <<< "${experiment}"


        PS_PORT="${PORT_COUNTER}"
        PORT_COUNTER=$((PORT_COUNTER + 1))
        # ----------------------------------------------------
        # Human-readable method name
        # ----------------------------------------------------

        if [ "${method}" = "topk" ]; then

            if [ "${topk_ratio}" = "0.10" ]; then

                method_name="topk10"

            elif [ "${topk_ratio}" = "0.01" ]; then

                method_name="topk1"

            else

                safe_ratio="$(
                    echo "${topk_ratio}" \
                    | tr '.' '_'
                )"

                method_name="topk_${safe_ratio}"

            fi

        else

            method_name="${method}"

        fi


        run_name="${short_model}_${method_name}_${BATCH_S}bs"
        run_dir="${EXPERIMENT_DIR}/${run_name}"

        if [ "${ROLE}" = "aggregator" ]; then
            done_marker="${run_dir}/DONE_AGGREGATOR"
            failed_marker="${run_dir}/FAILED_AGGREGATOR"
        elif [ "${ROLE}" = "executor" ]; then
            done_marker="${run_dir}/DONE_EXECUTOR"
            failed_marker="${run_dir}/FAILED_EXECUTOR"
        else
            done_marker="${run_dir}/DONE"
            failed_marker="${run_dir}/FAILED"
        fi

        mkdir -p "${run_dir}"


        # ----------------------------------------------------
        # Skip successful completed runs
        # ----------------------------------------------------

        if [ -f "${done_marker}" ]; then

            echo
            echo "============================================================"
            echo "Skipping completed run"
            echo "============================================================"
            echo "Run: ${run_name}"

            continue

        fi


        # Remove stale failed marker before retrying.
        rm -f "${failed_marker}"


        # ----------------------------------------------------
        # Record run start
        # ----------------------------------------------------

        echo "$(
            date -u +%FT%TZ
        ) START ${run_name}" \
            >> "${PROGRESS_LOG}"


        echo
        echo "============================================================"
        echo "Starting experiment"
        echo "============================================================"
        echo "Role           : ${ROLE}"
        echo "Aggregator IP  : ${AGGREGATOR_IP}"
        echo "Run name       : ${run_name}"
        echo "Model          : ${model}"
        echo "Method         : ${method}"
        echo "Top-K ratio    : ${topk_ratio}"
        echo "Participants   : ${NUM_PARTICIPANTS}"
        echo "Executors      : ${NUM_EXECUTORS}"
        echo "Rounds         : ${ROUNDS}"
        echo "Local steps    : ${LOCAL_STEPS}"
        echo "CPU threads    : ${CPU_COUNTS}"
        echo "Eval interval  : ${EVAL_INTERVAL}"
        echo "Output         : ${run_dir}"
        echo "============================================================"


        # ----------------------------------------------------
        # Run FedScale container
        # ----------------------------------------------------

        set +e

        sudo docker run \
            --rm \
            --network host \
            --shm-size=8g \
            --cpus="${CPU_COUNTS}" \
            \
            -v "${FEDSCALE_DIR}:/opt/FedScale" \
            -v "${run_dir}:/workspace/results" \
            -v "${PROJECT_DIR}/run-albert.sh:/workspace/run-transformer.sh:ro" \
            -v "${PROJECT_DIR}/huggingface-cache:/root/.cache/huggingface" \
            \
            -e FEDSCALE_HOME=/opt/FedScale \
            -e PYTHONPATH=/opt/FedScale \
            -e HF_TOKEN="${HF_TOKEN}" \
            \
            -e HF_HUB_DOWNLOAD_TIMEOUT=120 \
            -e HF_HUB_ETAG_TIMEOUT=120 \
            \
            -e OMP_NUM_THREADS="${CPU_COUNTS}" \
            -e MKL_NUM_THREADS="${CPU_COUNTS}" \
            -e OPENBLAS_NUM_THREADS="${CPU_COUNTS}" \
            -e NUMEXPR_NUM_THREADS="${CPU_COUNTS}" \
            \
            -e NUM_EXECUTORS="${NUM_EXECUTORS}" \
            -e NUM_PARTICIPANTS="${NUM_PARTICIPANTS}" \
            -e ROUNDS="${ROUNDS}" \
            -e LOCAL_STEPS="${LOCAL_STEPS}" \
            \
            -e MODEL="${model}" \
            -e METHOD="${method}" \
            -e TOPK_RATIO="${topk_ratio}" \
            -e TEST_RATIO="${TEST_RATIO}" \
            -e TEST_BSZ="${TEST_BSZ}" \
	    -e BATCH_SIZE="${BATCH_S}" \
            \
            -e PS_IP="${AGGREGATOR_IP}" \
            -e PS_PORT="${PS_PORT}" \
            -e EVAL_INTERVAL="${EVAL_INTERVAL}" \
            \
            -e RUN_NAME="${run_name}" \
            -e RESULTS=/workspace/results \
            \
            -e GEMM_TRACE_DIR=/workspace/results \
            -e GEMM_TRACE_METHOD="${method}" \
            -e OPERATOR_PROFILE_DIR=/workspace/results \
            \
            "${IMAGE}" \
            bash /workspace/run-transformer.sh "${ROLE}"

        exit_code=$?

        set -e


        # ----------------------------------------------------
        # Handle run success / failure
        # ----------------------------------------------------

        if [ "${exit_code}" -eq 0 ]; then

            touch "${done_marker}"

            echo "$(
                date -u +%FT%TZ
            ) DONE ${run_name}" \
                >> "${PROGRESS_LOG}"

            echo
            echo "Completed:"
            echo "  ${run_name}"

        else

            touch "${failed_marker}"

            echo "$(
                date -u +%FT%TZ
            ) FAILED ${run_name} exit_code=${exit_code}" \
                >> "${PROGRESS_LOG}"

            echo
            echo "WARNING:"
            echo "Run failed:"
            echo "  ${run_name}"
            echo "Exit code:"
            echo "  ${exit_code}"
            echo
            echo "The sweep will continue with the next run."

            continue

        fi


        # ----------------------------------------------------
        # Show generated files
        # ----------------------------------------------------

        if [[ "${ROLE}" == "executor" || "${ROLE}" == "all" ]]; then

        echo
        echo "Communication logs:"

        find "${run_dir}" \
            -maxdepth 1 \
            -name 'communication-executor-*.jsonl' \
            -print


        echo
        echo "GEMM logs:"

        find "${run_dir}" \
            -maxdepth 1 \
            -name 'gemm-executor-*.jsonl' \
            -print


        echo
        echo "Quality log:"

        if [ -f "${run_dir}/quality.jsonl" ]; then

            echo "  ${run_dir}/quality.jsonl"

        else

            echo "  No quality.jsonl yet."

        fi


        echo
        echo "Checkpoints:"

        find "${run_dir}" \
            -type f \
            -name '*.pt' \
            -print


        echo
        echo "Aggregator log:"
        echo "  ${run_dir}/aggregator.log"


        echo
        echo "Executor logs:"

        find "${run_dir}" \
            -maxdepth 1 \
            -name 'executor-*.log' \
            -print

        fi

        if [[ "${ROLE}" == "aggregator" || "${ROLE}" == "all" ]]; then
            echo
            echo "Quality log:"
            if [ -f "${run_dir}/quality.jsonl" ]; then
                echo "  ${run_dir}/quality.jsonl"
            else
                echo "  No quality.jsonl yet."
            fi
            echo
            echo "Aggregator log:"
            echo "  ${run_dir}/aggregator.log"
        fi

        done

    done

done


# ------------------------------------------------------------
# Communication summary
# ------------------------------------------------------------

if [[ "${ROLE}" == "executor" || "${ROLE}" == "all" ]]; then

echo
echo "============================================================"
echo "Generating communication summary"
echo "============================================================"

python3 \
    "${PROJECT_DIR}/summarise-communication.py" \
    "${EXPERIMENT_DIR}" \
    --output \
    "${EXPERIMENT_DIR}/communication-summary.csv" \
    || true


# ------------------------------------------------------------
# Compute summary
# ------------------------------------------------------------

if find "${EXPERIMENT_DIR}" \
    -name 'gemm-executor-*.jsonl' \
    -print -quit \
    | grep -q .
then

    echo
    echo "============================================================"
    echo "Generating compute + communication summary"
    echo "============================================================"

    python3 \
        "${PROJECT_DIR}/summarise-albert-compute.py" \
        "${EXPERIMENT_DIR}" \
        --output \
        "${EXPERIMENT_DIR}/compute-summary.csv" \
        || true

else

    echo
    echo "WARNING:"
    echo "No GEMM trace files found."
    echo "Skipping compute summary."

fi


fi

# ------------------------------------------------------------
# Quality summary
# ------------------------------------------------------------

if [[ "${ROLE}" == "aggregator" || "${ROLE}" == "all" ]]; then

if find "${EXPERIMENT_DIR}" \
    -name 'quality.jsonl' \
    -print -quit \
    | grep -q .
then

    echo
    echo "============================================================"
    echo "Generating model quality summary"
    echo "============================================================"

    python3 \
        "${PROJECT_DIR}/summarise-quality.py" \
        "${EXPERIMENT_DIR}" \
        --output \
        "${EXPERIMENT_DIR}/quality-summary.csv" \
        || true

else

    echo
    echo "WARNING:"
    echo "No quality logs found."
    echo "Skipping quality summary."

fi


fi

# ------------------------------------------------------------
# Final report
# ------------------------------------------------------------

echo
echo "============================================================"
echo "Experiment sweep complete"
echo "============================================================"

echo
echo "Results directory:"
echo "  ${EXPERIMENT_DIR}"

echo
echo "Progress log:"
echo "  ${PROGRESS_LOG}"

echo
echo "Sweep configuration:"
echo "  ${CONFIG_FILE}"

echo
echo "Communication summary:"
echo "  ${EXPERIMENT_DIR}/communication-summary.csv"

echo
echo "Compute summary:"
echo "  ${EXPERIMENT_DIR}/compute-summary.csv"

echo
echo "Quality summary:"
echo "  ${EXPERIMENT_DIR}/quality-summary.csv"

echo
echo "To resume this sweep, run:"
echo
echo "EXPERIMENT_DIR=${EXPERIMENT_DIR} \\"
echo "./run-model-sweep.sh ${ROLE}"
echo

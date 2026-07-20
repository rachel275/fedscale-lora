#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/users/rachs/fedscale-lora}"
FEDSCALE_DIR="${FEDSCALE_DIR:-/users/rachs/FedScale}"
IMAGE="${IMAGE:-fedscale-lora:torch113}"

PARTICIPANT_COUNTS=(1 2 4 8 16)
METHODS=(full lora)

NUM_EXECUTORS="${NUM_EXECUTORS:-4}"
ROUNDS="${ROUNDS:-3}"
LOCAL_STEPS="${LOCAL_STEPS:-1}"

SWEEP_ID="${SWEEP_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
SWEEP_DIR="${PROJECT_DIR}/results/albert-${SWEEP_ID}"

mkdir -p "${SWEEP_DIR}"

echo "Sweep results: ${SWEEP_DIR}"

for method in "${METHODS[@]}"; do
    for participants in "${PARTICIPANT_COUNTS[@]}"; do

        run_name="albert_${method}_p${participants}"
        run_dir="${SWEEP_DIR}/${run_name}"

        mkdir -p "${run_dir}"

        echo
        echo "=================================================="
        echo "Running ALBERT federated experiment"
        echo "Method:       ${method}"
        echo "Participants: ${participants}"
        echo "Executors:    ${NUM_EXECUTORS}"
        echo "Rounds:       ${ROUNDS}"
        echo "Local steps:  ${LOCAL_STEPS}"
        echo "Output:       ${run_dir}"
        echo "=================================================="

        sudo docker run \
            --rm \
            --network host \
            --shm-size=8g \
            -v "${FEDSCALE_DIR}:/opt/FedScale" \
            -v "${run_dir}:/workspace/results" \
            -v "${PROJECT_DIR}/run-albert.sh:/workspace/run-albert.sh:ro" \
            -v "${PROJECT_DIR}/huggingface-cache:/root/.cache/huggingface" \
            -e FEDSCALE_HOME=/opt/FedScale \
            -e PYTHONPATH=/opt/FedScale \
            -e HF_HUB_DOWNLOAD_TIMEOUT=60 \
            -e HF_HUB_ETAG_TIMEOUT=60 \
            -e OMP_NUM_THREADS=1 \
            -e MKL_NUM_THREADS=1 \
            -e OPENBLAS_NUM_THREADS=1 \
            -e NUM_EXECUTORS="${NUM_EXECUTORS}" \
            -e NUM_PARTICIPANTS="${participants}" \
            -e ROUNDS="${ROUNDS}" \
            -e LOCAL_STEPS="${LOCAL_STEPS}" \
            -e METHOD="${method}" \
            -e RUN_NAME="${run_name}" \
            -e RESULTS=/workspace/results \
            "${IMAGE}" \
            bash /workspace/run-albert.sh

        echo
        echo "Completed ${run_name}"
    done
done

python3 "${PROJECT_DIR}/summarise-communication.py" \
    "${SWEEP_DIR}" \
    --output "${SWEEP_DIR}/summary.csv"

echo
echo "=================================================="
echo "Sweep complete"
echo "=================================================="
echo "Summary:"
echo "${SWEEP_DIR}/summary.csv"

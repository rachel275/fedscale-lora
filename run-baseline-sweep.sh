#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/users/rachs/fedscale-lora}"
FEDSCALE_DIR="${FEDSCALE_DIR:-/users/rachs/FedScale}"
IMAGE="${IMAGE:-fedscale-lora:torch113}"

PARTICIPANT_COUNTS=(1 2 4 8 16)

NUM_EXECUTORS="${NUM_EXECUTORS:-4}"
ROUNDS="${ROUNDS:-3}"
LOCAL_STEPS="${LOCAL_STEPS:-1}"

SWEEP_ID="${SWEEP_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
SWEEP_DIR="${PROJECT_DIR}/results/baseline-${SWEEP_ID}"

mkdir -p "${SWEEP_DIR}"

echo "Sweep results: ${SWEEP_DIR}"

for participants in "${PARTICIPANT_COUNTS[@]}"; do
    run_name="full_p${participants}"
    run_dir="${SWEEP_DIR}/${run_name}"

    mkdir -p "${run_dir}"

    echo
    echo "=================================================="
    echo "Running full-model FedAvg"
    echo "Participants: ${participants}"
    echo "Executors:    ${NUM_EXECUTORS}"
    echo "Rounds:       ${ROUNDS}"
    echo "Output:       ${run_dir}"
    echo "=================================================="

    sudo docker run \
        --rm \
        --network host \
        --shm-size=8g \
        -v "${FEDSCALE_DIR}:/opt/FedScale" \
        -v "${PROJECT_DIR}/configs:/workspace/configs:ro" \
        -v "${run_dir}:/workspace/results" \
        -v "${PROJECT_DIR}/run-local.sh:/workspace/run-local.sh:ro" \
        -e FEDSCALE_HOME=/opt/FedScale \
        -e PYTHONPATH=/opt/FedScale \
        -e OMP_NUM_THREADS=1 \
        -e MKL_NUM_THREADS=1 \
        -e OPENBLAS_NUM_THREADS=1 \
        -e NUM_EXECUTORS="${NUM_EXECUTORS}" \
        -e NUM_PARTICIPANTS="${participants}" \
        -e ROUNDS="${ROUNDS}" \
        -e LOCAL_STEPS="${LOCAL_STEPS}" \
        -e RUN_NAME="${run_name}" \
        -e RESULTS=/workspace/results \
        "${IMAGE}" \
        bash /workspace/run-local.sh

    echo "Completed ${run_name}"
done

python3 "${PROJECT_DIR}/summarise-communication.py" \
    "${SWEEP_DIR}" \
    --output "${SWEEP_DIR}/summary.csv"

echo
echo "Sweep complete."
echo "Summary: ${SWEEP_DIR}/summary.csv"

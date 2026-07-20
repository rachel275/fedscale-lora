#!/usr/bin/env bash
set -euo pipefail

export FEDSCALE_HOME=/opt/FedScale
export PYTHONPATH=/opt/FedScale

# -----------------------------
# Experiment parameters
# (can be overridden via environment variables)
# -----------------------------
NUM_EXECUTORS="${NUM_EXECUTORS:-4}"
NUM_PARTICIPANTS="${NUM_PARTICIPANTS:-4}"
ROUNDS="${ROUNDS:-2}"
LOCAL_STEPS="${LOCAL_STEPS:-1}"

RUN_NAME="${RUN_NAME:-femnist_full}"
RESULTS="${RESULTS:-/workspace/results}"

# CPU settings
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "${RESULTS}"

COMMON_ARGS=(
    --job_name "${RUN_NAME}"
    --log_path "${RESULTS}"

    --ps_ip 127.0.0.1
    --ps_port 20010

    --experiment_mode simulation

    --num_executors "${NUM_EXECUTORS}"
    --num_participants "${NUM_PARTICIPANTS}"

    --data_set femnist
    --data_dir /opt/FedScale/benchmark/dataset/data/femnist
    --data_map_file /opt/FedScale/benchmark/dataset/data/femnist/client_data_mapping/train.csv

    --device_conf_file /opt/FedScale/benchmark/dataset/data/device_info/client_device_capacity
    --device_avail_file /opt/FedScale/benchmark/dataset/data/device_info/client_behave_trace

    --model resnet18

    --rounds "${ROUNDS}"
    --eval_interval 1

    --filter_less 21

    --num_loaders 0
    --local_steps "${LOCAL_STEPS}"

    --learning_rate 0.05

    --batch_size 4
    --test_bsz 4

    --use_cuda False
    --save_checkpoint False
)

cleanup() {
    echo
    echo "Stopping FedScale..."
    jobs -pr | xargs -r kill || true
}

trap cleanup EXIT INT TERM

echo "========================================"
echo "Starting Aggregator"
echo "========================================"

python -u \
    fedscale/cloud/aggregation/aggregator.py \
    "${COMMON_ARGS[@]}" \
    --this_rank 0 \
    > "${RESULTS}/aggregator.log" 2>&1 &

sleep 5

echo "========================================"
echo "Starting ${NUM_EXECUTORS} Executors"
echo "========================================"

for RANK in $(seq 1 "${NUM_EXECUTORS}")
do
    python -u \
        fedscale/cloud/execution/executor.py \
        "${COMMON_ARGS[@]}" \
        --this_rank "${RANK}" \
        > "${RESULTS}/executor-${RANK}.log" 2>&1 &
done

echo
echo "========================================"
echo "FedScale running"
echo "========================================"
echo "Run name      : ${RUN_NAME}"
echo "Participants  : ${NUM_PARTICIPANTS}"
echo "Executors     : ${NUM_EXECUTORS}"
echo "Rounds        : ${ROUNDS}"
echo
echo "Aggregator log:"
echo "  tail -f ${RESULTS}/aggregator.log"
echo
echo "Executor log:"
echo "  tail -f ${RESULTS}/executor-1.log"
echo

wait

#!/usr/bin/env bash
set -euo pipefail

export FEDSCALE_HOME=/opt/FedScale
export PYTHONPATH=/opt/FedScale

# Keep CPU usage sensible
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

RESULTS=/workspace/results
mkdir -p "${RESULTS}"

COMMON_ARGS=(
    --job_name femnist_local
    --log_path "${RESULTS}"

    --ps_ip 127.0.0.1
    --ps_port 20010

    --this_rank 0

    --experiment_mode simulation
    --num_executors 4
    --num_participants 4

    --data_set femnist
    --data_dir /opt/FedScale/benchmark/dataset/data/femnist
    --data_map_file /opt/FedScale/benchmark/dataset/data/femnist/client_data_mapping/train.csv

    --device_conf_file /opt/FedScale/benchmark/dataset/data/device_info/client_device_capacity
    --device_avail_file /opt/FedScale/benchmark/dataset/data/device_info/client_behave_trace

    --model resnet18

    --rounds 2
    --eval_interval 1

    --filter_less 21

    --num_loaders 0

    --local_steps 1
    --learning_rate 0.05

    --batch_size 4
    --test_bsz 4

    --use_cuda False
    --save_checkpoint False
)

cleanup() {
    echo
    echo "Stopping FedScale..."
    jobs -pr | xargs -r kill
}
trap cleanup EXIT INT TERM

echo "======================================"
echo "Starting Aggregator"
echo "======================================"

python -u \
    fedscale/cloud/aggregation/aggregator.py \
    "${COMMON_ARGS[@]}" \
    > "${RESULTS}/aggregator.log" 2>&1 &

sleep 5

echo "======================================"
echo "Starting Executors"
echo "======================================"

for RANK in 1 2 3 4
do
    python -u \
        fedscale/cloud/execution/executor.py \
        "${COMMON_ARGS[@]}" \
        --this_rank "${RANK}" \
        > "${RESULTS}/executor-${RANK}.log" 2>&1 &
done

echo
echo "FedScale started."
echo
echo "Aggregator log:"
echo "  tail -f ${RESULTS}/aggregator.log"
echo
echo "Executor logs:"
echo "  tail -f ${RESULTS}/executor-1.log"
echo
wait

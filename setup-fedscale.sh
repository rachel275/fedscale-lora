#!/usr/bin/env bash
#
# Prepare a host for the FedScale full/LoRA/Top-K transformer experiments.
#
# Typical two-node use:
#   node0: ./setup-fedscale-lora.sh aggregator
#   node1: AGGREGATOR_IP=<node0-ip> ./setup-fedscale-lora.sh executor
#
# The script is deliberately idempotent: rerunning it keeps existing repositories,
# datasets, caches, and results. It validates local experiment modifications but
# does not overwrite them or start a training sweep.

set -Eeuo pipefail

usage() {
    cat <<'USAGE'
Usage:
  ./setup-fedscale-lora.sh [all|aggregator|executor]

Common examples:
  ./setup-fedscale-lora.sh aggregator

  AGGREGATOR_IP=128.0.0.10 ./setup-fedscale-lora.sh executor

Important environment variables:
  PROJECT_DIR       Experiment checkout (default: ~/fedscale-lora)
  FEDSCALE_DIR      Modified FedScale checkout (default: ~/FedScale)
  BLOG_URL           Dataset URL (defaults to the official FedScale Blog archive)
  BLOG_ARCHIVE      Download/cache location for blog.tar.gz
  IMAGE             Docker image name (default: fedscale-lora:torch113)
  MODEL             Hugging Face model ID (default: meta-llama/Llama-3.2-1B)
  HF_TOKEN          Required for gated Llama access
  BUILD_IMAGE=0     Reuse an image that is already present
  PRELOAD_MODEL=1   Check access and cache the selected model before a run
  INSTALL_PACKAGES=0
                     Do not install missing apt packages automatically
USAGE
}

ROLE="${1:-all}"
case "${ROLE}" in
    all|aggregator|executor) ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac

USER_NAME="${SUDO_USER:-${USER}}"
USER_HOME="$(getent passwd "${USER_NAME}" | cut -d: -f6)"
if [[ -z "${USER_HOME}" ]]; then
    echo "Could not determine the home directory for ${USER_NAME}." >&2
    exit 1
fi

PROJECT_DIR="${PROJECT_DIR:-${USER_HOME}/fedscale-lora}"
FEDSCALE_DIR="${FEDSCALE_DIR:-${USER_HOME}/FedScale}"
IMAGE="${IMAGE:-fedscale-lora:torch113}"
AGGREGATOR_IP="${AGGREGATOR_IP:-127.0.0.1}"
MODEL="${MODEL:-meta-llama/Llama-3.2-1B}"

DATASET_ROOT="${FEDSCALE_DIR}/benchmark/dataset"
BLOG_DIR="${DATASET_ROOT}/data/blog"
BLOG_URL="${BLOG_URL:-https://fedscale.eecs.umich.edu/dataset/blog.tar.gz}"
BLOG_ARCHIVE="${BLOG_ARCHIVE:-${DATASET_ROOT}/.downloads/blog.tar.gz}"
HF_CACHE_DIR="${PROJECT_DIR}/huggingface-cache"

# Optional switches (0=no, 1=yes).
INSTALL_PACKAGES="${INSTALL_PACKAGES:-1}"
BUILD_IMAGE="${BUILD_IMAGE:-1}"
PRELOAD_MODEL="${PRELOAD_MODEL:-0}"

log() {
    printf '\n[%s] %s\n' "$(date +%H:%M:%S)" "$*"
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

run_as_user() {
    if [[ "$(id -u)" -eq 0 && "${USER_NAME}" != "root" ]]; then
        sudo -u "${USER_NAME}" -- "$@"
    else
        "$@"
    fi
}

need_command() {
    command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

install_host_packages() {
    local missing=()
    local command_name
    for command_name in git wget tar docker; do
        command -v "${command_name}" >/dev/null 2>&1 || missing+=("${command_name}")
    done

    if (( ${#missing[@]} == 0 )); then
        log "Host prerequisites are already installed."
        return 0
    fi

    [[ "${INSTALL_PACKAGES}" == "1" ]] || \
        die "Missing commands: ${missing[*]}. Set INSTALL_PACKAGES=1 or install them manually."
    command -v apt-get >/dev/null 2>&1 || \
        die "Automatic package installation currently supports apt-based hosts only."

    log "Installing host prerequisites."
    sudo apt-get update
    sudo apt-get install -y ca-certificates curl git tar wget docker.io
    sudo systemctl enable --now docker
}

ensure_directories() {
    log "Creating persistent cache, result, and dataset directories."
    run_as_user mkdir -p \
        "${PROJECT_DIR}/results" \
        "${HF_CACHE_DIR}" \
        "$(dirname "${BLOG_ARCHIVE}")" \
        "${DATASET_ROOT}/data" \
        "${BLOG_DIR}/client_data_mapping"
}

exclude_large_dataset_from_git() {
    local exclude_file="${FEDSCALE_DIR}/.git/info/exclude"
    [[ -f "${exclude_file}" ]] || return 0

    local entry
    for entry in \
        "/benchmark/dataset/blog.tar.gz" \
        "/benchmark/dataset/.downloads/" \
        "/benchmark/dataset/data/blog/" \
        "/benchmark/dataset/data/device_info/"; do
        if ! grep -Fqx "${entry}" "${exclude_file}"; then
            printf '%s\n' "${entry}" | run_as_user tee -a "${exclude_file}" >/dev/null
        fi
    done

    # A local exclude cannot hide a file that is already tracked.
    if git -C "${FEDSCALE_DIR}" ls-files --error-unmatch \
        benchmark/dataset/blog.tar.gz >/dev/null 2>&1; then
        echo "WARNING: blog.tar.gz is already tracked by Git. Remove it from the index/history before pushing." >&2
    fi
}

download_blog_dataset() {
    if [[ "${ROLE}" != "executor" && "${ROLE}" != "all" ]]; then
        log "Skipping the Blog dataset on the aggregator node."
        return 0
    fi

    if [[ -d "${BLOG_DIR}/train" && -d "${BLOG_DIR}/test" ]]; then
        log "Blog dataset is already downloaded and extracted."
        run_as_user mkdir -p "${BLOG_DIR}/client_data_mapping"
        return 0
    fi

    if [[ ! -f "${BLOG_ARCHIVE}" ]]; then
        log "Downloading the FedScale Blog dataset (approximately 833 MB)."
        local partial_archive="${BLOG_ARCHIVE}.part"
        run_as_user mkdir -p "$(dirname "${BLOG_ARCHIVE}")"

        # wget -c resumes an interrupted download. The .part suffix prevents an
        # incomplete archive from being treated as ready on the next run.
        run_as_user wget \
            --no-check-certificate \
            --continue \
            --tries=5 \
            --timeout=60 \
            --output-document="${partial_archive}" \
            "${BLOG_URL}"
        run_as_user mv "${partial_archive}" "${BLOG_ARCHIVE}"
    else
        log "Using cached Blog archive at ${BLOG_ARCHIVE}."
    fi

    log "Checking the downloaded archive."
    tar -tzf "${BLOG_ARCHIVE}" >/dev/null || \
        die "The Blog archive is incomplete or corrupt: ${BLOG_ARCHIVE}"

    log "Extracting the Blog dataset."
    local staging_dir
    staging_dir="$(mktemp -d)"
    trap "rm -rf -- '${staging_dir}'" EXIT
    tar -xzf "${BLOG_ARCHIVE}" -C "${staging_dir}"

    local source_blog=""
    if [[ -d "${staging_dir}/blog/train" ]]; then
        source_blog="${staging_dir}/blog"
    elif [[ -d "${staging_dir}/data/blog/train" ]]; then
        source_blog="${staging_dir}/data/blog"
    else
        source_blog="$(find "${staging_dir}" -type d -path '*/blog/train' -printf '%h\n' -quit)"
    fi
    [[ -n "${source_blog}" && -d "${source_blog}/train" ]] || \
        die "The archive does not contain a recognizable blog/train directory."

    run_as_user mkdir -p "${BLOG_DIR}"
    # Copy rather than replace so an interrupted previous extraction can be repaired.
    run_as_user cp -a "${source_blog}/." "${BLOG_DIR}/"
    run_as_user mkdir -p "${BLOG_DIR}/client_data_mapping"
    [[ -d "${BLOG_DIR}/test" ]] || die "Extraction completed without a blog/test directory."

    trap - EXIT
    rm -rf -- "${staging_dir}"
}

validate_device_profiles() {
    local device_info="${DATASET_ROOT}/data/device_info"
    local capacity="${device_info}/client_device_capacity"
    local availability="${device_info}/client_behave_trace"

    if [[ -f "${capacity}" && -f "${availability}" ]]; then
        log "FedScale device capacity and availability traces are present."
        return 0
    fi

    echo "NOTE: Realistic FedScale virtual-time simulation needs both:" >&2
    echo "  ${capacity}" >&2
    echo "  ${availability}" >&2
    echo "The experiment can still run without them if those YAML options are omitted." >&2
}

validate_model_id() {
    if [[ "${MODEL}" == "meta-llama/Llama3.2-1B" ]]; then
        die "Invalid model ID '${MODEL}'. Use meta-llama/Llama-3.2-1B (note the extra hyphen)."
    fi
}

validate_experiment_code() {
    log "Validating the local FedScale experiment modifications."

    local missing=0
    local check_path pattern description
    while IFS='|' read -r check_path pattern description; do
        if [[ ! -f "${FEDSCALE_DIR}/${check_path}" ]] || \
            ! grep -Eq "${pattern}" "${FEDSCALE_DIR}/${check_path}"; then
            echo "MISSING: ${description} (${check_path})" >&2
            missing=1
        fi
    done <<'CHECKS'
fedscale/cloud/config_parser.py|choices=.*full.*lora.*topk|full/LoRA/Top-K parser options
fedscale/cloud/fllibs.py|get_peft_model|PEFT model initialization
fedscale/cloud/execution/torch_client.py|get_peft_model_state_dict|adapter-only LoRA client updates
fedscale/cloud/execution/executor.py|set_lora_weights|adapter-only LoRA model reception
fedscale/cloud/execution/executor.py|communication_metrics|communication instrumentation
fedscale/cloud/execution/executor.py|gemm_trace|GEMM tracer attachment
fedscale/cloud/execution/gemm_trace.py|def attach|GEMM tracer implementation
fedscale/cloud/internal/torch_model_adapter.py|set_lora_weights|LoRA model adapter support
CHECKS

    if (( missing != 0 )); then
        die "This checkout is missing required experiment patches. Use your modified FedScale branch before running the sweep."
    fi

    local peft_wraps
    peft_wraps="$(grep -Ec '^[[:space:]]*model[[:space:]]*=[[:space:]]*get_peft_model' \
        "${FEDSCALE_DIR}/fedscale/cloud/fllibs.py" || true)"
    if (( peft_wraps != 1 )); then
        die "Expected exactly one active get_peft_model() assignment in fllibs.py; found ${peft_wraps}."
    fi

    run_as_user python3 - \
        "${FEDSCALE_DIR}/fedscale/cloud/config_parser.py" \
        "${FEDSCALE_DIR}/fedscale/cloud/fllibs.py" \
        "${FEDSCALE_DIR}/fedscale/cloud/execution/torch_client.py" \
        "${FEDSCALE_DIR}/fedscale/cloud/execution/executor.py" \
        "${FEDSCALE_DIR}/fedscale/cloud/execution/gemm_trace.py" \
        "${FEDSCALE_DIR}/fedscale/cloud/internal/torch_model_adapter.py" <<'PY'
import ast
import pathlib
import sys

for filename in sys.argv[1:]:
    path = pathlib.Path(filename)
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
print("Python syntax checks: OK")
PY
}

build_container_image() {
    [[ "${BUILD_IMAGE}" == "1" ]] || {
        log "Skipping image build because BUILD_IMAGE=${BUILD_IMAGE}."
        return 0
    }

    local dockerfile=""
    local build_context=""
    if [[ -f "${PROJECT_DIR}/Dockerfile" ]]; then
        dockerfile="${PROJECT_DIR}/Dockerfile"
        build_context="${PROJECT_DIR}"
    elif [[ -f "${PROJECT_DIR}/Dockerfile.torch113" ]]; then
        dockerfile="${PROJECT_DIR}/Dockerfile.torch113"
        build_context="${PROJECT_DIR}"
    elif [[ -f "${PROJECT_DIR}/docker/Dockerfile" ]]; then
        dockerfile="${PROJECT_DIR}/docker/Dockerfile"
        build_context="${PROJECT_DIR}"
    elif [[ -f "${PROJECT_DIR}/docker/Dockerfile.torch113" ]]; then
        dockerfile="${PROJECT_DIR}/docker/Dockerfile.torch113"
        build_context="${PROJECT_DIR}"
    elif [[ -f "${FEDSCALE_DIR}/Dockerfile" ]]; then
        dockerfile="${FEDSCALE_DIR}/Dockerfile"
        build_context="${FEDSCALE_DIR}"
    fi
    [[ -n "${dockerfile}" ]] || \
        die "No Dockerfile found. Set BUILD_IMAGE=0 if ${IMAGE} is already installed."

    log "Building Docker image ${IMAGE} from ${dockerfile}."
    sudo docker build -t "${IMAGE}" -f "${dockerfile}" "${build_context}"
}

verify_container_mounts() {
    log "Verifying the FedScale bind mount inside the container."
    local docker_args=(
        run --rm
        -v "${FEDSCALE_DIR}:/opt/FedScale:ro"
        -v "${HF_CACHE_DIR}:/root/.cache/huggingface"
        -e FEDSCALE_HOME=/opt/FedScale
        -e PYTHONPATH=/opt/FedScale
    )
    if [[ -n "${HF_TOKEN:-}" ]]; then
        docker_args+=(-e HF_TOKEN)
    fi
    docker_args+=("${IMAGE}" python -c
        "import fedscale; print('FedScale import: OK')")
    sudo docker "${docker_args[@]}"

    if [[ "${ROLE}" == "executor" || "${ROLE}" == "all" ]]; then
        sudo docker run --rm \
            -v "${DATASET_ROOT}:/opt/FedScale/benchmark/dataset:ro" \
            "${IMAGE}" \
            bash -lc '
                test -d /opt/FedScale/benchmark/dataset/data/blog/train
                test -d /opt/FedScale/benchmark/dataset/data/blog/test
                test -d /opt/FedScale/benchmark/dataset/data/blog/client_data_mapping
                echo "Blog dataset mount: OK"
            '
    fi
}

preload_huggingface_model() {
    [[ "${PRELOAD_MODEL}" == "1" ]] || {
        if [[ "${MODEL}" == meta-llama/* && -z "${HF_TOKEN:-}" ]]; then
            echo "NOTE: export HF_TOKEN with access to ${MODEL} before the first run." >&2
        fi
        return 0
    }

    [[ -n "${HF_TOKEN:-}" ]] || die "PRELOAD_MODEL=1 requires HF_TOKEN."
    log "Checking access to ${MODEL} and preloading its tokenizer/config."
    sudo docker run --rm \
        -e HF_TOKEN \
        -e MODEL="${MODEL}" \
        -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \
        "${IMAGE}" \
        python -c \
        "import os; from transformers import AutoConfig, AutoTokenizer; m=os.environ['MODEL']; AutoConfig.from_pretrained(m, token=True); AutoTokenizer.from_pretrained(m, token=True); print('Hugging Face access: OK')"
}

write_local_environment() {
    local env_file="${PROJECT_DIR}/setup.env"
    umask 077
    {
        printf 'export PROJECT_DIR=%q\n' "${PROJECT_DIR}"
        printf 'export FEDSCALE_DIR=%q\n' "${FEDSCALE_DIR}"
        printf 'export IMAGE=%q\n' "${IMAGE}"
        printf 'export AGGREGATOR_IP=%q\n' "${AGGREGATOR_IP}"
        printf 'export MODEL=%q\n' "${MODEL}"
        printf 'export FEDSCALE_HOME=%q\n' "${FEDSCALE_DIR}"
    } > "${env_file}"
    chown "${USER_NAME}:" "${env_file}" 2>/dev/null || true
}

main() {
    install_host_packages
    need_command git
    need_command tar
    need_command docker

    [[ -d "${FEDSCALE_DIR}/fedscale" ]] || \
        die "FedScale checkout not found at ${FEDSCALE_DIR}."
    [[ -d "${PROJECT_DIR}" ]] || \
        die "Experiment directory not found at ${PROJECT_DIR}."

    ensure_directories
    exclude_large_dataset_from_git
    download_blog_dataset
    validate_device_profiles
    validate_model_id
    validate_experiment_code
    build_container_image
    verify_container_mounts
    preload_huggingface_model
    write_local_environment

    log "Setup complete for role: ${ROLE}"
    echo
    echo "On this node:"
    echo "  source \"${PROJECT_DIR}/setup.env\""
    echo "  cd \"${PROJECT_DIR}\""
    echo "  ./run-model-sweep.sh \"${ROLE}\""
    if [[ "${ROLE}" == "executor" ]]; then
        echo
        echo "Confirm AGGREGATOR_IP points to the aggregator before starting."
    fi
    if [[ "${MODEL}" == meta-llama/* ]]; then
        echo
        echo "For gated Llama models, export HF_TOKEN in the shell that starts the sweep."
    fi
}

main "$@"



#!/usr/bin/env bash
# Evaluate TensorRT engine on KITTI using mmdeploy/tools/test.py
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname "$0")" && pwd)
REPO_DIR=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

# Conda env
if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" != "openmmlab" ]]; then
    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
        conda activate openmmlab
    else
        echo "[ERROR] conda not found to activate openmmlab" >&2
        exit 1
    fi
fi

DEPLOY_CFG=${DEPLOY_CFG:-"${SCRIPT_DIR}/deploy_trt_int8.py"}
MODEL_CFG=${MODEL_CFG:-"${SCRIPT_DIR}/configs/second_hv_secfpn_8xb6-80e_kitti-3d-car.py"}
ENGINE=${ENGINE:-"${REPO_DIR}/work_dirs/SECOND_qat/mmdeploy/second_trt_int8/end2end.engine"}
DEVICE=${DEVICE:-"cuda:0"}

if [[ ! -f "$DEPLOY_CFG" ]]; then
    echo "[ERROR] Deploy config not found: $DEPLOY_CFG" >&2
    exit 1
fi
if [[ ! -f "$MODEL_CFG" ]]; then
    echo "[ERROR] Model config not found: $MODEL_CFG" >&2
    exit 1
fi
if [[ ! -f "$ENGINE" ]]; then
    echo "[ERROR] TensorRT engine not found: $ENGINE" >&2
    exit 1
fi

MMDEPLOY_TEST=$(python - <<'PY'
import mmdeploy, pathlib, sys
path = pathlib.Path(mmdeploy.__file__).resolve().parent / 'tools' / 'test.py'
print(path)
PY
)

python "$MMDEPLOY_TEST" \
    --deploy-cfg "$DEPLOY_CFG" \
    --model-cfg "$MODEL_CFG" \
    --model "$ENGINE" \
    --device "$DEVICE" \
    "$@"

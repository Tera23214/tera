#!/bin/bash
# qsub template for terao_gamp_gaussian/Dence_Alternating/random_graph_version/run_gamp.py
#
# Before submitting, set the queue / group for your project, either by:
#   1. uncommenting and editing the #PBS lines below, or
#   2. passing them to qsub directly:
#      qsub -q <gpu_queue> --group=<your_group> run_gamp_qsub.sh
#      qsub -q <gpu_queue> --group=<your_group> -v DEVICES=0,1,N=400 run_gamp_qsub.sh
#
#PBS -N gamp_random_graph
#PBS -l elapstim_req=24:00:00
##PBS -q <gpu_queue>
##PBS --group=<your_group>
##PBS -l gpunum_job=1
##PBS -l cpunum_job=8
##PBS --venode=1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
RESULTS_ROOT="${RESULTS_ROOT:-$SCRIPT_DIR/results}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"

module purge >/dev/null 2>&1 || true
module load BaseGPU/2026
module load BasePy/2026

if [ -n "${VENV_PATH:-}" ]; then
    . "$VENV_PATH/bin/activate"
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

CMD=(
    "$PYTHON_BIN"
    -u
    "$SCRIPT_DIR/run_gamp.py"
    --M "${M:-50}"
    --alpha-start "${ALPHA_START:-0.2}"
    --alpha-stop "${ALPHA_STOP:-5.0}"
    --alpha-step "${ALPHA_STEP:-0.2}"
    --max-steps "${MAX_STEPS:-50000}"
    --num-replicas "${NUM_REPLICAS:-5}"
    --shared-seed "${SHARED_SEED:-1}"
    --student-seed-base "${STUDENT_SEED_BASE:-100}"
    --torch-threads "${TORCH_THREADS:-1}"
    --save-every-replicas "${SAVE_EVERY_REPLICAS:-5}"
    --results-root "$RESULTS_ROOT"
)

if [ -n "${N:-}" ]; then
    CMD+=(--N "$N")
else
    CMD+=(--N1 "${N1:-200}" --N2 "${N2:-200}")
fi

if [ -n "${DAMPING:-}" ]; then
    CMD+=(--damping "$DAMPING")
fi

if [ -n "${DAMPING_SCHEDULE:-}" ]; then
    CMD+=(--damping-schedule "$DAMPING_SCHEDULE")
fi

if [ -n "${BETA_SCALE:-}" ]; then
    CMD+=(--beta-scale "$BETA_SCALE")
fi

if [ -n "${BETA_MAX:-}" ]; then
    CMD+=(--beta-max "$BETA_MAX")
fi

if [ -n "${NOISE_VAR:-}" ]; then
    CMD+=(--noise-var "$NOISE_VAR")
fi

if [ -n "${CONVERGENCE_THRESHOLD:-}" ]; then
    CMD+=(--convergence-threshold "$CONVERGENCE_THRESHOLD")
fi

if [ -n "${DEVICES:-}" ]; then
    CMD+=(--devices "$DEVICES")
elif [ "${ALLOW_CPU:-0}" = "1" ]; then
    CMD+=(--allow-cpu --cpu-workers "${CPU_WORKERS:-1}")
fi

if [ "${DETERMINISTIC:-0}" = "1" ]; then
    CMD+=(--deterministic)
fi

LOG_FILE="$LOG_DIR/run_gamp_${PBS_JOBID:-local}_$(date +%Y%m%d_%H%M%S).log"

printf 'Job started: %s\n' "$(date)"
printf 'Working directory: %s\n' "$REPO_ROOT"
printf 'Results root: %s\n' "$RESULTS_ROOT"
printf 'Log file: %s\n' "$LOG_FILE"
printf 'Command:'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}" "$@" 2>&1 | tee "$LOG_FILE"

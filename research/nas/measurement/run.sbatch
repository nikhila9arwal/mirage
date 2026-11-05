#!/usr/bin/env bash
#SBATCH --account=bgpd-dtai-gh
#SBATCH --partition=ghx4,ghx4-interactive
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:35:00
#SBATCH --job-name=nas-fifo
#SBATCH --output=nas-fifo-%j.log
set -euo pipefail
# Submit from the repository root. sbatch copies this file to a spool directory,
# so BASH_SOURCE would not locate the source tree while the job is running.
cd "${SLURM_SUBMIT_DIR:?Submit this script with sbatch from the repository root}"
TASK_SESSION="$PWD/research/nas"
test -f "$TASK_SESSION/measurement/measure.py"
TASK_PYTHON="${NAS_PYTHON:-python3}"
: "${NAS_WORKLOAD:?Set NAS_WORKLOAD to the tokenized 64-prompt fixture}"
: "${NAS_HF_REFERENCE:?Set NAS_HF_REFERENCE to its independent HF report}"
: "${NAS_BASELINE_OUTPUTS:?Set NAS_BASELINE_OUTPUTS to preserved NAS outputs.pt}"
export OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
ulimit -c 0
TASK_OUTPUT="$TASK_SESSION/results/job_${SLURM_JOB_ID}"
mkdir -p "$TASK_SESSION/results"
mkdir "$TASK_OUTPUT"
TASK_VARIANTS=(control_histogram control_fifo diagnostic_fifo)
# Compile the small replay beforehand with tests/run_checks.py --compile-gpu.
srun --ntasks=1 timeout --signal=TERM --kill-after=10s 60 \
  "$TASK_SESSION/.build/fifo_gpu_baseline" > "$TASK_OUTPUT/fifo_gpu_baseline.log" 2>&1
srun --ntasks=1 timeout --signal=TERM --kill-after=30s 2040 \
  "$TASK_PYTHON" -B -u "$TASK_SESSION/measurement/measure.py" \
  --workload "$NAS_WORKLOAD" --hf-reference "$NAS_HF_REFERENCE" \
  --baseline-outputs "$NAS_BASELINE_OUTPUTS" \
  --variants "${TASK_VARIANTS[@]}" --completion-budget 16 \
  --output "$TASK_OUTPUT/measurement" --budget-seconds 2000

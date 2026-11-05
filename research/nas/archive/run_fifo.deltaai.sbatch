#!/usr/bin/env bash
#SBATCH --account=bgpd-dtai-gh
#SBATCH --partition=ghx4,ghx4-interactive
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:35:00
#SBATCH --job-name=nas-fifo-drain16
#SBATCH --output=/work/hdd/bgpd/elu1/nas_selector_fifo_20261104T220853Z/logs/fifo-%j.log
set -euo pipefail
cd /u/elu1/mpk_selector_fifo_20261104T220853Z
# One shared checkpoint load; no slow legacy-selector repeat. Match the
# uninstrumented histogram/FIFO controls, then profile FIFO maintenance too.
TASK_COMPLETION_BUDGET="${1:-16}"
case "$TASK_COMPLETION_BUDGET" in
  1|16) ;;
  *) echo "Expected completion budget 1 or 16" >&2; exit 2 ;;
esac
export PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/25.5/cuda/12.9/bin:/opt/cray/pe/gcc-native/14/bin:$PATH
export OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
ulimit -c 0
TASK_OUTPUT="/work/hdd/bgpd/elu1/nas_selector_fifo_20261104T220853Z/results/job_${SLURM_JOB_ID}"
mkdir "$TASK_OUTPUT"
srun --ntasks=1 timeout --signal=TERM --kill-after=10s 120 \
  tests/fifo_gpu > "$TASK_OUTPUT/fifo_gpu_tests.log" 2>&1
nvidia-smi --query-gpu=timestamp,index,uuid,power.draw,clocks.sm,clocks.mem,temperature.gpu,memory.used,utilization.gpu,utilization.memory --format=csv -l 5 > "$TASK_OUTPUT/gpu_telemetry.csv" &
TASK_MONITOR=$!
trap 'kill "$TASK_MONITOR" 2>/dev/null || true' EXIT
srun --ntasks=1 timeout --signal=TERM --kill-after=30s 2040 \
 /u/elu1/venvs/reoalece/bin/python -B -u measurement/measure.py \
 --variants control_histogram control_fifo diagnostic_fifo \
 --completion-budget "$TASK_COMPLETION_BUDGET" \
 --output "$TASK_OUTPUT/measurement" --budget-seconds 1950

#!/bin/bash -l
#SBATCH -J waves_ddf_randoms
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#SBATCH -p general              # <-- change if needed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=1:00:00
#SBATCH --array=0-3

set -euo pipefail

# --- Load your Python environment ---
module purge || true

export PYTHONNOUSERSITE=1


ENV_PY=/its/home/sp624/.conda/envs/regions_ev/bin/python

$ENV_PY -c "import sys; print(sys.executable)"

mkdir -p logs

# ---- user config ----
SCRIPT="/its/home/sp624/waves_masking/randoms_code/waves_randoms/make_waves_ddf_randoms.py"  # <-- change
OUTDIR="/mnt/lustre/projects/astro/general/sp624/waves_randoms/"            # <-- change
MASKDIR="/its/home/sp624/waves_masking/randoms_code/waves_randoms/masks"                              # <-- change to HPC path if different

N=100000000 # CHANGE to 200000000
SEED=42  #1st batch = 142, second batch =42
CHUNK=10000000
BLOCK=10000000

# Map array index -> region
regions=(WD01 WD02 WD03 WD10)
REGION="${regions[$SLURM_ARRAY_TASK_ID]}"

# Per-region masks from your listing
if [[ "$REGION" == "WD01" ]]; then
  STARMASK="${MASKDIR}/WD01_stars.dat"
  GHOSTMASK="${MASKDIR}/WD01_ghosts.dat"
elif [[ "$REGION" == "WD02" ]]; then
  STARMASK="${MASKDIR}/WD02_stars.dat"
  GHOSTMASK="${MASKDIR}/WD02_ghosts.dat"
elif [[ "$REGION" == "WD03" ]]; then
  STARMASK="${MASKDIR}/WD03_stars.dat"
  GHOSTMASK="${MASKDIR}/WD03_ghosts.dat"
elif [[ "$REGION" == "WD10" ]]; then
  STARMASK="${MASKDIR}/WD10_stars.dat"
  GHOSTMASK="${MASKDIR}/WD10_ghosts.dat"
else
  echo "Unknown region: $REGION" >&2
  exit 2
fi

echo "Running region=${REGION} job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID}"
echo "Using masks:"
echo "  STARMASK=${STARMASK}"
echo "  GHOSTMASK=${GHOSTMASK}"
echo "Start: $(date)"

$ENV_PY -u "$SCRIPT" \
  --region "$REGION" \
  --nrandoms "$N" \
  --seed "$SEED" \
  --chunk-size "$CHUNK" \
  --block-size "$BLOCK" \
  --save-location "$OUTDIR" \
  --starmask-path "$STARMASK" \
  --ghostmask-path "$GHOSTMASK" \
  --no-polygon \

echo "Done: $(date)"


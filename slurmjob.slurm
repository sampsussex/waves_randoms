#!/bin/bash -l
#SBATCH -J waves_wide_randoms
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#SBATCH -p general              # <-- change if needed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=2:00:00
#SBATCH --array=0-2

set -euo pipefail

# --- Load your Python environment ---
module purge || true

export PYTHONNOUSERSITE=1


ENV_PY=/its/home/sp624/.conda/envs/regions_ev/bin/python

$ENV_PY -c "import sys; print(sys.executable)"

mkdir -p logs

# ---- user config ----
SCRIPT="/its/home/sp624/waves_masking/randoms_code/waves_randoms/make_waves_wide_randoms.py"  # <-- change
OUTDIR="/mnt/lustre/projects/astro/general/sp624/waves_randoms/"            # <-- change
MASKDIR="/its/home/sp624/waves_masking/randoms_code/waves_randoms/masks"                              # <-- change to HPC path if different

N=1000000000 # CHANGE to 100000000
SEED=142 #1st batch = 142, second batch =42
CHUNK=10000000
BLOCK=10000000

# Map array index -> region
regions=(waves-wide_n waves-wide_s waves-deep)
REGION="${regions[$SLURM_ARRAY_TASK_ID]}"

# Per-region masks from your listing
if [[ "$REGION" == "waves-wide_n" ]]; then
  STARMASK="${MASKDIR}/starmask_waves_n.dat"
  GHOSTMASK="${MASKDIR}/ghostmask_waves_n.dat"
  POLYMASK="${MASKDIR}/ngc_n.dat"
elif [[ "$REGION" == "waves-wide_s" ]]; then
  STARMASK="${MASKDIR}/starmask_waves_s.dat"
  GHOSTMASK="${MASKDIR}/ghostmask_waves_s.dat"
  POLYMASK="${MASKDIR}/ngc_s.dat"
elif [[ "$REGION" == "waves-deep" ]]; then
  STARMASK="${MASKDIR}/starmask_waves_s.dat"
  GHOSTMASK="${MASKDIR}/ghostmask_waves_s.dat"
  POLYMASK="${MASKDIR}/ngc_s.dat"
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
  --polygonmask-path "$POLYMASK" \


echo "Done: $(date)"


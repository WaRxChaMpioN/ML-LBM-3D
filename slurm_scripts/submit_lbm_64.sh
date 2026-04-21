#!/bin/bash
#SBATCH --job-name=lbm64
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=1-10%2
#SBATCH --output=/beegfs/scratch/kaushal_jha/lbm64_%A_%a.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kaushal_jha@mines.edu

echo "=== LBM 64 Task ${SLURM_ARRAY_TASK_ID} START: $(date) ==="
export OPENBLAS_NUM_THREADS=1
module load apps/python3/2024.05
module load libs/cuda/12.2
conda activate mlenv

DATASET=/beegfs/scratch/kaushal_jha/velML3DdistDataset_64
cd /beegfs/scratch/kaushal_jha/ML_LBM_3D

CHUNK=100
START=$(( (SLURM_ARRAY_TASK_ID - 1) * CHUNK + 1 ))
END=$(( SLURM_ARRAY_TASK_ID * CHUNK ))

if   [ ${END} -le 800 ]; then   SUBSET="train"
elif [ ${START} -ge 901 ]; then SUBSET="test"
else                             SUBSET="validation"
fi

echo "Samples ${START}-${END} | Subset: ${SUBSET}"

/home/m10957667/.conda/envs/mlenv/bin/python3 run_lbm_3d.py \
    --dataset_dir ${DATASET} \
    --subset      ${SUBSET} \
    --start_id    ${START} \
    --end_id      ${END}

echo "=== LBM Task ${SLURM_ARRAY_TASK_ID} DONE: $(date) ==="
echo "Done: $(ls ${DATASET}/${SUBSET}_outputs/ 2>/dev/null | wc -l) files"

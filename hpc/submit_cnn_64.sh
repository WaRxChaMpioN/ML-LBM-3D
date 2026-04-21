#!/bin/bash
#SBATCH --job-name=CNN3D_64
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=64G
#SBATCH --time=5-00:00:00
#SBATCH --output=/beegfs/scratch/kaushal_jha/cnn3d_64_%j.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kaushal_jha@mines.edu

echo "=== CNN 64³ TRAIN START: $(date) ==="
export OPENBLAS_NUM_THREADS=1
module load apps/python3/2024.05
module load libs/cuda/12.2
conda activate mlenv

DATASET=/beegfs/scratch/kaushal_jha/velML3DdistDataset_64
cd /beegfs/scratch/kaushal_jha/ML_LBM_3D

echo "Dataset check:"
echo "  Train:  $(ls ${DATASET}/train_outputs/ 2>/dev/null | wc -l)/800"
echo "  Val:    $(ls ${DATASET}/validation_outputs/ 2>/dev/null | wc -l)/100"

/home/m10957667/.conda/envs/mlenv/bin/python3 poreScaleVelMain.py \
    --train           True \
    --nDims           3 \
    --numFilters      8 \
    --baseKernelSize  8 \
    --residual-blocks 1 \
    --gLoss           L2 \
    --alpha           01 \
    --inputType       dist \
    --outputType      vel \
    --width           64 \
    --height          64 \
    --depth           64 \
    --batch-size      1 \
    --valPlot         False \
    --learnRate       1e-4 \
    --num-epochs      500 \
    --epoch-step      50 \
    --trainIDs        1-800 \
    --valIDs          801-900 \
    --dataset         ${DATASET}

echo "=== CNN 64³ TRAIN DONE: $(date) ==="

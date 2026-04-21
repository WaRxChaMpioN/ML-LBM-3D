#!/bin/bash
#SBATCH --job-name=infer_64
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100:1
#SBATCH --time=1:00:00
#SBATCH --output=/beegfs/scratch/kaushal_jha/infer64_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kaushal_jha@mines.edu

module load apps/python3/2024.05
module load libs/cuda/12.2
conda activate mlenv

cd /beegfs/scratch/kaushal_jha/ML_LBM_3D

echo "=== INFERENCE START: $(date) ==="

/home/m10957667/.conda/envs/mlenv/bin/python3 poreScaleVelMain.py \
    --train False \
    --test  True \
    --nDims 3 \
    --numFilters      8 \
    --baseKernelSize  4 \
    --residual-blocks 1 \
    --gLoss L2 --alpha 0.1 \
    --inputType  dist \
    --outputType vel \
    --width 64 --height 64 --depth 64 \
    --batch-size 1 \
    --testInputs /beegfs/scratch/kaushal_jha/velML3DdistDataset_64/test_inputs \
    --restore ./outputs/20260407-205356-velCNN-kaushal-1-4-8-L2-0.1-1-0-1.0/ckpt-23

echo "=== INFERENCE DONE: $(date) ==="
echo "Predictions: $(ls /beegfs/scratch/kaushal_jha/velML3DdistDataset_64/test_inputs/CNNOutputs-ckpt-9-*/ 2>/dev/null | wc -l) files"

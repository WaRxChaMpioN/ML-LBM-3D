#!/bin/bash
#SBATCH --job-name=cnn_64_v2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=logs/cnn_64_v2_%j.out
#SBATCH --error=logs/cnn_64_v2_%j.err

# K4N8 3D CNN training (v2 — L2 + physics continuity loss, alpha=0.1)
# Best model: val loss 0.0076 at epoch 510, R^2 ~0.85
# Wendian HPC, NVIDIA V100 32 GB

module load python/3.11 cuda/11.8
source ~/venv/ml-lbm/bin/activate

mkdir -p logs checkpoints

echo "Starting CNN training v2 (physics loss): $(date)"

python3 poreScaleVelMain.py \
    --train True \
    --nDims 3 \
    --width 64 \
    --height 64 \
    --depth 64 \
    --numFilters 8 \
    --baseKernelSize 4 \
    --learnRate 1e-4 \
    --batch-size 4 \
    --epochs 1000 \
    --alpha 0.1 \
    --dataset ./dataset_64 \
    --checkpoint ./checkpoints/model_k4n8_v2 \
    --n_train 800 \
    --n_val 100

echo "CNN training v2 complete: $(date)"
echo "Best checkpoint at: checkpoints/model_k4n8_v2_best.h5"

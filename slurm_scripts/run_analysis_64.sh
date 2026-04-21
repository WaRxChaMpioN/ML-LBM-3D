#!/bin/bash
#SBATCH --job-name=analysis_64
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/beegfs/scratch/kaushal_jha/analysis64_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kaushal_jha@mines.edu

echo "=== ANALYSIS START: $(date) ==="
export OPENBLAS_NUM_THREADS=1
module load apps/python3/2024.05
conda activate mlenv

SCRATCH=/beegfs/scratch/kaushal_jha
DATASET=${SCRATCH}/velML3DdistDataset_64
CNN_DIR=${DATASET}/test_inputs/CNNOutputs-ckpt-23-velMLdistDataset
OUT_DIR=${SCRATCH}/ML_LBM_3D/analysis_3d_64_ckpt23

echo "CNN dir: ${CNN_DIR}"

/home/m10957667/.conda/envs/mlenv/bin/python3 ${SCRATCH}/ML_LBM_3D/paper_plots_3d.py \
    --dataset_dir ${DATASET} \
    --cnn_dir     ${CNN_DIR} \
    --subset      test \
    --start_id    901 \
    --end_id      1000 \
    --sigma       0.5 \
    --output_dir  ${OUT_DIR}

echo "=== ANALYSIS DONE: $(date) ==="
echo "Plots: $(ls ${OUT_DIR}/*.png 2>/dev/null | wc -l) files"

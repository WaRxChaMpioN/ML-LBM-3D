#!/bin/bash
#SBATCH --job-name=geom_gen_64
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/geom_gen_%j.out
#SBATCH --error=logs/geom_gen_%j.err

# Geometry generation on Wendian HPC (Colorado School of Mines)
# Generates 1000 correlated porous geometries at 64^3 resolution
# Implements Cirpka & Attinger (2003) via Liu & Mostaghimi (2017)

module load python/3.11
source ~/venv/ml-lbm/bin/activate

mkdir -p logs dataset_64/{train,val,test}

echo "Starting geometry generation: $(date)"

python3 generate_geometry_3d.py \
    --dims 3 \
    --n_total 1000 \
    --output_dir ./dataset_64 \
    --size 64 \
    --phi_min 0.3 \
    --phi_max 0.7 \
    --corr_length 0.1 \
    --train_frac 0.8 \
    --val_frac 0.1 \
    --seed 42

echo "Geometry generation complete: $(date)"
echo "Output: ./dataset_64"

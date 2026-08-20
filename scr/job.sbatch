#!/bin/bash
#SBATCH -A berzelius-2026-95
#SBATCH -p berzelius
#SBATCH --gpus=1
#SBATCH -t 10:00:00
#SBATCH -J p-5-13mm
#SBATCH --mail-type=ALL
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j.err

cd /proj/multipress/users/x_sebja/Exp_32_Jul_cfd/scr || exit 1

echo "=============================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "PWD: $(pwd)"
echo "=============================="

apptainer exec --nv /proj/multipress/users/x_sebja/torch_2.2.1-gpu.sif \
python -u main_pressure_early_stopping.py

echo "=============================="
echo "End: $(date)"
echo "=============================="
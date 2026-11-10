#!/bin/bash
#SBATCH --partition=gpu-ampere
#SBATCH -n 1
#SBATCH --nodes=1
#SBATCH -G 1
#SBATCH --gres=gpu:1
#SBATCH -t 96:00:00
#SBATCH --job-name=DRP_phi_rmse
#SBATCH --output=bash_logs/DRP_phi_rmse.out
#SBATCH --error=bash_logs/DRP_phi_rmse.err

mkdir -p bash_logs
python3 plot_phi_rmse.py
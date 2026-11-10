#!/bin/bash
###--------SBATCH --partition=gpu-tesla
###SBATCH -n 10
####SBATCH --nodes=1
#SBATCH --mem=MaxMemPerNode
###-------SBATCH -G 1
###--------SBATCH --gres=gpu:1
#SBATCH -t 96:00:00
#SBATCH --job-name=fft_400
#SBATCH --output=out_fft_solver

#------modules ----------------
conda init bash
. "$HOME"/.bashrc
conda activate fft_solver
module purge

#------------------------------\

echo "STARTING JOB"

rock_list=(B2)
size_big_list=(900)
size_small_list=(75 90 100 150 180 225 300 900)
padding_list=(0)

for rock in "${rock_list[@]}"; do
    for size_big in "${size_big_list[@]}"; do
        for padding in "${padding_list[@]}"; do
            for size_small in "${size_small_list[@]}"; do

                srun --exclusive -n1 -N1 python -u fft_homogenization.py -r "${rock}" -S "${size_big}" -s "$size_small" -p "$padding" -g True 

            done
        done
    done
done

wait

echo "COMPLETED JOB"

#!/bin/sh
#SBATCH --partition=gpu-tesla
#SBATCH -n 10
#SBATCH --nodes=1
#SBATCH -G 1
#SBATCH --gres=gpu:1
#SBATCH -t 96:00:00
#SBATCH --job-name=DRPCNNHHM
#SBATCH --output=DRPCNNHHM.out
#SBATCH --error=DRPCNNHHM.err

# cp -r /path/to/shared_data/DigitalRock/cnn_hhm/models/model_1200 model
# for rock type `B1`
clear
# rm ._* figures/cnn_hhm*.png
# rm -rf figures/cnn_hhm_moduli*.png figures/fine_tune* data/fine_tune*
rm -rf model/fine_tune*
rm -rf data/cnn_hhm_moduli*
conda init
conda activate DRP

rock_types=("B1" "B2" "CG" "FB1" "FB2")
# rock_types=("FB2")
cube_sizes=(300 600)
modes=("" "--homogenization" "--hs_bound" "--hs_bound --benchmark_model")
# modes=("--hs_bound --benchmark_model")
# Standard evaluations
for rock in "${rock_types[@]}"; do
  for size in "${cube_sizes[@]}"; do
    for mode in "${modes[@]}"; do
      rm -rf data_hhm/
      python3 eval_cnn_hhm_moduli.py --largecube_size $size $mode --rock_type $rock
    done
  done
  # rm -rf data_hhm/ *.dat
  python3 eval_cnn_hhm_training.py --rock_type $rock
done

# echo "Now start fine-tuning the model for different rock types..."

# Fine-tuning evaluations (NOT needed)
# for rock in "B2" "CG" "FB1" "FB2"; do
#   for size in "${cube_sizes[@]}"; do
#     for mode in "${modes[@]}"; do
#       rm -rf data_hhm/
#       extra_args="--fine_tune"
#       [[ $rock == "FB2" ]] && extra_args="$extra_args --epochs 75"
#       python3 eval_cnn_hhm_moduli.py --largecube_size $size $mode --rock_type $rock $extra_args
#     done
#   done
#   rm -rf data_hhm/ *.dat
#   extra_args="--fine_tune"
#   [[ $rock == "FB2" ]] && extra_args="$extra_args --epochs 75"
#   python3 eval_cnn_hhm_training.py --rock_type $rock $extra_args
# done
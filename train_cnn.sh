#!/bin/bash
#SBATCH --partition=gpu-ampere
#SBATCH -n 10
#SBATCH --nodes=1
#SBATCH -G 1
#SBATCH --gres=gpu:1
#SBATCH -t 96:00:00
#SBATCH --job-name=DRP_cnn_elas_moduli
#SBATCH --output=DRPCNNElasMod_all.out
#SBATCH --error=DRPCNNElasMod_all.err

# Pass script args (e.g. --no_train) to every train_cnn call
SCRIPT_ARGS=("$@")

train_cnn() { # Function to train CNN _benchmark
    local extra_args=("$@")
    echo "————————————————————————————————————————————————————————————"
    echo " Running train_cnn() with flags: ${extra_args[*]:-none}"
    echo "————————————————————————————————————————————————————————————"
    start_time=$(date +%s)

    python3 -i train_cnn.py \
        --scan_dir ../../../machine_learning/size_75/data/data_B1/skin_0/images/ \
        --dat_file ../../../machine_learning/size_75/data/data_B1/skin_0/stiffness.dat \
        --epochs 100 \
        --batch_size 8 \
        --scan_shape 75 75 75 \
        --lr 1e-3 \
        --train_ratio 0.8 \
        --stop_option test \
        --early_stop \
        "${extra_args[@]}"

    end_time=$(date +%s)
    elapsed=$((end_time - start_time))
    echo "Training finished in ${elapsed}s"
    echo
}

echo "(1) - Direct stiffness tensor prediction"
train_cnn "${SCRIPT_ARGS[@]}"

echo "(2) - Predicting homogenized bulk & shear moduli"
train_cnn --homogenization "${SCRIPT_ARGS[@]}"

echo "(3) - HS-bound normalized parameters"
train_cnn --hs_bound "${SCRIPT_ARGS[@]}"

echo "(4) - HS-bound normalized parameters using Benchmark 1"
train_cnn --hs_bound --benchmark_model "${SCRIPT_ARGS[@]}"
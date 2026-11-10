#!/bin/bash
#SBATCH --partition=gpu-ampere
#SBATCH -n 10
#SBATCH --nodes=1
#SBATCH -G 1
#SBATCH --gres=gpu:1
#SBATCH -t 96:00:00
#SBATCH --job-name=DRP_cnn_elas_moduli
#SBATCH --output=DRPCNNDiffRocks.out
#SBATCH --error=DRPCNNDiffRocks.err

train_cnn() {
    local rock_type="$1"; shift
    local extra_args=("$@")
    echo "────────────────────────────────────────────────────────────────"
    echo " Rock type: ${rock_type}    |    Flags: ${extra_args[*]:-none}"
    echo "────────────────────────────────────────────────────────────────"
    local start_time=$(date +%s)

    python3 -i train_cnn.py \
        --scan_dir ../../../machine_learning/size_75/data/data_${rock_type}/skin_0/images/ \
        --dat_file ../../../machine_learning/size_75/data/data_${rock_type}/skin_0/stiffness.dat \
        --figures_dir "fig_${rock_type}/" \
        --rock_type "${rock_type}" \
        --epochs 100 \
        --batch_size 8 \
        --scan_shape 75 75 75 \
        --lr 1e-3 \
        --train_ratio 0.8 \
        --stop_option test \
        --early_stop \
        "${extra_args[@]}"

    local end_time=$(date +%s)
    echo "-> Training for ${rock_type} (${extra_args[*]:-stiffness}) finished in $((end_time - start_time))s"
    echo
}

rock_types=(B1 B2 FB1 FB2 CG)
for rock in "${rock_types[@]}"; do
    echo
    echo "##############################"
    echo "##  Starting experiments for ${rock}  ##"
    echo "##############################"
    echo
    # 1) direct stiffness‐tensor prediction
    echo "(1) Direct stiffness tensor prediction"
    train_cnn "${rock}"
    # 2) homogenized bulk & shear
    echo "(2) Predicting homogenized bulk & shear moduli"
    train_cnn "${rock}" --homogenization
    # 3) HS-bound normalized parameters
    echo "(3) HS-bound normalized parameters"
    train_cnn "${rock}" --hs_bound
    # 4) HS-bound + Benchmark1’s model
    echo "(4) HS-bound + Benchmark1’s model"
    train_cnn "${rock}" --hs_bound --benchmark_model
done
import argparse, os, torch, random, numpy as np
seed = 2025
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from util_machine_learning import MLDRP, resolve_checkpoint_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a 3D CNN to predict elastic properties from digital rock scans")
    parser.add_argument("--scan_dir", type=str, required=True)
    parser.add_argument("--dat_file", type=str, required=True)
    parser.add_argument("--key", type=str, default=None)
    parser.add_argument("--figures_dir", type=str, default="figures")
    parser.add_argument("--model_dir", type=str, default="model")
    parser.add_argument("--model_name", type=str, default="cnn3d_rock")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--scan_shape", type=int, nargs=3, default=[100, 100, 100])
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--stop_option", type=str, default='train')
    parser.add_argument("--homogenization", action="store_true", help="Predict effective moduli (2 outputs) from stiffness (36 outputs)")
    parser.add_argument("--hs_bound", action="store_true", help="Predict HS fraction (2 outputs) from stiffness (36 outputs)")
    parser.add_argument("-loss_avg_step", "--loss_avg_step", type=int, default=10)
    parser.add_argument("--early_stop", action="store_true", default=True, help="Enable early stopping")
    # Additional rock parameters (required if hs_bound is used)
    parser.add_argument("--rock_type", type=str, default="B1", choices=["B1", "B2", "CG", "FB1", "FB2"], help="Rock type")
    parser.add_argument("--E1", type=float, help="Young's modulus for pore", default=2.903225806452e-04)
    parser.add_argument("--nu1", type=float, help="Poisson ratio for pore", default=0.4516129032258)
    parser.add_argument("--E2", type=float, help="Young's modulus for mineral", default=95.29411764705883)
    parser.add_argument("--nu2", type=float, help="Poisson ratio for mineral", default=0.058823529411764705)
    parser.add_argument("--benchmark_model", action="store_true", help="Use Benchmark 1 (prior) model for homogenization")
    parser.add_argument("--no_train", action="store_true", help="Skip training; load existing model and only evaluate/plot")
    args = parser.parse_args()

    # Prefer Drive-downloaded size75/<rock>/ when present.
    local_scan = os.path.join('size75', args.rock_type, 'images')
    local_dat = os.path.join('size75', args.rock_type, 'stiffness.dat')
    if os.path.isdir(local_scan) and os.path.isfile(local_dat):
        args.scan_dir = local_scan
        args.dat_file = local_dat

    args.figures_dir = os.path.abspath(args.figures_dir)
    os.makedirs(args.figures_dir, exist_ok=True)
    ml_drp = MLDRP(args)
    ml_drp.load_data()

    model_path = os.path.join(args.model_dir, f"{ml_drp.model_name}_{ml_drp.key}_{args.rock_type}.pt")
    if args.no_train:
        if not os.path.isfile(model_path):
            raise SystemExit(f"--no_train: model not found at {model_path}; run without --no_train to train first.")
        print(f"Loading existing model from {model_path} (eval/plot only)")
        model_path = resolve_checkpoint_path(model_path)
        ckpt = torch.load(model_path, map_location=ml_drp.device)
        ml_drp.model.load_state_dict(ckpt)

    ml_drp.prepare_train_test()
    if not args.no_train:
        ml_drp.train_model()
    ml_drp.eval_model()
    ml_drp.plot_pred()
    ml_drp.deploy_cnn_stiffness_pred(output_file='cnn_pred')

    if not args.no_train:
        np.save(os.path.join(args.figures_dir, f"loss_{ml_drp.key}.npy"),
                np.array([ml_drp.train_loss_list, ml_drp.test_loss_list]))
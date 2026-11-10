import matplotlib.pyplot as plt
import argparse, torch, random, numpy as np
import shutil, glob, os
seed = 2025
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from util_machine_learning import MLDRP
from util_visualization import TerminalColors as colors

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a 3D CNN to predict elastic properties from digital rock scans")
    parser.add_argument("--scan_dir", type=str, default="../../../machine_learning/size_75/data/data_B1/skin_0/images/")
    parser.add_argument("--dat_file", type=str, default="../../../machine_learning/size_75/data/data_B1/skin_0/stiffness.dat")
    parser.add_argument("--key", type=str, default=None)
    parser.add_argument("--figures_dir", type=str, default="test_figures")
    parser.add_argument("--model_dir", type=str, default="model")
    parser.add_argument("--model_name", type=str, default="cnn3d")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--scan_shape", type=int, nargs=3, default=[75, 75, 75])
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--stop_option", type=str, default='train')
    parser.add_argument("--homogenization", action="store_true", help="Predict effective moduli (2 outputs) from stiffness (36 outputs)")
    parser.add_argument("--hs_bound", action="store_true", help="Predict HS fraction (2 outputs) from stiffness (36 outputs)")
    parser.add_argument("-loss_avg_step", "--loss_avg_step", type=int, default=10)
    parser.add_argument("--early_stop", action="store_true", default=True, help="Enable early stopping")
    # Additional rock parameters (required if hs_bound is used)
    parser.add_argument("--E1", type=float, help="Young's modulus for pore", default=2.903225806452e-04)
    parser.add_argument("--nu1", type=float, help="Poisson ratio for pore", default=0.4516129032258)
    parser.add_argument("--E2", type=float, help="Young's modulus for mineral", default=95.29411764705883)
    parser.add_argument("--nu2", type=float, help="Poisson ratio for mineral", default=0.058823529411764705)
    parser.add_argument("--benchmark_model", action="store_true", help="Use Benchmark 1 (prior) model for homogenization")
    args = parser.parse_args()

    ml_drp = MLDRP(args)
    ml_drp.load_data()
    ml_drp.prepare_train_test()
    ml_drp.train_model()
    ml_drp.eval_model()
    ml_drp.plot_pred()

    # ================= Begin Test Case =================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sd = ml_drp.model.state_dict()
    # to_save = {'fc2.weight': sd['fc2.weight'].cpu().numpy(),
    #            'fc3.weight': sd['fc3.weight'].cpu().numpy(),}
    # np.savez_compressed(f'ref_data/fc2_3_weights_{device}.npz', **to_save)

    ref = np.load(f"ref_data/fc2_3_weights_{device}.npz")
    ref_fc2 = ref["fc2.weight"]
    ref_fc3 = ref["fc3.weight"]

    cur_fc2 = sd["fc2.weight"].cpu().numpy()
    cur_fc3 = sd["fc3.weight"].cpu().numpy()

    match_fc2 = np.allclose(cur_fc2, ref_fc2, rtol=1e-6, atol=1e-8)
    match_fc3 = np.allclose(cur_fc3, ref_fc3, rtol=1e-6, atol=1e-8)
    if match_fc2 and match_fc3:
        print(f"test case for CNN elastic moduli prediction {colors.GREEN}PASSED{colors.RESET}")
    else:
        print(f"test case for CNN elastic moduli prediction {colors.RED}FAILED{colors.RESET}")
        print(f"Differences in fc2 & fc3 weights: {diff2} & {diff3}")
    # ================= End Test Case =================
    [shutil.rmtree(d) for d in glob.glob("test_*") if os.path.isdir(d)]
import os, re, torch, tqdm, numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from util_visualization import TerminalColors, plotting_params, figures_machine_learning
from util_elasticity import compute_moduli_atomman
from util_data_processing import load_labels, load_scans_and_labels, RockDataset
plt.rcParams.update(plotting_params)


def stiffness_symmetrize(C):
    """
    Impose symmetry on stiffness tensor(s): C_ij = C_ji.
    C: shape (N, 36) or (N, 6, 6). Returns same shape.
    """
    C = np.asarray(C)
    was_flat = C.shape[-1] == 36
    if was_flat:
        C = C.reshape(-1, 6, 6)
    out = 0.5 * (C + np.swapaxes(C, -2, -1))
    return out.reshape(-1, 36) if was_flat else out


def stiffness_upper_triangle_21(C):
    """
    Extract the 21 independent components (upper triangle) from 6x6 stiffness.
    C: shape (N, 6, 6). Returns (N, 21) in Voigt order:
    C11,C12,C13,C14,C15,C16, C22,C23,C24,C25,C26, C33,C34,C35,C36, C44,C45,C46, C55,C56, C66.
    """
    C = np.asarray(C)
    if C.ndim == 2:
        C = C[np.newaxis, ...]
    n = C.shape[0]
    tri = []
    for i in range(6):
        for j in range(i, 6):
            tri.append(C[:, i, j])
    return np.stack(tri, axis=1)  # (N, 21)

# ====================== CNN Elastic Moduli Predictions =========================

def resolve_checkpoint_path(path: str) -> str:
    """Return path if the checkpoint file exists; otherwise return path unchanged."""
    import os
    return path


class CNN3D(torch.nn.Module):
    def __init__(self, rock_size=100, mode="stiffness", E1=None, nu1=None, E2=None, nu2=None, f1=None):
        """
        mode: one of "stiffness", "moduli", or "hs_bound".
          - "stiffness": output raw 36 numbers (flattened 6×6 tensor);
          - "moduli": reshape to 6×6, then compute effective moduli using compute_moduli;
          - "hs_bound": predict a 6×6 stiffness tensor, compute effective moduli, normalize them,
                        and then interpolate between HS lower and upper bounds.
        When mode=="hs_bound", the parameters E1, nu1, E2, nu2 (rock properties) and f1 (porosity)
        must be provided.
        """
        super(CNN3D, self).__init__()
        self.mode = mode
        # Save rock parameters (only used in hs_bound mode).
        self.E1 = E1; self.nu1 = nu1; self.E2 = E2; self.nu2 = nu2; self.f1 = f1
        self.input_shape = (rock_size, rock_size, rock_size)
        # Convolutional layers.
        self.conv1 = torch.nn.Conv3d(1, 8, kernel_size=3, padding=1)
        self.pool1 = torch.nn.MaxPool3d(2)
        self.conv2 = torch.nn.Conv3d(8, 32, kernel_size=3, padding=1)
        self.pool2 = torch.nn.MaxPool3d(2)
        self.conv3 = torch.nn.Conv3d(32, 128, kernel_size=3, padding=1)
        self.pool3 = torch.nn.MaxPool3d(2)
        # Determine flattened dimension using a dummy input.
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, *self.input_shape)
            dummy_output = self.pool3(
                self.conv3(
                    self.pool2(
                        self.conv2(
                            self.pool1(
                                self.conv1(dummy_input)
                            )
                        )
                    )
                )
            )
            flattened_size = dummy_output.numel() // dummy_output.shape[0]
        self.fc1 = torch.nn.Linear(flattened_size, 256)
        # For both moduli and hs_bound mode, we need to predict a 6x6 stiffness tensor
        # so fc2 should output 36 numbers.
        self.fc2 = torch.nn.Linear(256, 36)
        self.fc3 = torch.nn.Linear(256, 2)  # For moduli mode, output 2 numbers (K, G)

    def compute_moduli_torch(self, C, avg_scheme='Hill'): 
    # !!! archived function: NOT used --- can be used for debugging purposes to impose moduli in CNN output
    # Note: Hill average implemented in CNN so autograd keeps the graph (atomman/numpy would break it)
    # because when training the model, torch needs to backprop through the stiffness tensor (differentiable)
    # and using atomman (or even convert the matrices just to numpy) will break the graph.
    # Check: https://discuss.pytorch.org/t/backpropagating-loss-through-model-after-purposefully-breaking-graph/99560
        """
        Compute the Hill-averaged bulk (K) and shear (G) moduli from a 6×6 stiffness matrix C
        using differentiable PyTorch operations.
        C is expected to have shape (batch, 6, 6).
        """
        C_sym = 0.5 * (C + C.transpose(-2, -1))
        C11 = C_sym[..., 0, 0]
        C22 = C_sym[..., 1, 1]
        C33 = C_sym[..., 2, 2]
        C12 = C_sym[..., 0, 1]
        C13 = C_sym[..., 0, 2]
        C23 = C_sym[..., 1, 2]
        K_V = (C11 + C22 + C33 + 2*(C12+C13+C23)) / 9.0
        s = torch.inverse(C_sym)
        s11 = s[..., 0, 0]
        s22 = s[..., 1, 1]
        s33 = s[..., 2, 2]
        s12 = s[..., 0, 1]
        s13 = s[..., 0, 2]
        s23 = s[..., 1, 2]
        K_R = 1.0 / (s11 + s22 + s33 + 2*(s12+s13+s23))
        K = 0.5 * (K_V + K_R)
        C44 = C_sym[..., 3, 3]
        C55 = C_sym[..., 4, 4]
        C66 = C_sym[..., 5, 5]
        G_V = (C11+C22+C33 - C12 - C13 - C23 + 3*(C44+C55+C66)) / 15.0
        s44 = s[..., 3, 3]
        s55 = s[..., 4, 4]
        s66 = s[..., 5, 5]
        G_R = 15.0 / (4*(s11+s22+s33) - 4*(s12+s13+s23) + 3*(s44+s55+s66))
        G = 0.5 * (G_V + G_R)
        return K, G

    def forward(self, x, f1=None):
        # Run through the convolutional backbone and fc1.
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.relu(self.conv3(x))
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))

        if self.mode == "stiffness":
            return self.fc2(x)
        elif self.mode == "moduli":
            stiffness = self.fc2(x).view(-1, 6, 6)
            # K, G = self.compute_moduli_torch(stiffness) # now we don't need this (but can be recalled for debugging)
            K, G = self.fc3(x).split(1, dim=1)
            K, G = K.squeeze(1), G.squeeze(1)  # (B,)
            return torch.stack([K, G], dim=1)
        elif self.mode == "hs_bound":
            assert f1 is not None, "hs_bound mode requires per-sample porosity tensor f1"
            norm_K, norm_G = self.fc3(x).split(1, dim=1)  # (B,1)
            # 4) compute HS bounds *per sample* using f1
            f2 = 1 - f1 # shape (batch,)
            # E1, nu1, E2, nu2 = self.E1, self.nu1, self.E2, self.nu2
            self.k1  = self.E1 / (3*(1 - 2*self.nu1))
            self.k2  = self.E2 / (3*(1 - 2*self.nu2))
            self.mu1 = self.E1 / (2*(1 + self.nu1))
            self.mu2 = self.E2 / (2*(1 + self.nu2))
            # broadcast f1,f2 into same shape as norm_K
            f1b = f1.unsqueeze(1)
            f2b = f2.unsqueeze(1)

            lowerK = self.k1 + f2b/((1/(self.k2-self.k1)) + (f1b/(self.k1+4*self.mu1/3)))
            upperK = self.k2 + f1b/((1/(self.k1-self.k2)) + (f2b/(self.k2+4*self.mu2/3)))

            lowerG = self.mu1 + f2b/((1/(self.mu2-self.mu1)) + \
                                     (f1b/((5*self.mu1*(self.k1+4*self.mu1/3))/(2*(self.k1+2*self.mu1)))))
            upperG = self.mu2 + f1b/((1/(self.mu1-self.mu2)) + \
                                     (f2b/((5*self.mu2*(self.k2+4*self.mu2/3))/(2*(self.k2+2*self.mu2)))))

            # 5) interpolate back into [lower,upper]
            Kfinal = lowerK + norm_K*(upperK - lowerK + 1e-8)
            Gfinal = lowerG + norm_G*(upperG - lowerG + 1e-8)
            return torch.cat([Kfinal, Gfinal], dim=1)
        else: raise ValueError("Invalid mode selected for CNN3D.")


class CNN3D_BENCHMARK(torch.nn.Module):
    # def __init__(self, dropout_prob=0.5, input_shape=(75,75,75)):
    def __init__(self, dropout_prob=0.5, rock_size=100, mode="hs_bound", E1=None, nu1=None, E2=None, nu2=None, f1=None):
        super(CNN3D_BENCHMARK, self).__init__()
        self.mode = mode
        if self.mode != 'hs_bound':
            raise ValueError("CNN3D_BENCHMARK only supports mode='hs_bound'")

        # initialize elasticity parameters
        self.E1, self.nu1 = E1, nu1
        self.E2, self.nu2 = E2, nu2
        self.input_shape = (rock_size, rock_size, rock_size)
        # build conv backbone exactly as before…
        self.conv = torch.nn.Sequential(
            torch.nn.BatchNorm3d(1),
            torch.nn.Conv3d(1, 64, 3, padding=1), torch.nn.PReLU(),
            torch.nn.Dropout3d(dropout_prob),  torch.nn.BatchNorm3d(64),

            torch.nn.Conv3d(64, 64, 5, padding=2), torch.nn.PReLU(),
            torch.nn.AvgPool3d(2),                torch.nn.BatchNorm3d(64),

            torch.nn.Conv3d(64, 32, 3, padding=1), torch.nn.PReLU(),
            torch.nn.AvgPool3d(2),                torch.nn.BatchNorm3d(32),

            torch.nn.Conv3d(32, 32, 3, padding=1), torch.nn.PReLU(),
            torch.nn.Dropout3d(dropout_prob),     torch.nn.BatchNorm3d(32),

            torch.nn.Conv3d(32, 16, 3, padding=1), torch.nn.PReLU(),
            torch.nn.AvgPool3d(2),                torch.nn.BatchNorm3d(16),

            torch.nn.Conv3d(16,  8, 3, padding=1), torch.nn.PReLU(),
            torch.nn.Dropout3d(dropout_prob),     torch.nn.BatchNorm3d(8),)

        # determine flatten size automatically
        with torch.no_grad():
            dummy = torch.rand(1, 1, *self.input_shape)
            out   = self.conv(dummy)
            self._flatten_size = out.numel() // out.shape[0]

        # now build FC head using Benchmark 1 framework
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(self._flatten_size, 64),
            torch.nn.PReLU(),
            torch.nn.BatchNorm1d(64),
            torch.nn.Dropout(dropout_prob),

            torch.nn.Linear(64, 32),
            torch.nn.PReLU(),
            torch.nn.BatchNorm1d(32),

            torch.nn.Linear(32, 2),
            torch.nn.Tanh(),)

    def forward(self, x, f1=None):
        x = self.conv(x)
        x = x.flatten(start_dim=1)
        # x = self.fc(x) -> I don't like this, modify the HS bound prediction adjust to my framework

        # ====== Evaluations for K and G ======
        if self.mode == "hs_bound":
            assert f1 is not None, "hs_bound mode requires per-sample porosity tensor f1"
            norm_K, norm_G = self.fc(x).split(1, dim=1)  # (B,1)
            # self.fc3(x).split(1, dim=1)  # (B,1)
            # 4) compute HS bounds *per sample* using f1
            f2 = 1 - f1                 # shape (batch,)
            # E1, nu1, E2, nu2 = self.E1, self.nu1, self.E2, self.nu2
            self.k1  = self.E1 / (3*(1 - 2*self.nu1))
            self.k2  = self.E2 / (3*(1 - 2*self.nu2))
            self.mu1 = self.E1 / (2*(1 + self.nu1))
            self.mu2 = self.E2 / (2*(1 + self.nu2))
            # broadcast f1,f2 into same shape as norm_K
            f1b = f1.unsqueeze(1)
            f2b = f2.unsqueeze(1)

            lowerK = self.k1 + f2b/((1/(self.k2-self.k1)) + (f1b/(self.k1+4*self.mu1/3)))
            upperK = self.k2 + f1b/((1/(self.k1-self.k2)) + (f2b/(self.k2+4*self.mu2/3)))

            lowerG = self.mu1 + f2b/((1/(self.mu2-self.mu1)) + \
                                     (f1b/((5*self.mu1*(self.k1+4*self.mu1/3))/(2*(self.k1+2*self.mu1)))))
            upperG = self.mu2 + f1b/((1/(self.mu1-self.mu2)) + \
                                     (f2b/((5*self.mu2*(self.k2+4*self.mu2/3))/(2*(self.k2+2*self.mu2)))))

            # 5) interpolate back into [lower,upper]
            Kfinal = lowerK + norm_K*(upperK - lowerK + 1e-8)
            Gfinal = lowerG + norm_G*(upperG - lowerG + 1e-8)
            return torch.cat([Kfinal, Gfinal], dim=1)
            # return 0.5 * (x + 1.0) -> prior benchmark previous way (and I don't like it)
        else: raise ValueError("Invalid mode selected for CNN3D_BENCHMARK.")


# -------------------------- MLDRP Class --------------------------
class MLDRP:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.colors            = TerminalColors
        self.scan_dir          = getattr(args, 'scan_dir', None)
        self.dat_file          = getattr(args, 'dat_file', None)
        self.key               = getattr(args, 'key', None)
        self.scan_shape        = tuple(args.scan_shape)
        self.lr                = getattr(args, 'lr', 1e-3)
        self.epochs            = getattr(args, 'epochs', 1000)
        self.batch_size        = getattr(args, 'batch_size', 8)
        self.train_ratio       = getattr(args, 'train_ratio', 0.8)
        self.early_stop        = getattr(args, 'early_stop', False)
        self.loss_avg_step     = getattr(args, 'loss_avg_step', 10)
        self.model             = None
        self.optimizer         = None
        self.criterion         = torch.nn.MSELoss()
        self.full_dataset      = None
        self.train_dataset     = None
        self.test_dataset      = None
        self.train_loader      = None
        self.test_loader       = None
        self.train_loss_list   = []
        self.test_loss_list    = []
        self.train_preds       = None
        self.train_truth       = None
        self.test_preds        = None
        self.test_truth        = None
        self.pred_moduli       = None
        self.truth_moduli      = None
        self.figs_ml           = figures_machine_learning(fig_dir=self.args.figures_dir)
        self.model_dir         = getattr(args, 'model_dir', "model")
        self.model_name        = getattr(args, 'model_name', "cnn3d_rock")
        self.benchmark_model      = getattr(args, 'benchmark_model', False)
        self.rock_type         = getattr(args, 'rock_type', "B1")

    def load_data(self):
        X, Y, file_list, porosity_array = load_scans_and_labels(self.scan_dir, self.dat_file, self.scan_shape)
        print(f'* {self.colors.BLUE}(Overall) Input data: {X.shape}, Output data: {Y.shape}{self.colors.RESET}')
        if X.shape[0] == 0:
            raise ValueError(f"{self.colors.RED}No samples loaded. Check your scan directory and .dat file.{self.colors.RESET}")

        # If HS-bound mode is activated, precompute HS fractions using the porosity values from the stiffness file.
        if self.args.hs_bound:
            Y_hs = []
            for i in range(Y.shape[0]):
                stiffness = Y[i].reshape(6,6)
                bulk, shear = compute_moduli_atomman(stiffness)  # effective moduli from DNS
                Y_hs.append([bulk, shear])
            Y = np.array(Y_hs)
            print(f"Converted stiffness labels to normalized HS fractions. New label shape: {Y.shape}")

        elif self.args.homogenization:
            Y_moduli = []
            for i in range(Y.shape[0]):
                stiffness = Y[i].reshape(6,6)
                bulk, shear = compute_moduli_atomman(stiffness)
                Y_moduli.append([bulk, shear])
            Y = np.array(Y_moduli)
            print(f"Converted stiffness labels to moduli. New label shape: {Y.shape}")

        self.full_dataset = RockDataset(X, porosity_array, Y)

        target_dim = Y.shape[1]
        if target_dim == 36:
            print("Using CNN3D for full stiffness tensor prediction (36 outputs).")
            self.model = CNN3D(rock_size=self.scan_shape[0], mode="stiffness").to(self.device)
        elif target_dim == 2:
            if self.args.hs_bound:
                print("Using CNN3D for HS bound prediction (2 outputs).")
                if self.benchmark_model:
                    self.model = CNN3D_BENCHMARK(dropout_prob=0.5,
                                        rock_size=self.scan_shape[0], mode="hs_bound",
                                        E1=self.args.E1, nu1=self.args.nu1,
                                        E2=self.args.E2, nu2=self.args.nu2,
                                        f1=None).to(self.device)
                else:
                    self.model = CNN3D(rock_size=self.scan_shape[0], mode="hs_bound",
                                        E1=self.args.E1, nu1=self.args.nu1,
                                        E2=self.args.E2, nu2=self.args.nu2,
                                        f1=None).to(self.device)
            elif self.args.homogenization:
                print("Using CNN3D for moduli prediction (2 outputs).")
                self.model = CNN3D(rock_size=self.scan_shape[0], mode="moduli").to(self.device)
            else:
                raise ValueError("Unsupported target dimension: 2 without homogenization/hs_bound flag.")
        else:
            raise ValueError(f"Unsupported target dimension: {target_dim}")
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # after setting the model's mode, we can set the key
        if self.key is None:
            self.key = 'stiffness' if self.model.mode == None else self.model.mode
        if self.benchmark_model:
            self.key = 'benchmark_hs_bound'

    def init_model_from_args(self):
        """Instantiate CNN from flags only (no dataset I/O). For timing/deploy without load_data()."""
        if self.args.hs_bound:
            print("Using CNN3D for HS bound prediction (2 outputs).")
            if self.benchmark_model:
                self.model = CNN3D_BENCHMARK(
                    dropout_prob=0.5, rock_size=self.scan_shape[0], mode="hs_bound",
                    E1=self.args.E1, nu1=self.args.nu1, E2=self.args.E2, nu2=self.args.nu2, f1=None).to(self.device)
            else:
                self.model = CNN3D(
                    rock_size=self.scan_shape[0], mode="hs_bound",
                    E1=self.args.E1, nu1=self.args.nu1, E2=self.args.E2, nu2=self.args.nu2, f1=None).to(self.device)
        elif self.args.homogenization:
            print("Using CNN3D for moduli prediction (2 outputs).")
            self.model = CNN3D(rock_size=self.scan_shape[0], mode="moduli").to(self.device)
        else:
            print("Using CNN3D for full stiffness tensor prediction (36 outputs).")
            self.model = CNN3D(rock_size=self.scan_shape[0], mode="stiffness").to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        if self.key is None:
            self.key = 'stiffness' if self.model.mode is None else self.model.mode
        if self.benchmark_model:
            self.key = 'benchmark_hs_bound'

    def prepare_train_test(self):
        train_size = int(self.train_ratio * len(self.full_dataset))
        test_size = len(self.full_dataset) - train_size
        self.train_dataset, self.test_dataset = random_split(self.full_dataset, [train_size, test_size])
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader  = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
        print(f"Dataset split: {train_size} training samples, {test_size} test samples.")

    def train_model(self, save_model=True, plot_loss=True): # one can chose not to save the model
        self.model.train()
        for epoch in tqdm.tqdm(range(self.epochs), desc="Training Progress"):
            running_train_loss = 0.0
            for scans, porosities, labels in self.train_loader:
                scans      = scans.to(self.device)
                porosities = porosities.to(self.device)
                labels     = labels.to(self.device)
                self.optimizer.zero_grad()
                outputs    = self.model(scans, porosities) 
                loss       = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_train_loss += loss.item()
            avg_train_loss = running_train_loss / len(self.train_loader)
            self.train_loss_list.append(avg_train_loss)
            self.model.eval()
            running_test_loss = 0.0
            with torch.no_grad():
                for scans, porosities, labels in self.test_loader:
                    scans      = scans.to(self.device)
                    porosities = porosities.to(self.device)
                    labels     = labels.to(self.device)
                    outputs    = self.model(scans, porosities) 
                    loss       = self.criterion(outputs, labels)
                    running_test_loss += loss.item()
            avg_test_loss = running_test_loss / len(self.test_loader)
            self.test_loss_list.append(avg_test_loss)
            self.model.train()
            if (epoch + 1) % 10 == 0:
                tqdm.tqdm.write(f"Epoch [{epoch+1}/{self.epochs}], Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}")
            loss_list = self.test_loss_list if self.args.stop_option == 'test' else self.train_loss_list
            if self.early_stop and len(loss_list) >= self.loss_avg_step * 2:
                prev_avg = np.mean(loss_list[-(self.loss_avg_step * 2):-self.loss_avg_step])
                curr_avg = np.mean(loss_list[-self.loss_avg_step:])
                if curr_avg > prev_avg:
                    print(f"Early stopping triggered: {self.loss_avg_step}-step average loss increased.")
                    break
        os.makedirs(self.model_dir, exist_ok=True)
        if save_model:
            torch.save(self.model.state_dict(), f"{self.model_dir}/{self.model_name}_{self.key}_{self.rock_type}.pt")
            print(f"---> model saved to {self.model_dir}/{self.model_name}_{self.key}.pt")
        if plot_loss:
            self.figs_ml.plot_loss_curve(np.array(self.train_loss_list), np.array(self.test_loss_list), fig_key=self.key)

    def eval_model(self):
        self.model.eval()
        train_preds_list, train_truth_list = [], []
        with torch.no_grad():
            for scans, porosities, labels in self.train_loader:
                scans      = scans.to(self.device)
                porosities = porosities.to(self.device)
                outputs    = self.model(scans, porosities)

                if self.args.hs_bound:
                    # compute per-sample HS lower/upper
                    f1  = porosities                  # (B,)
                    f2  = 1 - f1
                    k1  = self.args.E1 / (3*(1 - 2*self.args.nu1))
                    k2  = self.args.E2 / (3*(1 - 2*self.args.nu2))
                    mu1 = self.args.E1 / (2*(1 + self.args.nu1))
                    mu2 = self.args.E2 / (2*(1 + self.args.nu2))

                    f1b = f1.unsqueeze(1)            # (B,1)
                    f2b = f2.unsqueeze(1)

                    lowerK = k1  + f2b/((1/(k2-k1)) + (f1b/(k1 + 4*mu1/3)))
                    upperK = k2  + f1b/((1/(k1-k2)) + (f2b/(k2 + 4*mu2/3)))
                    lowerG = mu1 + f2b/((1/(mu2-mu1)) + (f1b/((5*mu1*(k1+4*mu1/3))/(2*(k1+2*mu1)))))
                    upperG = mu2 + f1b/((1/(mu1-mu2)) + (f2b/((5*mu2*(k2+4*mu2/3))/(2*(k2+2*mu2)))))

                    # now clamp
                    outputs[:,0:1] = torch.max(torch.min(outputs[:,0:1], upperK), lowerK)
                    outputs[:,1:2] = torch.max(torch.min(outputs[:,1:2], upperG), lowerG)

                train_preds_list.append(outputs.cpu().numpy())
                train_truth_list.append(labels.cpu().numpy())

        self.train_preds = np.concatenate(train_preds_list, axis=0)
        self.train_truth = np.concatenate(train_truth_list, axis=0)
        test_preds_list, test_truth_list = [], []
        with torch.no_grad():
            for scans, porosities, labels in self.test_loader:
                scans      = scans.to(self.device)
                porosities = porosities.to(self.device)
                outputs    = self.model(scans, porosities)

                if self.args.hs_bound:
                    # compute per-sample HS lower/upper
                    f1  = porosities                  # (B,)
                    f2  = 1 - f1
                    k1  = self.args.E1 / (3*(1 - 2*self.args.nu1))
                    k2  = self.args.E2 / (3*(1 - 2*self.args.nu2))
                    mu1 = self.args.E1 / (2*(1 + self.args.nu1))
                    mu2 = self.args.E2 / (2*(1 + self.args.nu2))

                    f1b = f1.unsqueeze(1)            # (B,1)
                    f2b = f2.unsqueeze(1)

                    lowerK = k1  + f2b/((1/(k2-k1)) + (f1b/(k1 + 4*mu1/3)))
                    upperK = k2  + f1b/((1/(k1-k2)) + (f2b/(k2 + 4*mu2/3)))
                    lowerG = mu1 + f2b/((1/(mu2-mu1)) + (f1b/((5*mu1*(k1+4*mu1/3))/(2*(k1+2*mu1)))))
                    upperG = mu2 + f1b/((1/(mu1-mu2)) + (f2b/((5*mu2*(k2+4*mu2/3))/(2*(k2+2*mu2)))))

                    # now clamp - HS bound predictions
                    outputs[:,0:1] = torch.max(torch.min(outputs[:,0:1], upperK), lowerK)
                    outputs[:,1:2] = torch.max(torch.min(outputs[:,1:2], upperG), lowerG)

                test_preds_list.append(outputs.cpu().numpy())
                test_truth_list.append(labels.cpu().numpy())

        self.test_preds = np.concatenate(test_preds_list, axis=0)
        self.test_truth = np.concatenate(test_truth_list, axis=0)
        all_preds = np.concatenate([self.train_preds, self.test_preds], axis=0)
        all_truth = np.concatenate([self.train_truth, self.test_truth], axis=0)
        r2_train  = r2_score(self.train_truth.flatten(), self.train_preds.flatten())
        r2_test   = r2_score(self.test_truth.flatten(), self.test_preds.flatten())
        print(f"Train R2: {r2_train:.3f}, Test R2: {r2_test:.3f}")

        cnn_results = [ self.train_truth.flatten(), self.train_preds.flatten(),
                        self.test_truth.flatten(), self.test_preds.flatten(),
                        r2_train, r2_test, 25, f"{self.args.figures_dir}/cnn_train_test_{self.key}.png" ] # 25 is the bins for the plot

        self.figs_ml.plot_pred_w_uncertainty(*cnn_results)

        if self.train_truth.shape[1] == 36:
            all_pred_stiff_tensor  = stiffness_symmetrize(all_preds).reshape(-1, 6, 6)  # impose symmetry
            all_truth_stiff_tensor = stiffness_symmetrize(all_truth.reshape(-1, 6, 6)).reshape(-1, 6, 6)  # FFT truth, symmetrize for consistency
            self.pred_moduli  = np.array([compute_moduli_atomman(C) for C in all_pred_stiff_tensor])
            self.truth_moduli = np.array([compute_moduli_atomman(C) for C in all_truth_stiff_tensor])
            r2_bulk  = r2_score(self.truth_moduli[:, 0], self.pred_moduli[:, 0])
            r2_shear = r2_score(self.truth_moduli[:, 1], self.pred_moduli[:, 1])
            print(f"R2 for Bulk modulus:  {r2_bulk:.3f}")
            print(f"R2 for Shear modulus: {r2_shear:.3f}")
            # Parity plot for 21 independent stiffness constants
            truth_21 = stiffness_upper_triangle_21(all_truth_stiff_tensor)
            pred_21  = stiffness_upper_triangle_21(all_pred_stiff_tensor)
            r2_21    = r2_score(truth_21.flatten(), pred_21.flatten())
            out_21   = f"{self.args.figures_dir}/cnn_train_test_{self.key}_21_parity.png"
            self.figs_ml.plot_stiffness_21_parity(truth_21.flatten(), pred_21.flatten(), r2_21, out_21)
            print(f"R2 (21 stiffness constants): {r2_21:.3f}")
        else:
            self.pred_moduli  = all_preds
            self.truth_moduli = all_truth
            r2_bulk  = r2_score(self.truth_moduli[:, 0], self.pred_moduli[:, 0])
            r2_shear = r2_score(self.truth_moduli[:, 1], self.pred_moduli[:, 1])
            print(f"R2 for Bulk modulus:  {r2_bulk:.3f}")
            print(f"R2 for Shear modulus: {r2_shear:.3f}")

    def plot_pred(self, save_moduli=True):
        cur_dir = os.getcwd()
        self.plot_CNN_predictions(cur_dir, self.test_dataset, self.pred_moduli, self.truth_moduli,
                                  r2_score(self.truth_moduli[:, 0], self.pred_moduli[:, 0]),
                                  r2_score(self.truth_moduli[:, 1], self.pred_moduli[:, 1]),
                                  key=self.key, fig_dir=self.args.figures_dir)
        if save_moduli:
            np.save(f"{self.args.figures_dir}/cnn_pred_moduli_{self.key}.npy", self.pred_moduli)
            np.save(f"{self.args.figures_dir}/cnn_truth_moduli_{self.key}.npy", self.truth_moduli)

    @staticmethod
    def plot_CNN_predictions(cur_dir, test_dataset, pred_moduli, truth_moduli, r2_bulk, r2_shear, key=None, fig_dir=None):
        base = f'{cur_dir}/../../../machine_learning/size_75/data/data_B1/skin_0'
        try:
            bulk_HS   = np.loadtxt(f'{base}/bulk_mod_HS_bound.dat')
            shear_HS  = np.loadtxt(f'{base}/shear_mod_HS_bound.dat')
            dns_bulk  = np.loadtxt(f'{base}/bulk_mod.dat')
            dns_shear = np.loadtxt(f'{base}/shear_mod.dat')
            has_hs = True
        except Exception:
            has_hs = False

        test_indices       = test_dataset.indices
        n_test             = len(test_indices)
        if has_hs:
            test_dns_bulk  = dns_bulk[test_indices]
            test_dns_shear = dns_shear[test_indices]
            test_bulk_HS   = bulk_HS[test_indices]
            test_shear_HS  = shear_HS[test_indices]
        else:
            test_dns_bulk  = np.arange(n_test, dtype=float)
            test_dns_shear = np.arange(n_test, dtype=float)
            test_bulk_HS   = np.zeros((n_test, 2))
            test_shear_HS  = np.zeros((n_test, 2))
        test_truth_moduli  = truth_moduli[test_indices]
        test_pred_moduli   = pred_moduli[test_indices]
        ind_sort_bulk      = np.argsort(test_dns_bulk)
        sorted_dns_bulk    = test_dns_bulk[ind_sort_bulk]
        sorted_pred_bulk   = test_pred_moduli[ind_sort_bulk, 0]
        sorted_bulk_lower  = test_bulk_HS[ind_sort_bulk, 0]
        sorted_bulk_upper  = test_bulk_HS[ind_sort_bulk, 1]
        ind_sort_shear     = np.argsort(test_dns_shear)
        sorted_dns_shear   = test_dns_shear[ind_sort_shear]
        sorted_pred_shear  = test_pred_moduli[ind_sort_shear, 1]
        sorted_shear_lower = test_shear_HS[ind_sort_shear, 0]
        sorted_shear_upper = test_shear_HS[ind_sort_shear, 1]

        fig, ax = plt.subplots(1, 2, figsize=(9, 4.75))
        ax[0].scatter(test_truth_moduli[:, 0], test_pred_moduli[:, 0], alpha=0.5, color='darkblue', label=rf'$R^2={r2_bulk:.3f}$')
        ax[0].plot([0, 40], [0, 40], '--', color='C1')
        if has_hs:
            ax[0].plot(sorted_dns_bulk, sorted_bulk_lower, color='red', linestyle='-', label='HS bounds')
            ax[0].plot(sorted_dns_bulk, sorted_bulk_upper, color='red', linestyle='-')
        ax[0].set_xlabel(r"$K_{eff}^{\rm DNS}$ [GPa]")
        ax[0].set_ylabel(r"$K_{eff}^{\rm CNN}$ [GPa]")
        ax[0].legend(loc='upper left')
        ax[1].scatter(test_truth_moduli[:, 1], test_pred_moduli[:, 1], alpha=0.5, color='darkblue', label=rf'$R^2={r2_shear:.3f}$')
        ax[1].plot([0, 50], [0, 50], '--', color='C1')
        if has_hs:
            ax[1].plot(sorted_dns_shear, sorted_shear_lower, color='red', linestyle='-', label='HS bounds')
            ax[1].plot(sorted_dns_shear, sorted_shear_upper, color='red', linestyle='-')
        ax[1].set_xlabel(r"$G_{eff}^{\rm DNS}$ [GPa]")
        ax[1].set_ylabel(r"$G_{eff}^{\rm CNN}$ [GPa]")
        ax[1].legend(loc='upper left')

        plt.tight_layout()
        out_dir = fig_dir if os.path.isabs(fig_dir) else os.path.join(cur_dir, fig_dir)
        os.makedirs(out_dir, exist_ok=True)
        out_png = os.path.join(out_dir, f"cnn_moduli_scatter_{key}.png")
        plt.savefig(out_png, dpi=300)
        print(f"---> save plot as {out_png}")
        plt.close()
        np.savez(out_png.replace('.png', '.npz'),
                bulk_truth=test_truth_moduli[:, 0], bulk_pred=test_pred_moduli[:, 0],
                shear_truth=test_truth_moduli[:, 1], shear_pred=test_pred_moduli[:, 1],
                bulk_hs_lower_x=sorted_dns_bulk, bulk_hs_lower_y=sorted_bulk_lower,
                bulk_hs_upper_x=sorted_dns_bulk, bulk_hs_upper_y=sorted_bulk_upper,
                shear_hs_lower_x=sorted_dns_shear, shear_hs_lower_y=sorted_shear_lower,
                shear_hs_upper_x=sorted_dns_shear, shear_hs_upper_y=sorted_shear_upper)

    def deploy_cnn_stiffness_pred(self, output_file: str):
        from util_data_processing import write_stiffness_dat
        """
        If self.model.mode == 'stiffness', runs the trained CNN over all rocks
        and writes out a .dat file of predicted 6×6 stiffness tensors in the
        same format as the original stiffness.dat.
        """

        X, Y, file_list, porosity_array = load_scans_and_labels(
            self.scan_dir, self.dat_file, self.scan_shape)
        ds = RockDataset(X, porosity_array, Y)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=False)

        self.model.eval()

        if self.model.mode == "stiffness":
            output_file = f'{output_file}_stiffness.dat'
            preds = []
            with torch.no_grad():
                for scans, porosities, _ in tqdm.tqdm(dl, desc="Inferring on Rock Scans"):
                    scans = scans.to(self.device)
                    out   = self.model(scans)                   # (B,36)
                    preds.append(out.cpu().numpy())
            all_preds = np.concatenate(preds, axis=0)         # (N,36)
            all_preds = stiffness_symmetrize(all_preds)        # impose C_ij = C_ji (21 independent constants)
            poro_tuples = [(float(p), float(p)) for p in porosity_array]
            write_stiffness_dat(all_preds, output_file,
                filenames = file_list, porosities = poro_tuples)
            print(f"---> CNN‐predicted stiffness tensors saved to {output_file}")

        elif self.model.mode == "moduli":
            output_file = f'{output_file}_moduli.dat'
            mods = []
            with torch.no_grad():
                for scans, porosities, _ in tqdm.tqdm(dl, desc="Inferring on Rock Scans"):
                    scans = scans.to(self.device)
                    out   = self.model(scans)        # (B,2) in [GPa]
                    mods.append(out.cpu().numpy())
            all_mods = np.vstack(mods)            # (N,2)

            stiff_preds = [] # build isotropic stiffness tensors
            for K, G in all_mods:
                lam  = K - 2.0*G/3.0
                C11  = lam + 2.0*G
                C12  = lam
                C44  = G
                C = np.array([  [C11, C12, C12,   0,   0,   0],
                                [C12, C11, C12,   0,   0,   0],
                                [C12, C12, C11,   0,   0,   0],
                                [  0,   0,   0, C44,   0,   0],
                                [  0,   0,   0,   0, C44,   0],
                                [  0,   0,   0,   0,   0, C44], ], dtype=float)
                stiff_preds.append(C.flatten())

            stiff_preds = np.vstack(stiff_preds)  # (N,36)
            poro_tups = [(float(p), float(p)) for p in porosity_array]
            write_stiffness_dat( stiff_preds, output_file,
                filenames  = file_list, porosities = poro_tups )
            print(f"---> Isotropic‐homogenized stiffness tensors saved to {output_file}")

        elif self.model.mode == "hs_bound":
                # choose output names
            if self.benchmark_model:
                mod_file   = f"{output_file}_benchmark_hs_bound_moduli.dat"
                stiff_file = f"{output_file}_benchmark_hs_bound_stiffness.dat"
            else:
                mod_file   = f"{output_file}_hs_bound_moduli.dat"
                stiff_file = f"{output_file}_hs_bound_stiffness.dat"

            # 1) run inference to get (K_final, G_final)
            hs_mods = []
            with torch.no_grad():
                for scans, porosities, _ in tqdm.tqdm(dl, desc="Inferring HS‑bound"):
                    scans = scans.to(self.device)
                    out   = self.model(scans, porosities.to(self.device))  # (B,2)
                    hs_mods.append(out.cpu().numpy())
            all_mods = np.vstack(hs_mods)  # (N,2)

            # 2) save raw HS‑bound moduli
            np.savetxt(mod_file, all_mods, fmt="%.6e")
            print(f"---> HS‑bound moduli saved to {mod_file}")

            # 3) rebuild isotropic stiffness from K_final, G_final
            stiff_preds = []
            for Kf, Gf in all_mods:
                lam  = Kf - 2.0*Gf/3.0
                C11  = lam + 2.0*Gf
                C12  = lam
                C44  = Gf
                C = np.array([
                    [C11, C12, C12,   0,   0,   0],
                    [C12, C11, C12,   0,   0,   0],
                    [C12, C12, C11,   0,   0,   0],
                    [  0,    0,   0, C44,   0,   0],
                    [  0,    0,   0,   0, C44,   0],
                    [  0,    0,   0,   0,   0, C44]
                ], dtype=float)
                stiff_preds.append(C.flatten())
            all_stiffs = np.vstack(stiff_preds)  # (N,36)

            # 4) write .dat exactly like your other branches
            from util_data_processing import write_stiffness_dat
            poro_tups = [(float(p), float(p)) for p in porosity_array]
            write_stiffness_dat(all_stiffs, stiff_file,
                                filenames  = file_list, porosities = poro_tups)

            print(f"---> HS‑bound stiffness tensors saved to {stiff_file}")
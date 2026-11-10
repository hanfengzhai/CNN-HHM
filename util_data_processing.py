import os, re, torch, numpy as np
from torch.utils.data import Dataset

def load_labels(dat_file):
    """
    Reads a stiffness.dat file with header lines of the form:
      # file = Rock_75_1.raw, porosity including skin =  0.30162,  porosity excluding skin =  0.30162 
    followed by 6 lines of stiffness tensor data.
    
    Returns a dictionary mapping filename --> stiffness tensor (as a numpy array)
    and a dictionary mapping filename --> porosity (taken here as the "porosity including skin").
    """
    labels = {}
    porosity_dict = {}
    with open(dat_file, 'r') as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("# file"):
            # Extract the filename.
            m = re.search(r"#\s*file\s*=\s*(\S+)", line)
            if not m:
                i += 1
                continue
            filename = m.group(1).strip().strip(",")
            # Extract porosity from the header.
            m_porosity = re.search(r"porosity including skin\s*=\s*([\d.eE+-]+)", line)
            if m_porosity:
                porosity_val = float(m_porosity.group(1))
            else:
                porosity_val = None
            # Read next 6 nonempty lines as the tensor.
            tensor_lines = []
            j = i + 1
            while j < len(lines) and len(tensor_lines) < 6:
                current_line = lines[j].strip()
                if current_line:
                    tensor_lines.append(current_line)
                j += 1
            if len(tensor_lines) == 6:
                tensor = []
                for tl in tensor_lines:
                    row_vals = list(map(float, tl.split()))
                    tensor.append(row_vals)
                labels[filename] = np.array(tensor)
                porosity_dict[filename] = porosity_val
            else:
                print(f"Warning: Expected 6 lines for stiffness tensor for file {filename}, got {len(tensor_lines)}")
            i = j
        else:
            i += 1
    return labels, porosity_dict


def load_scans_and_labels(scan_dir, dat_file, scan_shape=(100,100,100)):
    """
    Loads raw scans from scan_dir and their associated stiffness tensors from dat_file.
    Also returns porosity values extracted from the header lines.
    """
    labels, porosity_dict = load_labels(dat_file)
    X_list, Y_list, porosity_list, file_list = [], [], [], []
    raw_files = sorted([fname for fname in os.listdir(scan_dir) if fname.endswith(".raw")])
    print("Number of digital rocks", len(raw_files))
    for fname in raw_files:
        if fname not in labels:
            print(f"Warning: No label found for file {fname}")
            continue
        full_path = os.path.join(scan_dir, fname)
        try:
            scan_data = np.fromfile(full_path, dtype=np.int8)
        except Exception as e:
            print(f"Error reading file {full_path}: {e}")
            continue
        if scan_data.size != np.prod(scan_shape):
            print(f"Warning: File {fname} size {scan_data.size} does not match expected shape {scan_shape}")
            continue
        scan_data = scan_data.reshape(scan_shape)
        # Add channel dimension for PyTorch: (1, D, H, W)
        scan_data = scan_data[np.newaxis, ...]
        X_list.append(scan_data)
        Y_list.append(labels[fname].flatten())
        porosity_list.append(porosity_dict[fname])
        file_list.append(fname)
    X = np.array(X_list)
    Y = np.array(Y_list)
    porosity_array = np.array(porosity_list)
    return X, Y, file_list, porosity_array


class RockDataset(Dataset):
    def __init__(self, scans, porosities, labels):
        self.scans      = scans
        self.porosities = porosities   # shape (N,)
        self.labels     = labels

    def __len__(self):
        return len(self.scans)

    def __getitem__(self, idx):
        scan      = torch.tensor(self.scans[idx],      dtype=torch.float32)
        label     = torch.tensor(self.labels[idx],     dtype=torch.float32)
        if self.porosities is not None:
            poro = torch.tensor(self.porosities[idx], dtype=torch.float32)
            return scan, poro, label
        else:
            return scan, label


def augment_volume(volume): 
    # !!! archived function: not used for current training
    """
    Given a 3D volume (shape: (D, H, W)), return a list of 12 rotated versions.
    For the x-axis, we rotate the (H, W) plane; for y-axis, rotate (D, W); for z-axis, rotate (D, H).
    """
    augmented = []
    for k in range(4): # Rotate around x-axis: axes=(1,2) (preserves D)
        augmented.append(np.rot90(volume, k=k, axes=(1,2)))
    for k in range(4): # Rotate around y-axis: axes=(0,2) (preserves H)
        augmented.append(np.rot90(volume, k=k, axes=(0,2)))
    for k in range(4): # Rotate around z-axis: axes=(0,1) (preserves W)
        augmented.append(np.rot90(volume, k=k, axes=(0,1)))
    return augmented


def write_stiffness_dat(pred_stiffness, out_file, filenames=None, porosities=None): 
    # !!! archived function: can be used for future evaluation of CNN prediction quality on HHM moduli calculation
    """
    pred_stiffness : numpy array of shape (N, 36), containing the predicted 6x6 stiffness for N rocks
    out_file       : name of the .dat file to write
    filenames      : optional list of length N, each entry is the raw file name (e.g. "Rock_100_1.raw")
    porosities     : optional list of length N, each entry is a tuple (por_incl, por_excl) 
                     e.g. (0.05906, 0.05906). If None, dummy values can be used.
    """
    N = pred_stiffness.shape[0]
    if filenames is None: # If no filenames given, create generic ones
        filenames = [f"Rock_800_{i+1}.raw" for i in range(N)]
    if porosities is None: # If no porosities given, use dummy zeros
        porosities = [(0.0, 0.0)] * N

    with open(out_file, "w") as f:
        for i in range(N):
            # Reshape the i-th row into 6x6
            stiff_6x6 = pred_stiffness[i].reshape(6, 6)

            # Write the header line, e.g.:
            # -> # file = Rock_100_1.raw, porosity including skin =  0.05906,  porosity excluding skin =  0.05906 
            f.write(f"# file = {filenames[i]}, porosity including skin =  {porosities[i][0]:.5f},  porosity excluding skin =  {porosities[i][1]:.5f} \n\n")

            for row in stiff_6x6: # Write the 6 lines of stiffness data
                # Each value is printed with e.g. 4 decimals; adjust formatting to suit
                f.write(" ".join(f"{val:12.4f}" for val in row) + "\n")
            f.write("\n\n")
from pathlib import Path
from fft_homogenization import compute_homogenized_stiffness

def compute_moduli_from_y_nu(Y: float, nu: float):
    """
        Compute bulk (K) and shear (mu) moduli from Young's modulus Y and Poisson ratio nu.
    """
    K = Y / (3 * (1 - 2 * nu))
    mu = Y / (2 * (1 + nu))
    return K, mu

def gassmann_bulk(K_dry: float, K_s: float, phi: float, K_fluid: float):# -> float:
    """
        Predict saturated bulk modulus using Gassmann's equation.
    """
    num = (1 - K_dry / K_s) ** 2
    den = phi / K_fluid + (1 - phi) / K_s - K_dry / (K_s ** 2) # 
    return K_dry + num / den

def process_homogenization(vox_dir: Path, method: str, mat_type: str,
                           avg_scheme: str, size_big: int, size_smalls: list):
    """
        Run FFT-based homogenization for specified sizes and return computed moduli lists.
    """
    results = {
        'size_smalls': size_smalls,
        'K_dry': [], 'G_dry': [],
        'K_sat': [], 'G_sat': []}

    for size_small in size_smalls:
        print(f"processing rocks of subcube size {size_small} (big cube {size_big})")
        bulk_mods = []
        shear_mods = []
        for state in ['dry', 'sat']:
            vox_fn = vox_dir / f"stiffness_{method}_{size_big}_{size_small}_{state}.dat"
            out_fn = Path(f"stiffness_{method}_{size_big}_{size_small}_{mat_type}_{state}.dat")
            out_fn.unlink(missing_ok=True)
            bulk, shear = compute_homogenized_stiffness(
                vox_fn, out_fn,
                mat_type=mat_type, avg_scheme=avg_scheme,
                return_moduli=True)
            bulk_mods.append(bulk)
            shear_mods.append(shear)
        # store
        results['K_dry'].append(bulk_mods[0])
        results['G_dry'].append(shear_mods[0])
        results['K_sat'].append(bulk_mods[1])
        results['G_sat'].append(shear_mods[1])
    return results

def compute_moduli_atomman(C, avg_scheme='Hill'): # Hill, Voigt, Reuss
    import atomman as am
    """
        Compute the Hill-averaged bulk and shear modulus from a 6×6 stiffness matrix C using the atomman package.
    """
    C_sym = 0.5 * (C + C.T) # Ensure the stiffness matrix is symmetric.
    
    elastic_constants = am.ElasticConstants(Cij=C_sym)
    bulk = elastic_constants.bulk(avg_scheme)
    shear = elastic_constants.shear(avg_scheme)
    return bulk, shear
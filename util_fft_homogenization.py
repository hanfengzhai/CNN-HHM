import matplotlib.pyplot as plt
from click import style
import scipy.sparse.linalg as sp
import os, re, itertools, numpy as np
from scipy.optimize import curve_fit
from tqdm import tqdm
from util_visualization import TerminalColors as colors

class FFT_GOOSE(object):
    """ This is an implementation of FFT based solver for elasticity.
        Here number of voxels must be even in all three directions.
        Ref: https://github.com/tdegeus/GooseFFT
        """

    def __init__(self, stiffness_voxel_file: str, print_info=False):
        """Initialization of class

        Args:
            stiffness_voxel_file (str): name of the file containing stiffness of voxels
        """
        super(FFT_GOOSE, self).__init__()
        self.ndim = 3
        self.stiffness_voxel_file = stiffness_voxel_file
        self.print_info = print_info

    def setup(self):
        """This function setup some variable and functions to be employed in future calculations
        """
        # read stiffness file and infer number of voxels
        self.stiffness_set = np.loadtxt(self.stiffness_voxel_file).reshape(-1, 6, 6)
        self.N = int(np.around(self.stiffness_set.shape[0] ** (1 / 3), 0))
        self.is_num_voxel_even = (self.N % 2) == 0
        self.ndof = self.ndim ** 2 * self.N ** 3  # number of degrees-of-freedom

        # define linear algebra operators
        self.trans2 = lambda A2: np.einsum("ijxyz          ->jixyz  ", A2)
        self.ddot42 = lambda A4, B2: np.einsum("ijklxyz,lkxyz  ->ijxyz  ", A4, B2)
        self.ddot44 = lambda A4, B4: np.einsum("ijklxyz,lkmnxyz->ijmnxyz", A4, B4)
        self.dot22 = lambda A2, B2: np.einsum("ijxyz  ,jkxyz  ->ikxyz  ", A2, B2)
        self.dot24 = lambda A2, B4: np.einsum("ijxyz  ,jkmnxyz->ikmnxyz", A2, B4)
        self.dot42 = lambda A4, B2: np.einsum("ijklxyz,lmxyz  ->ijkmxyz", A4, B2)
        self.dyad22 = lambda A2, B2: np.einsum("ijxyz  ,klxyz  ->ijklxyz", A2, B2)

        # identity tensor                                               [single tensor]
        self.i = np.eye(self.ndim)
        # identity tensors                                            [grid of tensors]
        self.I = np.einsum("ij,xyz", self.i, np.ones([self.N, self.N, self.N]))
        self.I4 = np.einsum(
            "ijkl,xyz->ijklxyz",
            np.einsum("il,jk", self.i, self.i),
            np.ones([self.N, self.N, self.N]),
        )
        self.I4rt = np.einsum(
            "ijkl,xyz->ijklxyz",
            np.einsum("ik,jl", self.i, self.i),
            np.ones([self.N, self.N, self.N]),
        )
        self.I4s = (self.I4 + self.I4rt) / 2.0
        self.II = self.dyad22(self.I, self.I)

        # frequency vector
        if self.is_num_voxel_even:
            self.freq = np.arange(-self.N / 2.0, self.N / 2.0)  # coordinate axis -> freq. axis
        else:
            self.freq = np.arange(-(self.N - 1) / 2.0, +(self.N + 1) / 2.0)  # coordinate axis -> freq. axis

        # (inverse) Fourier transform (for each tensor component in each direction)
        self.fft = lambda x: np.fft.fftshift(
            np.fft.fftn(np.fft.ifftshift(x), [self.N, self.N, self.N]))
        self.ifft = lambda x: np.fft.fftshift(
            np.fft.ifftn(np.fft.ifftshift(x), [self.N, self.N, self.N]))

        # functions for the projection 'G', and the product 'G : K : eps'
        self.G = lambda A2: np.real(
            self.ifft(self.ddot42(self.Ghat4, self.fft(A2)))).reshape(-1)
        self.K_deps = lambda depsm: self.ddot42(
            self.K4, depsm.reshape(self.ndim, self.ndim, self.N, self.N, self.N))
        self.G_K_deps = lambda depsm: self.G(self.K_deps(depsm))

    def construct_voxel_stiffness(self, mat_type: str, avg_scheme: str = "Hill"):
        import atomman as am
        """
        This function builds voxel-by-voxel stiffness matrix
        
        mat_type: Are voxels isotropic or anisotropic
        avg_scheme: If voxels are isotropic, how isotropic elastic constants are to be determined  from anisotropic stiffness matrix
        """
        if mat_type == "anisotropic":
            stiffness_aniso_list = np.zeros((self.N ** 3, 3, 3, 3, 3))

            for ii, stiffness in enumerate(self.stiffness_set):
                stiffness = 0.5 * (stiffness + stiffness.T)
                C = am.ElasticConstants(Cij=stiffness)
                stiffness_aniso_list[ii] = C.Cijkl

            stiffness_voxel = stiffness_aniso_list.reshape(
                (self.N, self.N, self.N, 3, 3, 3, 3))
            
            self.K4 = np.moveaxis(
                stiffness_voxel,
                source=[0, 1, 2, 3, 4, 5, 6],
                destination=[4, 5, 6, 0, 1, 2, 3],
            )

        if mat_type == "isotropic":
            bulk_mod_list = np.empty(self.N ** 3)
            shear_mod_list = np.empty_like(bulk_mod_list)
            for ii, stiffness in enumerate(self.stiffness_set):
                stiffness = 0.5 * (stiffness + stiffness.T)
                C = am.ElasticConstants(Cij=stiffness)
                bulk_mod_list[ii] = C.bulk(style=avg_scheme)
                shear_mod_list[ii] = C.shear(style=avg_scheme)

            bulk_mod_list = bulk_mod_list.reshape(self.N, self.N, self.N)
            shear_mod_list = shear_mod_list.reshape(self.N, self.N, self.N)
            self.K4 = bulk_mod_list * self.II + 2.0 * shear_mod_list * (
                self.I4s - 1.0 / 3.0 * self.II
            )

    def construct_projection_operator(self):
        """
        This function builds projection operator to obtain the compatible strain field.
        """

        delta = lambda i, j: float(i == j)  # Dirac delta function
        self.Ghat4 = np.zeros(
            [self.ndim, self.ndim, self.ndim, self.ndim, self.N, self.N, self.N]
        )  # zero initialize
        # - compute
        for i, j, l, m in itertools.product(range(self.ndim), repeat=4):
            for x, y, z in itertools.product(
                range(1, self.N), repeat=3
            ):  # no Nyquist frequency
                q = np.array(
                    [self.freq[x], self.freq[y], self.freq[z]]
                )  # frequency vector
                if not q.dot(q) == 0:  # zero freq. -> mean
                    self.Ghat4[i, j, l, m, x, y, z] = -(q[i] * q[j] * q[l] * q[m]) / (
                        q.dot(q)
                    ) ** 2 + (
                        delta(j, l) * q[i] * q[m]
                        + delta(j, m) * q[i] * q[l]
                        + delta(i, l) * q[j] * q[m]
                        + delta(i, m) * q[j] * q[l]
                    ) / (
                        2.0 * q.dot(q)
                    )

    def solve_elasticity(self, mat_type="anisotropic", avg_scheme="Hill"):

        """
        This function solves the elasticity problem and outputs average stiffness matrix.
        mat_type: is material isotropic or anisotropic
        avg_scheme: If material is isotropic, which scheme is to be used to calculate the isotropic elastic constants from anisotropic stiffness matrix ("Voigt", "Reuss", "Hill")
        """

        self.construct_voxel_stiffness(mat_type=mat_type, avg_scheme=avg_scheme)
        self.construct_projection_operator()

        load_dir = ["xx", "yy", "zz", "yz", "xz", "xy"]
        stiffmat = np.zeros((6, 6))

        for load_idx in tqdm(range(6), desc="Homogenization FFT_GOOSE Solver"):
            if self.print_info: tqdm.write(f"loading direaction = {load_dir[load_idx]}")

            # set macroscopic loading
            DE = np.zeros([self.ndim, self.ndim, self.N, self.N, self.N])

            if load_idx == 0:
                DE[0, 0] += 1.0
            elif load_idx == 1:
                DE[1, 1] += 1.0
            elif load_idx == 2:
                DE[2, 2] += 1.0
            elif load_idx == 3:
                DE[1, 2] += 0.5
                DE[2, 1] += 0.5
            elif load_idx == 4:
                DE[0, 2] += 0.5
                DE[2, 0] += 0.5
            elif load_idx == 5:
                DE[0, 1] += 0.5
                DE[1, 0] += 0.5

            # initialize stress and strain tensor                         [grid of tensors]
            sig = np.zeros([self.ndim, self.ndim, self.N, self.N, self.N])
            eps = np.zeros([self.ndim, self.ndim, self.N, self.N, self.N])

            # initial residual: distribute "DE" over grid using "K4"
            b = -self.G_K_deps(DE)
            eps += DE
            En = np.linalg.norm(eps)
            iiter = 0

            # iterate as long as the iterative update does not vanish
            while True:
                depsm, _ = sp.cg(
                    tol=1.0e-4,
                    A=sp.LinearOperator(
                        shape=(self.ndof, self.ndof),
                        matvec=self.G_K_deps,
                        dtype="float",
                    ),
                    b=b,
                    maxiter=100,
                )  # solve linear system using CG
                eps += depsm.reshape(
                    self.ndim, self.ndim, self.N, self.N, self.N
                )  # update DOFs (array -> tens.grid)
                sig = self.ddot42(self.K4, eps)  # new residual stress
                b = -self.G(sig)  # convert residual stress to residual
                if np.linalg.norm(depsm) / En < 1.0e-6 and iiter > 0:
                    if self.print_info:
                        tqdm.write( # only print if converged
                            f"step = {iiter}, error = {np.max(depsm)/En :10.2e}")  # print residual to the screen
                    break  # check convergence
                iiter += 1

            stiffmat[:, load_idx] = sig.mean(axis=(2, 3, 4))[ [0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1] ]
        print(f'FFT_GOOSE {colors.GREEN}converged in {iiter} iterations{colors.RESET}')
        return stiffmat


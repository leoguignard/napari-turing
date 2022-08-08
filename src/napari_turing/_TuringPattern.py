import numpy as np
from scipy.ndimage import convolve
from skimage.transform import resize
from skimage.color import rgb2gray


class TuringPattern:
    """docstring for TuringPattern"""

    def diffusion(self, arr, mu):
        to_cell = convolve(arr, self.kernel, mode="constant", cval=0)
        from_cell = self.nb_neighbs * arr
        out = mu * (to_cell - from_cell) / (self.dx * self.dy)
        return out

    def compute_turing(self, n=5):
        for _ in range(n):
            diff_A = self.diffusion(self.A, self.mu_a)
            self.A = self.A + self.dt * (
                diff_A + self.A - self.A**3 - self.I + self.k
            )
            diff_I = self.diffusion(self.I, self.mu_i)
            self.I = self.I + self.dt / self.tau * (diff_I + self.A - self.I)

            if self.boundaries in ["LR-Tube", "Infinite"]:
                self.A[:, -1] = self.A[:, 0]
                self.I[:, -1] = self.I[:, 0]
            if self.boundaries in ["TD-Tube", "Infinite"]:
                self.A[-1, :] = self.A[0, :]
                self.I[-1, :] = self.I[0, :]

    def __init__(
        self,
        mu_a=2.8e-4,
        mu_i=5e-3,
        tau=0.1,
        k=-0.005,
        dx=None,
        dy=None,
        dt=0.001,
        boundaries=False,
        A=None,
    ):
        self.mu_a = mu_a
        self.mu_i = mu_i
        self.tau = tau
        self.k = k
        size = 100
        self.size = size
        self.dt = dt
        if None in [dx, dy]:
            self.dx = self.dy = 2.0 / size
        else:
            self.dx = dx
            self.dy = dy

        if A is None:
            self.A = np.random.random((size, size))
            self.I = np.random.random((size, size))
        else:
            if len(A.shape) != 2 and not A.shape[-1] in (3, 4):
                print(f"Input images should be 2 dimensional ({A.shape=})")
                print(f"Using random distribution instead")
                self.A = np.random.random((size, size))
                self.I = np.random.random((size, size))
            else:
                if A.shape[-1] in (3, 4):
                    A = rgb2gray(A[..., :3])
                self.A = resize(A, (100, 100))
                max_A = np.percentile(self.A, 99)
                min_A = np.percentile(self.A, 1)
                self.A = 2 * (self.A - min_A) / (max_A - min_A)
                self.I = (self.A + np.random.random((size, size))) % 2
                self.A -= 1
                self.I -= 1
        self.init_A = self.A.copy()
        self.init_I = self.I.copy()
        self.boundaries = "Closed"
        self.kernel = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
        self.mask = np.ones_like(self.A)
        self.nb_neighbs = convolve(
            self.mask, self.kernel, mode="constant", cval=0
        )

from ._TuringPattern import TuringPattern, ModelParameter
from scipy.ndimage import convolve
import numpy as np
from typing import Optional

class Brusselator(TuringPattern):
    default_size = 200
    default_dx = 1
    default_dy = 1
    default_dt = 0.01
    default_contrast_limits=(0.3, 3.5)

    A = ModelParameter(
        name="A",
        value=1.,
        min=.1,
        max=5.,
        exponent=1,
        description="Concentration of productor of x",
    )
    B = ModelParameter(
        name="B",
        value=3.,
        min=.1,
        max=5.,
        exponent=1,
        description="Concentration of productor of y (combined with x)",
    )
    mu_x = ModelParameter(
        name="mu_x",
        value=2.,
        min=.1,
        max=5,
        exponent=1,
        description="Diffusion coefficient of x",
    )
    mu_y = ModelParameter(
        name="mu_y",
        description="Diffusion coefficient of y (10^-1)",
        value=2.,
        min=0.01,
        max=20,
        exponent=1e-1,
    )
    nb_pos = ModelParameter(
        name="nb_pos",
        value=1,
        min=1,
        max=300,
        exponent=1,
        description="Number of random perturbations",
        dtype=int
    )
    _necessary_parameters = [A, B, mu_x, mu_y, nb_pos]
    _tunable_parameters = [A, B, mu_x, mu_y, nb_pos]
    _concentration_names = ["X", "Y"]

    def _reaction(self, c: str) -> np.ndarray:
        if c == "X":
            return self.A + self.X**2 * self.Y - self.B * self.X - self.X
        elif c == "Y":
            return self.B * self.X - self.X**2 * self.Y

    def _diffusion(self, c: str) -> np.ndarray:
        if c == "X":
            arr = self.X
            mu = self.mu_x
        elif c == "Y":
            arr = self.Y
            mu = self.mu_y
        to_cell = convolve(arr, self.kernel.value, mode="constant", cval=0)
        from_cell = self.nb_neighbs * arr
        out = mu * (to_cell - from_cell) / (self.dx * self.dy)
        return out

    def init_concentrations(self, C: Optional[str] = None) -> None:
        pos = (np.random.random((2, self.nb_pos)) * self.size).astype(int)
        values = np.random.random(self.nb_pos)
        if C == "X" or C is None:
            X = np.ones((self.size, self.size))*self.A
            X[pos[0], pos[1]] += values
            self["X"] = X
        if C == "Y" or C is None:
            Y = np.ones((self.size, self.size))*self.B/self.A
            Y[pos[0], pos[1]] -= values
            self["Y"] = Y


    def __str__(self) -> str:
        return (
            "Equations (Brusselator model):\n"
            "  Concentration of two chemical components x and y\n"
            "    - dx/dt = mu_x * diffusion(x) + A + x^2*y - B * x - x\n"
            "    - dy/dt = mu_y * diffusion(i) + B * x - x^2*y"
        )

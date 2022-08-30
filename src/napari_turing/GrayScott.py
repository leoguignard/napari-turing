from ._TuringPattern import TuringPattern, ModelParameter
from scipy.ndimage import convolve
import numpy as np
from typing import Optional


class GrayScott(TuringPattern):
    default_size = 200
    default_dx = default_dy = 1
    default_dt = 1
    
    
    k = ModelParameter(
        name="k",
        description="First Parameter (k) (10^-2)",
        value=6.3,
        min=1,
        max=10,
        exponent=1e-2,
    )
    F = ModelParameter(
        name="F",
        description="Second Parameter (F) (10^-2)",
        value=3.,
        min=1,
        max=10,
        exponent=1e-2,
    )
    mu_x = ModelParameter(
        name="mu_x",
        description="Diffusion coefficient of x (10^-1)",
        value=2.,
        min=0.1,
        max=5,
        exponent=1e-1,
    )
    mu_y = ModelParameter(
        name="mu_y",
        description="Diffusion coefficient of y (10^-1)",
        value=1.,
        min=0.1,
        max=5,
        exponent=1e-1,
    )
    nb_pos = ModelParameter(
        name="nb_pos",
        value=5,
        min=1,
        max=300,
        exponent=1,
        description="Number of random perturbations",
        dtype=int
    )
    _necessary_parameters = [k, F, mu_x, mu_y, nb_pos]
    _tunable_parameters = _necessary_parameters
    _concentration_names = ["X", "Y"]


    def _reaction(self, c: str) -> np.ndarray:
        if c == "X":
            return -self.X * self.Y**2 + self.F * (1 - self.X)
        elif c == "Y":
            return self.X * self.Y**2 - (self.F + self.k) * self.Y

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
        if C == "X" or C is None:
            self["X"] = np.ones((self.size, self.size))
        if C == "Y" or C is None:
            Y = np.zeros((self.size, self.size))
            pos = (np.random.random((2, self.nb_pos)) * self.size).astype(int)
            Y[pos[0], pos[1]] = 1
            self["Y"] = Y

    def __str__(self) -> str:
        return (
            "Equations (Gray-Scott model):\n"
            "  Concentration of Activator (a) and Inhibitor (i)\n"
            "    - dx/dt = mu_x * diffusion(x) + -xy^2 + F(1 - x)\n"
            "    - dy/dt = mu_y * diffusion(i) + xy^2 - (F + k)y"
        )

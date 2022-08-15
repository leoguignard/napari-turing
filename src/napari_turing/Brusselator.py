from ._TuringPattern import TuringPattern, ModelParameter
from scipy.ndimage import convolve
import numpy as np
from typing import Optional

class Brusselator(TuringPattern):
    default_size = 200
    default_dx = 1
    default_dy = 1
    default_dt = 0.01

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
    _necessary_parameters = [A, B, mu_x, mu_y]
    _tunable_parameters = [A, B, mu_x, mu_y]
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
        if C == "X" or C is None:
            self["X"] = np.random.random(
                (self.size, self.size)
            )
        if C == "Y" or C is None:
            self["Y"] = np.random.random(
                (self.size, self.size)
            )

    def __str__(self) -> str:
        return (
            "Equations (Brusselator model):\n"
            "  Concentration of two chemical components x and y\n"
            "    - dx/dt = mu_x * diffusion(x) + A + x^2*y - B * x - x\n"
            "    - dy/dt = mu_y * diffusion(i) + B * x - x^2*y"
        )

from ._TuringPattern import TuringPattern, ModelParameter
from scipy.ndimage import convolve
import numpy as np
from typing import Optional


class FitzHughNagumo(TuringPattern):
    """docstring for FitzHughNagumoModel"""

    default_size = 100
    default_dx = default_dy = 2.0 / default_size
    default_dt = 0.001
    mu_a = ModelParameter(
        name="mu_a",
        value=2.8,
        min=1,
        max=5,
        exponent=1e-4,
        description="Activator diffusion coefficient (10^-4)",
    )
    mu_i = ModelParameter(
        name="mu_i",
        description="Inhibitor diffusion coefficient (10^-3)",
        value=5,
        min=2,
        max=7,
        exponent=1e-3,
    )
    tau = ModelParameter(
        name="tau",
        description="Reaction time ration between\nActivator and inhibitor",
        value=0.1,
        min=0.01,
        max=2,
        exponent=1,
    )
    k = ModelParameter(
        name="k",
        description="Is the activator a source (>0), a sink (<0)\nor neutral (0), (10^-3)",
        value=-5,
        min=-10,
        max=10,
        exponent=1e-3,
    )
    _necessary_parameters = [tau, k, mu_a, mu_i]
    _tunable_parameters = _necessary_parameters
    _concentration_names = ["A", "I"]


    def _reaction(self, c: str) -> np.ndarray:
        if c == "A":
            return self.A - self.A**3 - self.I + self.k
        elif c == "I":
            return (self.A - self.I) / self.tau

    def _diffusion(self, c: str) -> np.ndarray:
        if c == "A":
            arr = self.A
            mu = self.mu_a
        elif c == "I":
            arr = self.I
            mu = self.mu_i
        to_cell = convolve(arr, self.kernel.value, mode="constant", cval=0)
        from_cell = self.nb_neighbs * arr
        out = mu * (to_cell - from_cell) / (self.dx * self.dy)
        if c == "I":
            out /= self.tau
        return out

    def init_concentrations(self, C: Optional[str] = None) -> None:
        if C == "A" or C is None:
            self["A"] = np.random.random((self.size, self.size)) * 2 - 1
        if C == "I" or C is None:
            self["I"] = np.random.random((self.size, self.size)) * 2 - 1

    def __str__(self) -> str:
        return (
            "Equations (FitzHugh-Nagumo model):\n"
            "  Concentration of Activator (a) and Inhibitor (i)\n"
            "    - da/dt = mu_a * diffusion(a) + a - a^3 - i + k\n"
            "    - tau * di/dt = mu_i * diffusion(i) + a - i"
        )

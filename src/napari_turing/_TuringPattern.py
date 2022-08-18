from abc import abstractmethod
from typing import Optional, Union, Dict, List, Tuple, Set
import numpy as np
from scipy.ndimage import convolve
from skimage.transform import resize
from skimage.color import rgb2gray
from enum import Enum

class ModelParameter:
    def __init__(
        self,
        name: str = "",
        value: float=0,
        min: float=0,
        max: float=0,
        description: str = "",
        exponent: float = 1,
        dtype: type = float
    ) -> None:
        self.name = name
        self.min = min
        self.value = value
        self.max = max
        self.exponent = exponent
        self.description = description
        self.dtype = dtype


class DiffusionDirection(Enum):
    Isotrope = [
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ]
    Left = [
        [0, 1, 0],
        [2, 0, 0],
        [0, 1, 0],
    ]
    Right = [
        [0, 1, 0],
        [0, 0, 2],
        [0, 1, 0],
    ]
    Top = [
        [0, 2, 0],
        [1, 0, 1],
        [0, 0, 0],
    ]
    Bottom = [
        [0, 0, 0],
        [1, 0, 1],
        [0, 2, 0],
    ]


class Boundaries(Enum):
    Closed = "Closed"
    Left_Right_Tube = "LR-Tube"
    Top_Down_Tube = "TD-Tube"
    Inifinite = "Infinite"


class TuringPattern:
    """docstring for TuringPattern"""

    default_dt = 1
    default_dx = 1
    default_dy = 1
    default_size = 100
    default_color_map = 'viridis'
    default_interpolation = 'Spline36'
    _necessary_parameters = []
    _tunable_parameters = []
    _concentration_names = []
    default_contrast_limits=None

    increment = ModelParameter(
        name='Increment',
        value=100,
        min=10,
        max=1000,
        description="Number of steps per frame"
    )

    @abstractmethod
    def _reaction(self, c: str) -> np.ndarray:
        return self[c]

    @abstractmethod
    def _diffusion(self, c: str) -> np.ndarray:
        return self[c]

    def reaction(self) -> List[np.ndarray]:
        dC = []
        for c in self.concentrations:
            dC.append(self._reaction(c))
        return dC

    def diffusion(self) -> List[np.ndarray]:
        diffC = []
        for c in self.concentrations:
            diffC.append(self._diffusion(c))
        return diffC

    def compute_turing(self, n=5):
        for _ in range(n):
            reaction = self.reaction()
            diffusion = self.diffusion()
            for i, c in enumerate(self):
                self[c] = self[c] + self.dt * (reaction[i] + diffusion[i])
                if self.boundaries.value in ["LR-Tube", "Infinite"]:
                    tmp = self[c][:, -1].copy()
                    self[c][:, -1] = self[c][:, 0]
                    self[c][:, 0] = tmp
                    del tmp
                if self.boundaries.value in ["TD-Tube", "Infinite"]:
                    tmp = self[c][-1, :].copy()
                    self[c][-1, :] = self[c][0, :]
                    self[c][0, :] = tmp
                    del tmp

    @staticmethod
    def normalizing_input_image(A: np.ndarray, size: int):
        if len(A.shape) != 2 and not A.shape[-1] in (3, 4):
            print(f"Input images should be 2 dimensional ({A.shape=})")
            print(f"Using random distribution instead")
            A = np.random.random((size, size))
        else:
            if A.shape[-1] in (3, 4):
                A = rgb2gray(A[..., :3])
            A = resize(A, (size, size))
            max_A = np.percentile(A, 99)
            min_A = np.percentile(A, 1)
            if max_A != min_A:
                A = 2 * (A - min_A) / (max_A - min_A)
                A -= 1
        return A

    @abstractmethod
    def __str__(self) -> str:
        return "Equation:\n" "\tNot implemented yet"

    def _has_necessary_attr(self):
        missing_parameters = set([p.name for p in self._necessary_parameters]).difference(
            self.__dict__
        )
        if len(missing_parameters) != 0:
            raise Exception(
                f"Some parameters are missing:\n\t{missing_parameters}"
            )

    @abstractmethod
    def init_concentrations(self, C: Optional[str] = None) -> None:
        if C is None:
            for ci in self._concentration_names:
                self[ci] = np.random.random((self.size, self.size)) * 2 - 1
        else:
            self[C] = np.random.random((self.size, self.size)) * 2 - 1

    def reset(self):
        for c in self.concentrations:
            self[c] = self[f"init_{c}"].copy()

    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, item, value):
        self.__dict__[item] = value

    def __iter__(self):
        for c in self.concentrations:
            yield c

    def __init__(
        self,
        *,
        dx: Optional[float] = None,
        dy: Optional[float] = None,
        dt: Optional[float] = None,
        size: Optional[float] = None,
        kernel: Optional[DiffusionDirection] = None,
        boundaries: Optional[Boundaries] = None,
        concentrations: Union[
            Set[str], Tuple[str], List[str], Dict[str, np.ndarray]
        ] = ("A", "I"),
        seed: int=None,
        **kwargs,
    ):
        if seed is not None:
            np.random.seed(seed)
        self.__dict__.update(kwargs)
        if size is not None:
            self.size = size
        else:
            self.size = self.default_size
        if dt is not None:
            self.dt = dt
        else:
            self.dt = self.default_dt
        if None in [dx, dy]:
            self.dx = self.default_dx
            self.dy = self.default_dy
        else:
            self.dx = dx
            self.dy = dy
        if (
            isinstance(concentrations, set)
            or isinstance(concentrations, list)
            or isinstance(concentrations, tuple)
        ):
            concentrations_found = set()
            concentrations_double = []
            for c in concentrations:
                if c in concentrations_found:
                    concentrations_double.append(c)
                concentrations_found.add(c)
                self.init_concentrations(c)
            if 0 < len(concentrations_double):
                print(
                    (
                        "Careful, you gave at least twice the same name to a concentration\n"
                        f"\t{concentrations_double}"
                    )
                )
        elif isinstance(concentrations, dict):
            for name, C in concentrations.items():
                if C is not None:
                    self[name] = self.normalizing_input_image(C, self.size)
                else:
                    self.init_concentrations(name)
        else:
            self.init_concentrations()
        self.concentrations = list(concentrations)
        for c in self.concentrations:
            self.__dict__[f"init_{c}"] = self[c].copy()

        if not isinstance(boundaries, Boundaries):
            self.boundaries = Boundaries.Closed
        else:
            self.boundaries = boundaries
        if not isinstance(kernel, DiffusionDirection):
            self.kernel = DiffusionDirection.Isotrope
        else:
            self.kernel = kernel
        self.mask = np.ones((self.size, self.size), dtype=np.uint8)
        self.nb_neighbs = convolve(
            self.mask, self.kernel.value, mode="constant", cval=0
        )

        self._has_necessary_attr()

from enum import Enum
from .FitzHughNagumo import FitzHughNagumo
from .Brusselator import Brusselator
from .GrayScott import GrayScott

class AvailableModels(Enum):
    FitzHughNagumo = FitzHughNagumo
    Brusselator = Brusselator
    GrayScott = GrayScott
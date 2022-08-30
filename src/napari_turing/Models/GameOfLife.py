from ._TuringPattern import TuringPattern, ModelParameter
import numpy as np
from typing import Optional
from scipy.signal import convolve2d
from skimage.io import imread

# To create your own model you can use this template
# Some description is given bellow to help you with
# Creating your own model.
# Once you are happy with it, you can make it seen
# by napari_turing by updating the file `_model_list.py`

# This is an example based on the FitzHughNagumo model.
# There are 2 concentrations in competition:
#    - A: the activator
#    - I: the inhibitor
# The differential equations representing their evolutions are the following:
#    - da/dt = mu_a * diffusion(a) + a - a^3 - i + k (1)
#    - tau * di/dt = mu_i * diffusion(i) + a - i (2)
# There are the parameters that are common to any reaction-diffusion models:
# dt, dx, dy (the spatial parameters are not shown in the equations above)
# There are also parameters specific to the model:
# mu_a, k, tau, mu_i
# We will see bellow how all these parameters are handled.

class GameOfLife(TuringPattern):
    """Here is a template to create your own Model"""
    # default_board = (imread('https://conwaylife.com/w/images/4/49/Turingmachine_large.png') == 0).astype(int)
    default_board = np.zeros((300, 300), dtype=int)

    # Size of the initial grid (larger than 200 might create some latency)
    default_size = 100
    default_color_map = 'gray'
    default_interpolation = 'nearest'

    # Size of a pixel along the x direction
    default_dx = 1
    # Size of a pixel along the y direction
    default_dy = 1

    # Time resolution
    default_dt = 1

    # If your model ends up having `nan` numbers during its
    # run, you might want to decrease any or all
    # the value of dx, dy and or dt

    # Below is a list of all the parameters of the model.
    # They need to be listed here so they can be instantiated.
    # `value`: initial and default value
    # `min`, `max`: minimum and maximum values for the slider in napari
    # `exponent`: because the napari sliders do not have an infinite precision
    # it is sometimes important to give an exponent to the value so the user can
    # choose its parameter values more precisely
    # These are the parameters that are necessary to run the equations.
    _necessary_parameters = []
    # These are the parameters that can be modified via napari
    _tunable_parameters = _necessary_parameters
    # These are the name of the concentration tables
    _concentration_names = ["Board"]

    increment = ModelParameter(
        name='Increment',
        value=1,
        min=1,
        max=5,
        description="Number of steps per frame"
    )

    # The following allows to reset the values of the concentrations.
    # The function takes the name of the concentration to initialize.
    # If no name is given or if it is None all the concentrations are
    # reinitialized.
    #
    # The reason why this function is useful is that some models 
    # require specific initialisations for them to work correctly
    # In the following example the concentrations are reintinalized
    # to a random value between -1 and 1.
    # This is the default behavior, so if you don't need to change
    # it you don't have to implement the function.
    def init_concentrations(self, C: Optional[str] = None) -> None:
        im = np.array(
            [
                [0, 1, 0, 1, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 1, 0],
                [0, 0, 0, 1, 1, 1]
            ], dtype=int
        )
        shape_init = np.array(self.default_board.shape)
        shape_im = np.array(im.shape)
        start = shape_init//2 - shape_im//2
        end = start + shape_im
        if C is None:
            for ci in self.concentration_names():
                self[ci] = self.default_board
                self[ci][start[0]:end[0], start[1]:end[1]] = im
        else:
            self[C] = self.default_board
            self[C][start[0]:end[0], start[1]:end[1]] = im

    # This function allows to display some information about the model
    # in napari
    def __str__(self) -> str:
        return (
            "Game of Life rules:\n"
            "   - Any live cell with two or three\n"
            "     live neighbours survives.\n"
            "   - Any dead cell with three live\n"
            "     neighbours becomes a live cell.\n"
            "   - All other live cells die\n"
            "     in the next generation.\n"
        )

    # Declaring the reaction-diffusion equations
    # ------------------------------------------
    # The model ultimately will compute the reaction-diffusion step (roughly) as follow:
    # new_A = A + (_reaction('A') + _diffusion('A')) * dt
    # new_I = I + (_reaction('I') + _diffusion('I')) * dt
    # Note that the concentrations do not have to be labeled 'A' and 'I', the
    # model will deal with the concentration names by itself (as long as they are)
    # declared in the function `concentration_names`

    # This function defines the equations of the reactions.
    # It takes as an input which concentration to compute
    # (in this example we have to define how to compute A and I)
    def _reaction(self, c: str) -> np.ndarray:
        return -self.Board
    
    # This function defines the equations of the diffusion.
    # It takes as an input which concentration to compute
    # (in this example we have to define how to compute A and I)
    # Here we compute the diffusion as follow:
    # A cell gives an equal fraction mu of its concentration to its neighbors
    # A cell recieves an equal fraction mu of concentration from its neighbors
    # Neighbors = (left, right, above, below)
    # In the case of oriented diffusion the amount recieved and given to the neighbors
    # is imbalanced according to the position of the neighbor.
    def _diffusion(self, c: str) -> np.ndarray:
        kernel = np.array(
            [[1, 1, 1],
             [1, 0, 1],
             [1, 1, 1]]
        )
        nb_neighbs = convolve2d(self.Board, kernel, boundary='wrap', mode='same')
        # Checking the rules:
        #   Any live cell with two or three live neighbours survives.
        #   Any dead cell with three live neighbours becomes a live cell.
        #   All other live cells die in the next generation. Similarly, all other dead cells stay dead.
        new_live = np.zeros_like(nb_neighbs, dtype=int)
        new_live |= (nb_neighbs == 2) & self.Board
        new_live |= nb_neighbs == 3
        return new_live

from ._TuringPattern import TuringPattern, ModelParameter
import numpy as np
from typing import Optional
from scipy.ndimage import convolve

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

class ModelTemplate(TuringPattern):
    """Here is a template to create your own Model"""

    # Size of the initial grid (larger than 200 might create some latency)
    default_size = 100

    # Size of a pixel along the x direction
    default_dx = 1
    # Size of a pixel along the y direction
    default_dy = 1

    # Time resolution
    default_dt = 1

    # These are the name of the concentration tables
    _concentration_names = ["A", "I"]

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
    mu_a = ModelParameter(
        name="mu_a",  # Name of the parameter
        description="Activator diffusion coefficient (10^-4)",  # Description of the parameter for napari
        value=2.8,  # Initial and default value
        min=1,  # Minimum value the parameter can take
        max=5,  # Maximum value the parameter can take
        exponent=1e-4,  # All values given to this instance of the class will but multiplied by this value
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
    # These are the parameters that are necessary to run the equations.
    _necessary_parameters = [tau, k, mu_a, mu_i]
    # These are the parameters that can be modified via napari
    _tunable_parameters = _necessary_parameters

    # This function allows to display some information about the model
    # in napari
    def __str__(self) -> str:
        return (
            "Equations (FitzHugh-Nagumo model):\n"
            "  Concentration of Activator (a) and Inhibitor (i)\n"
            "    - da/dt = mu_a * diffusion(a) + a - a^3 - i + k\n"
            "    - tau * di/dt = mu_i * diffusion(i) + a - i"
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
        if C is None:
            for ci in self.concentration_names():
                self[ci] = np.random.random((self.size, self.size)) * 2 - 1
        else:
            self[C] = np.random.random((self.size, self.size)) * 2 - 1

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
        if c == "A":
            # Below is the reaction part of the equation (1)
            return self.A - self.A**3 - self.I + self.k 
        elif c == "I":
            # Below is the reaction part of the equation (2)
            return (self.A - self.I) / self.tau
    
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
        if c == "A":
            arr = self.A # Define the array of concentrations to diffuse for the reageant A
            mu = self.mu_a # Define the diffusion coefficient for the reageant A
        elif c == "I":
            arr = self.I # Define the array of concentrations to diffuse for the reageant I
            mu = self.mu_i # Define the diffusion coefficient for the reageant I
        
        # Computes what is recieved from neighboring cells
        from_cell = convolve(arr, self.kernel.value, mode="constant", cval=0)
        # Computes what is given to neighboring cells
        to_cell = self.nb_neighbs * arr

        # Computes the diffusion
        out = mu * (from_cell - to_cell) / (self.dx * self.dy)

        # In our case, the equation (2), for I specify that it has to be divided by tau
        if c == "I":
            out /= self.tau
        return out

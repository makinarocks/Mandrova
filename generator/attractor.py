"""
=======================
       Attractor
=======================

build 2019.01.31.17.00 (stable)

contributor:
  Jong Duk Shinn


"""
from generator.abstract_generator import AbstractGenerator

import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import odeint

class Attractor(AbstractGenerator):
    """
    The Attractor Class

    [What is this?]

    This class supports the following attractors:
      1. Lorenz
      2. Rössler

    When features are not 3, but still positive,
    the data would be expanded or shrinked to the feature input
    by multiplying new_features x 3 matrix to the original data.

    Users can expand this class by defining their own
    __generate_new_attractor() function.
    
    This class is inheriting AbstractGenerator class.
    Please refer to AbstractGenerator class for more information.

    """
    
    #Constructor
    def __init__(self, sname = "", params = {}, sample_size=1):
        self.category = params["attractor"]
        super().__init__(sname=sname,
                         params=params,
                         n_feature=params["features"],
                         sample_size=sample_size)

    # Private Functions
    def __generate_lorenz(self):
        def f(state, t):
            rho = self.params["rho"]
            sigma = self.params["sigma"]
            beta = self.params["beta"]
            x, y, z = state  # unpack the state vector
            return sigma*(y-x), x*(rho-z)-y, x*y-beta*z  # derivatives
        
        # Initialize
        initial_state = self.params["initial_state"]
        frm, dt = float(self.params["start_t"]), float(self.params["time_interval"])
        to = frm+float(dt*self.sample_size)
        t = np.arange(frm, to, dt)

        # Compute Lorenz Attractor
        states = odeint(f, initial_state, t)
        z = np.array(states).T

        # Generating Transformation Matrix or Z transpose
        n_feature = self.params["features"]
        if n_feature != 3 and n_feature > 0:
            W = np.random.uniform(-5, 5, size=(n_feature,3))
            x = W.dot(z).T # Z->X transformation [x(t) = Wz(t))]
        else:
            x = z.T

        # Creating Data Table for X
        for i in range(x.shape[1]):
            self.sens_gen_option_mapping[self.sname[i]] = "Lorenz_Attractor_"+str(i)
            self.sens_data_mapping[self.sname[i]] = x[:,i]
        
        self.generated = True

    def __generate_rossler(self):
        def f(state, t):
            a = self.params["a"]
            b = self.params["b"]
            c = self.params["c"]
            x, y, z = state  # unpack the state vector
            return -y-z, x+a*y, b+z*(x-c)  # derivatives
        
        # Initialize
        initial_state = self.params["initial_state"]
        frm, dt = float(self.params["start_t"]), float(self.params["time_interval"])
        to = frm+float(dt*self.sample_size)
        t = np.arange(frm, to, dt)

        # Compute Lorenz Attractor
        states = odeint(f, initial_state, t)
        z = np.array(states).T

        # Generating Transformation Matrix or Z transpose
        n_feature = self.params["features"]
        if n_feature != 3 and n_feature > 0:
            W = np.random.uniform(-5, 5, size=(n_feature,3))
            x = W.dot(z).T # Z->X transformation [x(t) = Wz(t))]
        else:
            x = z.T

        # Creating Data Table for X
        for i in range(x.shape[1]):
            self.sens_gen_option_mapping[self.sname[i]] = "Lorenz_Attractor_"+str(i)
            self.sens_data_mapping[self.sname[i]] = x[:,i]
        
        self.generated = True


    def __generate_another_attractor(self):
        raise NotImplementedError

    # Public Functions
    def generate(self, seed=None):
        # Seeding
        self.seed(seed)

        if self.category == "lorenz":
            self.__generate_lorenz()
        elif self.category == "rossler":
            self.__generate_rossler()
        # elif:
        #     self.__generate_another_attractor()
        else:
            raise NotImplementedError

        self.post_generation_process()
        return self.sens_data_mapping
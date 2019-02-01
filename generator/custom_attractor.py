"""
==============================
       Custom Attractor
==============================

build 2019.01.25.12.30 (stable)

contributor:
  Jong Duk Shinn


"""
from generator.abstract_generator import AbstractGenerator

import numpy as np

from sympy import sympify, symbols, lambdify
from scipy.integrate import odeint

class CustomAttractor(AbstractGenerator):
    """
    The Custom Attractor Class

    [What is this?]

    This is where users can make their own attractors as long as users define equations correctly.
    The Custom Attractor Class allows any dimension of Attractor implementation as long as
    computers can handle it.
    
    This class is inheriting AbstractGenerator class.
    Please refer to AbstractGenerator class for more information.

    """

    #Constructor
    def __init__(self, sname = "", params = {}, sample_size=1):
        self.functions = list(sympify(eq) for eq in params["at_eq"].split())
        self.variables = symbols(params["derivative_order"])

        if len(self.functions) != len(self.variables):
            out = "the number of equations and derivatives must be equal!"
            out += "\n            equations:   "+str(len(self.functions))
            out += "\n            derivatives: "+str(len(self.variables))
            raise ValueError(out)

        if len(params["initial"].keys()) != len(self.variables):
            out = "must include step and initial term for all variables: "+str(self.variables)
            out += "\n            ex) params = {\"eq\":\"x*y y*z x*z\", \"initial\":{\"x\":1, \"y\":1, \"z\":1}}"
            raise ValueError(out)

        super().__init__(sname=sname,
                         params=params,
                         n_feature=len(self.variables),
                         sample_size=sample_size)


    # Private Functions
    def __generate_custom_attractor(self):

        lambdified_f = lambdify(self.variables, self.functions, modules=["numpy"])

        def tuplify(array):
            tup = ()
            for value in array:
                if not isinstance(value, (float, np.float_)):
                    value = float(value[0])
                tup += (value,)
            return tup

        def f(state, t):
           return tuplify(lambdified_f(*state))
        
        # Initialize
        initial_state = [float(self.params["initial"][str(var)]) for var in self.variables]
        frm, dt = float(self.params["start_t"]), float(self.params["time_interval"])
        to = frm+float(dt*self.sample_size)
        t = np.arange(frm, to, dt)

        # Compute Attractor
        states = odeint(f, initial_state, t)
        x = np.array(states)

        # Creating Data Table for X
        for i in range(x.shape[1]):
            self.sens_gen_option_mapping[self.sname[i]] = "Custom_Attractor_"+str(i)
            self.sens_data_mapping[self.sname[i]] = x[:,i]
        
        self.generated = True

    # Public Functions
    def generate(self, seed=None):
        # Seeding
        self.seed(seed)
        self.__generate_custom_attractor()
        self.post_generation_process()
        return self.sens_data_mapping
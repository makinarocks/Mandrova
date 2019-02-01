"""
=======================
    Custom Equation
=======================

build 2019.01.25.12.30 (stable)

contributor:
  Jong Duk Shinn

https://www.scipy-lectures.org/advanced/sympy.html
https://docs.sympy.org/latest/tutorial/intro.html
https://github.com/sympy/sympy/wiki/Quick-examples

"""
from generator.abstract_generator import AbstractGenerator

import numpy as np

from sympy import sympify, lambdify, symbols


# Main Generator Class
class CustomEquation(AbstractGenerator):
    """
    The Custom Equation Class

    [What is this?]

    This is where users can use their own function to sample desired data form.
    The Custom Equation Class only allows a single equation as an input.
    
    This class is inheriting AbstractGenerator class.
    Please refer to AbstractGenerator class for more information.

    """

    # Constructor
    def __init__(self, sname="", params = {}, sample_size=10):
        super().__init__(sname=sname,
                         params=params,
                         n_feature=1,
                         sample_size=sample_size)

        self.functions = sympify(params["eq"])
        self.variables = list(str(var) for var in self.functions.free_symbols)

        if len(self.params["initial"].keys()) != len(self.variables):
            out = "must include step and initial term for all variables: "+str(self.variables)
            out += "\n            ex) params = {\"eq\":\"x+y+x*y\", \"step\":{\"x\":1.0, \"y\":0.5}, \"initial\":{\"x\":0, \"y\":1}}"
            raise ValueError(out)

    # Public Functions
    def generate(self, seed=None):
        """
        The Main Generation Function

        [What is this?]

        This function generates numbers based on the specified function.
        User can define either a constant step or a variable step.
        The variable step must be defined as an array of constant steps. 

        Keyword Arguments:
            seed {int} -- [seed for numpy.random] (default: {None})
        
        Raises:
            ValueError -- [raise if the length of "step" is not equal to the sample size]
        
        Returns:
            [dict] -- [returns a dictionary of {sensor-name: generated-numbers}]
        """
        
        var_at_t = {}
        for var in self.variables:
            initial = self.params["initial"][var]
            step = self.params["step"][var]
            if isinstance(step, int) or isinstance(step, float):
                if step == 0:
                    var_at_t[var] = np.full(self.sample_size, initial).astype(float)
                else:
                    end = float(initial+step*(self.sample_size-1))
                    var_at_t[var] = np.linspace(initial, end, self.sample_size)
            else:
                if len(step) == self.sample_size:
                    var_at_t[var] = np.asarray(step).flatten()
                else:
                    out = "incorrect step input: the length of step must be equal to the sample size!"
                    out += "\n length of step: "+str(len(step))
                    out += "\n sample size: "+str(self.sample_size)
                    raise ValueError(out)
        
        steps=list()
        vars_str = ""
        for var in self.variables:
            steps.append(var_at_t[var])
            vars_str += var+" "

        steps = list(map(tuple,np.asarray(steps).T))
        vars_str = symbols(vars_str)

        lambdified_f = lambdify(vars_str, self.functions, modules=["numpy"])
        samples = [lambdified_f(*t) for t in steps]

        self.sens_data_mapping[self.sname[0]] = np.array(samples).astype(float)

        self.post_generation_process()
        return self.sens_data_mapping
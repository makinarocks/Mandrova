"""
=======================
         Input
=======================

build 2019.01.25.12.30 (stable)

contributor:
  Jong Duk Shinn


"""

class Input:
    """
        The Generator Input Helper

        [what is this?]

        "Input" manages SensorDataGenerator class' generation specifications.
        Users

        [Example] input by add_option():

        input = Input()
        input.add_option("")

    """

    def __init__(self):
        """[summary]
        """

        self.options = {}
        self.spec_helper = SpecHelper()
        self.__necessary_params = list()
        for _, params in self.spec_helper.category_option.items():
            self.__necessary_params.append(params["__important"])
    
    def __check_for_important_parameters(self, params={}):
        """
        A Important Parameter Checker

        [What is this?]

        Each generator component must have its own unique parameter to specify its generator type
        during the initialization process of SensorDataGenerator class. For example,
        Stationary generator component has "distribution" as unique parameter
        so that when SensorDataGenerator attempts to call generator components,
        the Stationary generator will be called thanks to its unique parameter "distribution".

        This function checks whether the unique parameter is present in the input parameters or not.
        The function returns a single important parameter if the important parameters is well specified.
        
        Keyword Arguments:
            params {dict} -- [a dictionary of raw input parameters] (default: {{}})
        
        Raises:
            ValueError -- [raise if destination sensor name is abscent]
            ValueError -- [raise if unique and important parameter is unspecified]
            ValueError -- [raise if more than one important parameters are specified]
        
        Returns:
            [str] -- [returns the important parameter value]
        """

        if "sensor_names" not in params.keys():
            raise ValueError("\"sensor_names\" must be specified")

        indicator = 0
        important_param = None
        for param in self.__necessary_params:
            if param in params.keys():
                indicator += 1
                important_param = param

        if indicator == 0:
            raise ValueError("one of the following parameters must be specified:", self.__necessary_params)
        elif indicator > 1:
            raise ValueError("only one of the following parameters must be specified:", self.__necessary_params)
        else:
            return important_param


    # Public Functions
    def add_option(self, **kwargs):
        """
        Adds Options

        [What is this?]

        This function adds a desired option based on the user input parameters and generates an option dictionary
        for SensorDataGenerator class. Specific available options are explained as follows:
    
        [Example]

        ex1) sdg_input.add_option(sensor_names="Sensor1", distribution="normal", mu=0, sigma=2)
        ex2) sdg_input.add_option(sensor_names="Sensor2 Sensor3", distribution="multivariatenormal", mu=np.zeros(2), cov=np.eye(2))

        If SensorDataGenerator is empty,
        the numbers must be generated first or load data from either .csv or .xlsx .
        
        After generation or data loading,
        the data can be modified by SensorDataGenerator w/ modification option.

        [Supported Attractor and Its Declaration Format]

        ========================================================================
        |          Type          |     Declare as (=default parameter)         |
        ========================================================================
        |   Lorenz | Rossler     |     attractor= "lorenz" | "rossler",        |
        |                        |     rho = 28.0 | a = 0.2,                   |
        |                        |     sigma = 10.0 | b = 0.2,                 |
        |                        |     beta = 8.0/3.0 | c = 5.7,               |
        |                        |     initial_state = [1.0,1.0,1.0],          |
        |                        |     start_t = 0.0,                          |
        |                        |     time_interval = 0.1,                    |
        |                        |     features = 3  ~must be int([1, ~])      |
        ========================================================================

        [Supported Custom Attractor and Its Declaration Example]

        ========================================================================
        |          Type          |     Declare as (=just example not default)  |
        ========================================================================
        |   All                  |     at_eq = "x+y y*z-x x/z",                |
        |                        |     derivative_order="x y z",               |
        |                        |     initial = {"x":0.0, "y":1.0, "z":1.0},  |
        |                        |     start_t = 0.0,                          |
        |                        |     time_interval = 0.1                     |
        ========================================================================

        [Supported Custom Equation and Its Declaration Example]

        ========================================================================
        |          Type          |     Declare as (=default parameter)         |
        ========================================================================
        |   All                  |     eq = "x+y+x*y-x/y",                     |
        |                        |     initial = {"x":0, "y":1},               |
        |                        |     step = {"x":0.1, "y":0.1},              |
        ========================================================================

        [Supported Distributions and Its Declaration Example]
        
        ==================================================================================
        |          Type          |     Declare as (=default parameter)                   |
        ==================================================================================
        |    Normal              |    distribution = "normal", mu = 0, sigma = 1         |
        |    Log-normal          |    distribution = "lognormal", mu = 0, sigma = 1      |
        |    Multivariate-normal |    distribution = "multivariate", mu = [d0,d1,..,dn], |
        |                        |             cov = n x n positive-semidefinite matrix  |
        |    Gamma               |    distribution = "gamma", alpha = 0.5, beta = 0.5    |
        |    Beta                |    distribution = "beta",  alpha = 0.5, beta = 0.5    |
        |    Uniform             |    distribution = "uniform", lo = 0, hi = 1           |
        |    Exponential         |    distribution = "exponential", lambd = 1.0          |
        ==================================================================================
        """

        important_param = self.__check_for_important_parameters(kwargs)

        sname = kwargs["sensor_names"]
        if sname not in self.options.keys():
            self.options[sname] = list()

        specific_param = {}
        for key, value in kwargs.items():
            if key != "sensor_names":
                specific_param[key] = value
            

        generator = self.spec_helper.check_category(important_param)
        specific_param = self.spec_helper.get_complete_param_value(generator, specific_param, len(sname.split()))

        self.options[sname].append(specific_param)

    def clear(self):
        """
        A Option Clearer

        [What is this?]

        This function erases everything in self.options dictionary.
        """

        self.options.clear()

    def copula(self, sensor_names="", copula_type="gaussian", cov=None, frm=None, to=None):
        """
        A Copula Helper

        [What is this?]

        This function merges selected stationary sensors to create copula input option.  
        The following copula type(s) is/are supported:  
        1. Gaussian Copula (currently the only supported type)
        2. ...
        
        Keyword Arguments:
            sensor_names {str} -- [source sensors in previously created options] (default: {""})
            copula_type {str} -- [type of copula to be used] (default: {"gaussian"})
            cov {2D N x N Array} -- [covariance matrix of copula] (default: {None})
            frm {int} -- [range parameter] (default: {None})
            to {int} -- [range parameter] (default: {None})
        
        Raises:
            ValueError -- [raise if covariance is unspecified]
            ValueError -- [raise if the number of sensors selected is not equal to a dimension of the covariance matrix]
            ValueError -- [raise if sensor name does not exist in the pre-generated options]
            ValueError -- [raise if the selected sensor has more than one generation options]
            ValueError -- [raise if the selected sensor has non-distribution based generation option]
        """

        # Checking covariance matrix availability
        if cov is None:
            raise ValueError("please specify the covariance matrix")

        sname_list = sensor_names.split()
        cov = np.array(cov)
        if cov.shape[0] != len(sname_list):
            raise ValueError("the number of sensor names must be equal to the dimension of the covariance matrix")
        
        params = {"copula":copula_type}
        if frm is not None and isinstance(frm, (int, np.int_)):
            params["frm"] = frm
        if to is not None and isinstance(to, (int, np.int_)):
            params["to"] = to
        params["cov"] = cov

        for sn in sname_list:
            if sn not in self.options.keys():
                raise ValueError("cannot find sensor:", sn)
            if len(self.options[sn]) != 1:
                raise ValueError("cannot copula a sensor with more than one generation option!")
            if "distribution" not in self.options[sn][0].keys():
                raise ValueError("cannot copula non-stationary distribution based generator!")
            params[sn] = copy.deepcopy(self.options[sn])
            self.options.pop(sn)

        if sensor_names not in self.options.keys():
            self.options[sensor_names] = list()
        self.options[sensor_names].append(params)

    def is_empty(self):
        """
        Returns:
            [bool] -- [True if option is empty, False if not]
        """

        return len(self.options.keys()) == 0


"""
=======================
      SpecHelper
=======================

build 2019.01.31.14.30 (stable)

contributor:
  Jong Duk Shinn


"""

import re
import copy
import random
import numpy as np

from sympy import sympify

class SpecHelper:

    # Constructor
    def __init__(self):
        
        # Supported Options
        self.__stationary =     {"__important":"distribution",
                                 "normal" : "mu,sigma", 
                                 "gaussian" :  "mu,sigma", 
                                 "multivariatenormal" : "mu,cov",
                                 "lognormal" : "mu,sigma",
                                 "beta" : "alpha,beta",
                                 "gamma" : "alpha,beta",
                                 "uniform" : "lo,hi",
                                 "exponential" : "lambd",
                                 "expon" : "lambd"}

        self.__attractor =      {"__important":"attractor",
                                 "lorenz":"rho,sigma,beta,initial_state,start_t,time_interval,features",
                                 "rossler":"a,b,c,initial_state,start_t,time_interval,features"}

        self.__custom_equation= {"__important":"eq",
                                 "ce_all":"eq,step,initial"}

        self.__custom_attractor={"__important":"at_eq",
                                 "cat_all":"at_eq,derivative_order,initial,start_t,time_interval"}
        ##["add_generator_spec_here"]
      ##self.__new_generator  = {"category":"required,parameters,here"}

        # Spec Sheets
        self.category_option = {"stationary" : self.__stationary,
                                "attractor" : self.__attractor,
                                "custom_equation" : self.__custom_equation,
                                "custom_attractor": self.__custom_attractor}
                              ##"new_generator" : self.__new_generator}

        self.default_param_value = {"mu":0,
                                    "sigma":1,
                                    "alpha":0.5,
                                    "beta":0.5,
                                    "lambd":0.5,
                                    "lo":0,
                                    "hi":1}
                                  ##"new":1}


    # Private Functions
    def __just_chars(self, str=""):
        new_str = re.sub(str,r'[^a-zA-Z/]', '')
        return new_str


    # Private Functions for a Specific Generator
    def __set_missing_attractor_params(self, params={}):
        if "attractor" not in params.keys():
            raise ValueError("incorrect attractor specification:", params)

        default_params = {}
        if params["attractor"] == "lorenz":
            default_params = {"attractor":"lorenz",
                              "rho":28.0,
                              "sigma":10.0,
                              "beta":8.0/3.0,
                              "initial_state":[1.0,1.0,1.0],
                              "start_t":0.0,
                              "time_interval":0.1,
                              "anomaly":0.0,
                              "features":3}
        elif params["attractor"] == "rossler":
            default_params = {"attractor":"rossler",
                              "a":0.2,
                              "b":0.2,
                              "c":5.7,
                              "initial_state":[1.0,1.0,1.0],
                              "start_t":0.0,
                              "time_interval":0.1,
                              "anomaly":0.0,
                              "features":3}

        req_params = self.required_params(option=params["attractor"])
        for p in req_params:
            if p not in params.keys():
                params[p] = default_params[p]
        
        return params

    def __set_missing_custom_attractor_params(self, params={}):
        """
        example specification:
        params = {"at_eq":"10.0*(y-x) x*(28.0-z)-y x*y-(8.0/3.0)*z",
                  "derivative_order":"x y z",
                  "initial":{"x":1.0, "y":1.0, "z":1.0},
                  "start_t":0.0,
                  "time_interval":0.1}
        """

        if "at_eq" not in params.keys():
            raise ValueError("incorrect attractor specification:", params)

        req_params = self.required_params(option="cat_all")
        for p in req_params:
            if p not in params.keys():
                out = "unspecified parameter: "+str(p)
                out += "\nthe following parameters must be defined: "+str(req_params)
                raise ValueError(out)
        
        return params

    def __set_missing_custom_equation_params(self, params={}):
        if "eq" in params.keys():
            functions = sympify(params["eq"])
            variables = list(str(var) for var in functions.free_symbols)
            if "step" not in params.keys():
                params["step"] = {}
                for var in variables:
                    params["step"][var] = 0.1
            if "initial" not in params.keys():
                params["initial"] = {}
                for var in variables:
                    params["initial"][var] = 1.0
        
        else:
            out = "Warning! Using Default Input:\n"
            default_params = {"eq"  : "x+y-x/y+x*y+x**2",
                              "step": {"x":1.0, "y":0.5},
                              "initial":{"x":0, "y":0}}
            functions = sympify(default_params["eq"])
            variables = list(str(var) for var in functions.free_symbols)
            for k, v in default_params.items():
                out += str(k)+": "+str(v)+"\n"
            print(out)

            params = default_params

        return params

    def __set_missing_stationary_params(self, params={}, n_feature=1):
        if "distribution" not in params.keys():
            raise ValueError("incorrect distribution specification:", params)
        
        if params["distribution"] == "multivariatenormal" and n_feature == 1:
            raise ValueError("multivariatenormal distribution must have n_feature >= 2")

        req_params = self.required_params(option=params["distribution"])
        for p in req_params:
            if p not in params.keys():
                params[p] = self.get_default_param_value(p, n_feature)
        
        return params


    # Public Functions
    def check_supported_options(self, cat="all", params=False):
        # TODO
        cat = self.__just_chars(cat)
        if cat == "all":
            raise NotImplementedError
        elif cat != "all" and cat not in self.supported_generators():
            out =  "Error: Unsupported Category!: \""+str(cat)+"\"\n"
            out += "       available categories are:"
            for c in self.supported_generators():
                out += "        \""+str(c)+"\"\n"
            print(out)
            return False
        if params:
            raise NotImplementedError
        
    def check_category(self, important_param=""):
        """
            To see if which generator the option belongs to.
        
        """
        for category, option in self.category_option.items():
            if option["__important"] == important_param:
                return category

        raise ValueError("cannot find generator type w/ important parameter:", important_param)

    def get_complete_param_value(self, generator="", params={}, n_feature=1):
        """
            Generating Complete Parameters for Specified Generator and Category
        
        """
        # Unsupported Generator Type
        if generator not in self.category_option.keys():
            out = "unsupported generator: "+str(generator)+"\n"
            out += "  available generators are: "+str(self.category_option.keys())
            raise ValueError(out)

        # Stationary Generator
        if generator == "stationary":
            if "distribution" in params.keys():
                params = self.__set_missing_stationary_params(params, n_feature)

            elif "copula" in params.keys():
                for _, dist_spec in params.items():
                    dist_spec = self.__set_missing_stationary_params(dist_spec)

            else:
                raise NotImplementedError()

        # Attractor Generator
        elif generator == "attractor":
            params = self.__set_missing_attractor_params(params)

        # Custom Equation Generator
        elif generator == "custom_equation":
            params = self.__set_missing_custom_equation_params(params)

        # Custom Attractor Generator
        elif generator == "custom_attractor":
            params = self.__set_missing_custom_attractor_params(params)

        # # Future Generator
        # elif generator == "future":
        #   params = self.__set_missing_future_generator_params(params)

        else:
            raise NotImplementedError
        
        self.omit_redundant_params(params)
        return params

    def get_default_param_value(self, param="", n_feature=1):
        if param not in self.default_param_value.keys():
            out = "unsupported paramater: "+str(param)+"\n"
            out += "  support parameters are: "+str(list(self.default_param_value.keys()))
            ValueError(out)

        value = None
        if param == "cov":
            if n_feature < 2:
                raise ValueError("covarance matrix must have n_feature >= 2.")

            # Generating covariance matrix of
            #   correlation 0.7, sigma 1
            #   for all features
            value = np.eye(n_feature, dtype=float)
            value = value.flatten()
            for i in range(len(value)):
                if value[i] == 0: value[i] = 0.7
            value = value.reshape(n_feature, n_feature)
        else:
            value =  self.default_param_value[param]
        
        return value

    def omit_redundant_params(self, params={}):
        params_copy = copy.deepcopy(params)
        req_params = None
        if "distribution" in params.keys():
            req_params = self.required_params(option=params["distribution"])
        elif "eq" in params.keys():
            req_params = self.required_params(option="ce_all")
        elif "attractor" in params.keys():
            req_params = self.required_params(option=params["attractor"])
        elif "at_eq" in params.keys():
            req_params = self.required_params(option="cat_all")
        # elif "future_generator" in params.keys():
        #     req_params = self.required_params(option=params["future"] or "future_all")
        else:
            raise NotImplementedError
    
        req_params += ["frm", "to"]
        
        for p in params_copy.keys():
            if p not in req_params:
                params.pop(p)
        
        return params

    def required_params(self, option=""):
        req_params = []
        for _, param_table in self.category_option.items():
            if option in param_table.keys():
                req_params += param_table[option].split(",")
                req_params.append(param_table["__important"])

        if len(req_params) == 0:
            ValueError("could not find required parameters for:", option)
            
        return req_params

    def supported_generators(self):
        return list(self.category_option.keys())
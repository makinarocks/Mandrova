"""
=======================
      Stationary
=======================

build 2019.01.24.18.13 (stable)

contributor:
  Jong Duk Shinn


"""
from generator.abstract_generator import AbstractGenerator

import random
import numpy as np
import pandas as pd

from math import exp
from scipy import stats


class Stationary(AbstractGenerator):
    """
    The Stationary Class

    [What is this?]

    This class support the following distributions for random sampling:
      1. Normal (Gaussian)
      2. Uniform
      3. Beta
      4. Exponential
      5. Gamma
      6. Log-normal
      7. Multivariate-normal

    Given several distributions above,
    users are allowed to generated correlated data
    w/ the following copula:
      1. Gaussian 

    This class is inheriting AbstractGenerator class.
    Please refer to AbstractGenerator class for more information.

    """
    
    #Constructor
    def __init__(self, sname="", params = {}, sample_size=1):
        # variables and flags
        self.cov = None
        self.is_copula = False
        self.is_multivariatenormal = False

        # setting number of features
        n_feature = 1
        if "cov" in params.keys():
            n_feature = np.array(params["cov"]).shape[0]

        # initialization
        super().__init__(sname=sname,
                         params=params,
                         n_feature=n_feature,
                         sample_size=sample_size)

        self.__initialize()

    # Private Methods
    def __initialize(self):
        # Specification Retrieval
        # univariate dists or multivariate-normal
        if "distribution" in self.params.keys():
            if self.params["distribution"] == "multivariatenormal":
                self.is_multivariatenormal = True
                self.cov = np.array(self.params["cov"])
                if not isinstance(self.params["mu"], list) and not isinstance(self.params["mu"], np.ndarray):
                    if isinstance(self.params["mu"], int) or isinstance(self.params["mu"], float):
                        self.params["mu"] = np.full(self.cov.shape[0], self.params["mu"])
                    else:
                        raise ValueError("invalid parameter input! \"mu\" must be a list of float numbers.")

            for sn in self.sname:
                self.sens_gen_option_mapping[sn] = self.params

        # copula
        elif "copula" in self.params.keys():
            self.is_copula = True
            if self.params["copula"] == "gaussian":
                for specific_sname, specific_option in self.params.items():
                    if specific_sname == "cov":
                        self.cov = np.array(specific_option)
                    elif specific_sname not in ["copula", "frm", "to"]:
                            self.sens_gen_option_mapping[specific_sname] = specific_option[0]
            else:
                raise NotImplementedError("unsupported copula type!")
        
        else:
            raise NotImplementedError

    def __get_sample(self, sname="", x_unif_i=None):
        params = self.sens_gen_option_mapping[sname]
        dname = params["distribution"]
        if dname == "normal" or dname=="gaussian":
            dist = stats.norm(loc=params["mu"], scale=params["sigma"])
        elif dname == "uniform":
            dist = stats.uniform(loc=params["lo"], scale=params["hi"]-params["lo"]) 
        elif dname == "beta":
            dist = stats.beta(params["alpha"], params["beta"])
        elif dname == "exponential":
            dist = stats.expon(scale=1/params["lambd"])
        elif dname == "gamma":
            dist = stats.gamma(a=params["alpha"], scale=1/params["beta"])
        elif dname == "lognormal":
            dist = stats.lognorm(scale=exp(params["mu"]), s=params["sigma"])
        else:
            raise NotImplementedError

        if x_unif_i is not None:
            # Copula Sampling
            return dist.ppf(x_unif_i)
        else:
            # Univariate Sampling
            return dist.rvs(self.sample_size)


    # Public Methods
    def generate(self, seed=None):
        # Seeding
        self.seed(seed)

        if self.is_multivariatenormal:
            # Multivariate-normal
            sname = list(self.sens_gen_option_mapping.keys())[0]
            mu = self.sens_gen_option_mapping[sname]["mu"]
            mvnorm = stats.multivariate_normal(mean=mu, cov=self.cov)
            sample = mvnorm.rvs(self.sample_size)

            for index, sname in enumerate(self.sens_gen_option_mapping.keys()):
                self.sens_data_mapping[sname] = sample[:, index]

        elif self.is_copula:
            # Gaussian Copula
            mus = [0 for i in range(self.n_feature)]
            mvnorm = stats.multivariate_normal(mean=mus, cov=self.cov)
            x = mvnorm.rvs(self.sample_size)
            norm = stats.norm()
            x_unif = norm.cdf(x)

            for index, sname in enumerate(self.sens_gen_option_mapping.keys()):
                self.sens_data_mapping[sname] = self.__get_sample(sname, x_unif[:,index])
        
        else:
            # Independent Random Variable(s)
            sname = list(self.sens_gen_option_mapping.keys())[0]
            self.sens_data_mapping[sname] = self.__get_sample(sname)

        self.post_generation_process()
        return self.sens_data_mapping

    def save_cov_as_csv(self, fn="hello_cov"):
        if self.cov is None:
            raise ValueError("covariance does not exist")
        
        fn += ".csv"
        df = pd.DataFrame(self.cov)
        df.to_csv(fn, sep=',', encoding='utf-8')
        print("  data saved as:",fn)
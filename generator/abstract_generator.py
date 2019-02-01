"""
=======================
   Abstract Generator
=======================

build 2019.01.24.18.13 (stable)

contributor:
  Jong Duk Shinn


"""

import copy
import random
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.pyplot import savefig
from sklearn.preprocessing import Normalizer, StandardScaler


class AbstractGenerator:
    """
    A Generator Component For SensorDataGenerator class

    [What is this?]

    This class is an abstract class that is compatible with SensorDataGenerator class.
    """


    # Constructor
    def __init__(self, sname="", params={}, n_feature=1, sample_size=1):

        # Unsigned Max Integers
        self.MAX_INT = 2**31-1

        # Main Variables
        self.n_feature = n_feature
        self.sname = self.__sname_configurator(sname, n_feature)
        self.params = params
        self.sample_size = sample_size
        self.sens_gen_option_mapping = {}
        self.sens_data_mapping = {}
        self.__original_backup = {}

        # flags
        self.generated = False
        self.is_standardized = False
        self.is_normalized = False


    # Private Methods
    def __sname_configurator(self, sname, n_feature):
        # self.sname configuration
        self.sname = sname.split()
        len_sname = len(self.sname)
        if len_sname == 0:
            raise ValueError("sensor names unspecified!")
        if len_sname != n_feature:
            raise ValueError("incorrect number of sensor names: sname: "+str(self.sname)+", n_feature: "+str(n_feature))
        
        return self.sname


    # Public Methods
    def generate(self, seed=None):
        """[summary]
        
        Keyword Arguments:
            seed {[type]} -- [description] (default: {None})
        
        Raises:
            NotImplementedError -- [description]
        """
        raise NotImplementedError

    def post_generation_process(self):
        """
            Post Processing Function for Generation Function

            [What is this?]

            This function copies generated samples from child generator classes
            and save to __original_backup for future needs.
            
            [How to use this function w/ child generator classes?]

            Simply call "self.__post_generation_generate()" at the end of each child's 
            "generate()" function to save generated samples into "self.__original_backup".

            [Example Implementation]

            class ChildGen(AbstractGenerator):
                def __init__(self, ...):
                    self....

                def generate(self, ...):
                    ...
                    ...
                    super().__post_generation_generate()
                    return ...
        """
        self.__original_backup = copy.deepcopy(self.sens_data_mapping)
        self.generated = True

    def get_gen_option(self, sensor=None, info=False):
        """
            Returns Options Used to Generate Samples

            [What is this?]

            This function returns names of generation options for each/all sensor(s).
            To get generation option for a specific sensor,
                set parameter"sensor=DesiredName".
            To print-out the numbers generated to console,
                set parameter "info=True".

        """
        if sensor == None:
            if info: print(self.sens_gen_option_mapping)
            return self.sens_gen_option_mapping
        else:
            if sensor in self.sens_gen_option_mapping.keys():
                if info: print(self.sens_gen_option_mapping[sensor])
                return self.sens_gen_option_mapping[sensor]
            else:
                print("Error: Sensor \""+str(sensor)+"\" does not exist!")

        return False

    def get_generated_numbers(self, sensor=None, info=False):
        """
            Returns Numbers Generated

            [What is this?]

            This function returns generated sample for each/all sensor(s).
            To get generated samples of a specific sensor,
                set parameter "sensor="DesiredSensorName"".
            To print-out generated samples,
                set parameter "info=True".
        """
        if sensor == None:
            if info: print(self.sens_data_mapping)
            return self.sens_data_mapping
        else:
            if sensor in self.sens_data_mapping.keys():
                if info: print(self.sens_data_mapping[sensor])
                return self.sens_data_mapping[sensor]
            else:
                print("Error: Sensor \""+str(sensor)+"\" does not exist!")

        return False

    def get_original_data(self, info=False):
        """
            Returns Originally Created Samples by "generate()"

            [What is this?]

        """
        if info: print(self.__original_backup)
        return self.__original_backup

    def histogram(self, save_as=None):
        """
             A Histogram of All Sensor Data

             [What is this?]

             This function plots histograms of all sensor data
             based on "self.sens_data_mapping" in one graph.

             This function is available after initial data generation.
        """

        if not self.generated:
            print("Error: Please Generate Numbers First!")
            return False

        title, data_list = "", list()
        plt.figure()
        for sname, data in self.sens_data_mapping.items():
            title += sname+" "
            data_list.append(data)
        plt.hist(data_list)
        plt.xlabel('numbers generated')
        plt.ylabel('frequency')
        plt.title(title)
        if save_as != None:
            fn = save_as+".png"
            savefig(fn)
        plt.show()

    def seed(self, number=None):
        """
            Setting Seed for Generation

            [What is this?]

            This function sets the seeds to Numpy, Scipy, Random based generation methods.
            To specify the seed (integer),
                set parameter "seed=DesiredInteger".

            By default, pseudo-random integer will be used.
        """
        if number is None:
            number = random.randint(0, self.MAX_INT)
        
        np.random.seed(number)
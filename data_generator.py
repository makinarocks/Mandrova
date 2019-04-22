"""
=======================
 Sensor Data Generator
=======================

version 0.1
build 2019.01.21.18.30 (stable)

contributor:
  Jong Duk Shinn

"""

import copy
import random
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import random, linalg, stats
from sklearn.preprocessing import Normalizer, StandardScaler

from generator.utils import Input
from generator.stationary import Stationary as st
from generator.attractor import Attractor as at
from generator.custom_equation import CustomEquation as ce
from generator.custom_attractor import CustomAttractor as cat
##[import_new_generator]
##from generator.future_generator import FutureGen as fg

from matplotlib.pyplot import savefig


# Setting Numpy linewidth option to 128 characters
np.set_printoptions(linewidth=128)

class SensorDataGenerator(object):
    """
    The Sensor Data Generator

    [What is this?]

    This class is the highest level generator among all generator modules.
    This class takes an user input and generates data table
    based on the user input.

    Please refer to "generator.utils Input" for input details
    Please refer to "generator.abstract_generator" for basic generator module details
    Generator modules are available at "generator".
    
    """

    # Constructor
    def __init__(self):

        # Generation Input Features:
        self.generation_input = Input()

        # Core Generator Variables
        self.data = None
        self.sample_size = None
        self.__version__ = 0.1

        self.MAX_INT = 2**31 - 1
        self.MIN_POS_FLOAT = np.nextafter(0, 1)


    # Private Methods for Ordinary Generation
    def __initialize(self, input={}):
        """
        A Generator Initialization Helper

        [What is this?]

        This function distributes a set of generation specification inputs
        into each specified generator and initializes all generator components.

        This function checks the range parameters "frm" and "to",
        and returns generators for each sensor and its generation range,
        which will be used in "generate" function.

        Keyword Arguments:
            input {dict} -- [generation input from "generation_input"]
        
        Returns:
            [dict] -- [returns a dictionary of sensor names and generator objects]
            [dict] -- [returns a dictionary of sensor names and the range parameters]
        """

        sens_generator_mapping = {}
        sens_frm_to_mapping = {}
        for sname, params_list in input.items():
            if sname not in sens_generator_mapping.keys():
                sens_generator_mapping[sname] = list()
            for params in params_list:
                frm, to = 0, self.sample_size
                # setting "from"
                if "frm" in params.keys():
                    if params["frm"] >= self.sample_size or frm < 0:
                        raise ValueError("invalid range: 0 <= frm < to <= sample_size! frm:",params["frm"], "sample size:", self.sample_size)
                    else:
                        frm = params["frm"]

                # setting "to"
                if "to" in params.keys():
                    if params["to"] > self.sample_size or params["to"] <= 0:
                        raise ValueError("invalid range: 0 <= frm < to <= sample_size! to:", params["to"], "sample size:", self.sample_size)
                    else:
                        to = params["to"]
                
                frm, to = int(frm), int(to)
                if frm >= to:
                    raise ValueError("invalid range: 0 <= frm < to <= sample_size! frm, to:", frm, to, "sample size:", self.sample_size)

                sample_size = to-frm
                for sn in sname.split():
                    if sn not in sens_frm_to_mapping.keys():
                        sens_frm_to_mapping[sn] = list()
                    sens_frm_to_mapping[sn].append([frm, to])

                if "distribution" in params.keys() or "copula" in params.keys():
                    sens_generator_mapping[sname].append(st(sname=sname, params=params, sample_size=sample_size))
                elif "attractor" in params.keys():
                    sens_generator_mapping[sname].append(at(sname=sname, params=params, sample_size=sample_size))
                elif "eq" in params.keys():
                    sens_generator_mapping[sname].append(ce(sname=sname, params=params, sample_size=sample_size))
                elif "at_eq" in params.keys():
                    sens_generator_mapping[sname].append(cat(sname=sname, params=params, sample_size=sample_size))
                ##[initialize_new_generator]
                # elif "unique_parameter" in params.keys():
                #     sens_generator_mapping[sname].append(hello(sname=sname, params=params, sample_size=sample_size))

                else:
                    raise NotImplementedError

        return sens_generator_mapping, sens_frm_to_mapping

    def __generate_from_generator(self, sens_generator_mapping, seed=None):
        """
        A Generation Caller From Individual Generator(s).

        [What is this?]

        This function attempts to call "generate" from each generator
        inheriting AbsractGenerator class. The function returns a dictionary of
        generated numbers w/ their corresponding sensor name as the key.

        Arguments:
            sens_generator_mapping {dict} -- [generated data: (key: sensor name, value: generated numbers)]
        
        Keyword Arguments:
            seed {int} -- [a seed number for random number generation] (default: {None})
        
        Returns:
            [dict] -- [returns a dictionary that has sensor names as keys and generated numbers as values]
        """

        sens_data_mapping = {}
        for _, gen_list in sens_generator_mapping.items():
            for gen in gen_list:
                sens_data = gen.generate(seed)
                for sn, data in sens_data.items():
                    if sn not in sens_data_mapping.keys():
                        sens_data_mapping[sn] = list()
                    sens_data_mapping[sn].append(data)

        return sens_data_mapping


    # Paremeter Checking Functions
    def __check_data_existency(self):
        """
        A Data Existency Checker

        [What is this?]

        This function checks if data table was generated in the past.
        The function raises an error msg if self.data is None.

        Raises:
            ValueError -- [raise if never generated data]
        """

        if self.data is None:
            raise ValueError("please generate numbers first!")

    def __check_frm_to(self, frm=None, to=None, max_size=None):
        """
        A Range Checker for Range Parameters (frm, to)

        [What is this?]

        This function checks whether the two range parameter "frm" and "to" are
        correctly specified or not. This function corrects the two parameters
        to its default value if None is specified to any of the two parameters.
        If the range of the parameters are wrong, this function raises ValueError
        to notify users that the parameter(s) is/are out of range.

        The default values are as follows:
        frm = 0
        to = self.sample_size
        max_size = self.sample_size
        
        Keyword Arguments:
            frm {int} -- [range parameter: start] (default: {None})
            to {int} -- [range parameter: end] (default: {None})
            max_size {int} -- [upper limit of "to"] (default: {None})
        
        Raises:
            ValueError -- [raise if frm is not an integer]
            ValueError -- [raise if frm is out of correct range]
            ValueError -- [raise if to is not an integer]
            ValueError -- [raise if to is out of correct range]
        Returns:
            [int, int] -- [returns frm, to]
        """

        if max_size is None:
            max_size = self.sample_size

        if frm is None:
            frm = 0
        else:
            if not isinstance(frm, int):
                raise ValueError("frm must be an integer!")
            else:
                if frm >= max_size or frm < 0:
                    raise ValueError("frm must be: 0 <= frm < to <= "+str(max_size)+" !")

        if to is None:
            to = max_size
        else:
            if not isinstance(to, int):
                raise ValueError("to must be an integer!")
            else:
                if to > max_size or to < 0 or to <= frm:
                    raise ValueError("to must be: 0 <= frm < to <= "+str(max_size)+" !")
        return frm, to

    def __check_sensor(self, sensors=[]):
        """
        A Sensor Existency Checker

        [What is this?]

        This function checks whether specified sensors list "sensors"
        only contains existing sensor names or not.

        Keyword Arguments:
            sensors {list} -- [list of sensor names] (default: {[]})
        
        Raises:
            ValueError -- [raise if cannot find sensor from the data table]
        """

        if sensors is None:
            raise ValueError("please specify sensor name(s) in list form! ex) [\"Sensor1\", \"Sensor2\", ...]")
        invalid_sname = list(set(sensors)-set(self.sensor_names()))
        if len(invalid_sname) != 0:
            raise ValueError("Sensor: "+invalid_sname+" does not exist!")


    # Basic Calculation Among Sensors
    def __check_calc_params(self, sensors=[], save_to=None, frm=None, to=None):
        """
        A Calculation Parameter Checker

        [What is this?]
        
        This function checks required parameter values for sum, sub, mult, div functions.
        The default parameters are set as follows:
        - sensors = list of all sensor names if empty list
        - save_to = sensors[0]
        - frm = 0
        - to = data table sample size

        Keyword Arguments:
            sensors {list} -- [list of specified sensor names] (default: {[]})
            save_to {str} -- [name of destination sensor] (default: {None})
            frm {int} -- [range parameter] (default: {None})
            to {int} -- [range parameter] (default: {None})
        
        Returns:
            [list, str, int, int] -- [returns list of sensor names, destination sensor name, and range parameters]
        """

        if len(sensors) == 0: sensors = self.sensor_names()
        else: self.__check_sensor(sensors)
        frm, to = self.__check_frm_to(frm, to)
        if save_to is None:
            save_to = sensors[0]
        return sensors, save_to, frm, to

    def sum(self, sensors=[], save_to=None, frm=None, to=None):
        """
        Summation:
        result = Sensor1 + Sensor2 + ... + SensorN

        Keyword Arguments:
            sensors {list} -- [list of specified sensor names] (default: {[]})
            save_to {str} -- [name of destination sensor] (default: {None})
            frm {int} -- [range parameter] (default: {None})
            to {int} -- [range parameter] (default: {None})
        """

        sensors, save_to, frm, to = self.__check_calc_params(sensors, save_to, frm, to)
        new_data = self.data[sensors][frm:to].sum(axis=1)
        self.__assign_new_data(save_to, frm, to, new_data)

    def sub(self, sensors=[], save_to=None, frm=None, to=None):
        """
        Subtraction:
        result = Sensor1 - Sensor2 - ... - SensorN

        Keyword Arguments:
            sensors {list} -- [list of specified sensor names] (default: {[]})
            save_to {str} -- [name of destination sensor] (default: {None})
            frm {int} -- [range parameter] (default: {None})
            to {int} -- [range parameter] (default: {None})
        """

        sensors, save_to, frm, to = self.__check_calc_params(sensors, save_to, frm, to)
        new_data = self.data[sensors[0]][frm:to].sub(self.data[sensors[1:]][frm:to].sum(axis=1))
        self.__assign_new_data(save_to, frm, to, new_data)

    def mult(self, sensors=[], save_to=None, frm=None, to=None):
        """
        Multiplication:
        result = Sensor1 * Sensor2 * ... * SensorN

        Keyword Arguments:
            sensors {list} -- [list of specified sensor names] (default: {[]})
            save_to {str} -- [name of destination sensor] (default: {None})
            frm {int} -- [range parameter] (default: {None})
            to {int} -- [range parameter] (default: {None})
        """

        sensors, save_to, frm, to = self.__check_calc_params(sensors, save_to, frm, to)
        new_data = 1
        for sname in sensors:
            new_data *= self.data[sname][frm:to]
        self.__assign_new_data(save_to, frm, to, new_data)

    def div(self, sensors=[], save_to=None, frm=None, to=None):
        """
        Division:
        result = Sensor1 / Sensor2 / ... / SensorN

        Keyword Arguments:
            sensors {list} -- [list of specified sensor names] (default: {[]})
            save_to {str} -- [name of destination sensor] (default: {None})
            frm {int} -- [range parameter] (default: {None})
            to {int} -- [range parameter] (default: {None})
        """

        sensors, save_to, frm, to = self.__check_calc_params(sensors, save_to, frm, to)
        new_data = self.data[sensors[0]][frm:to]
        for sname in sensors[1:]:
            new_data /= self.data[sname][frm:to]
        self.__assign_new_data(save_to, frm, to, new_data)


    # General Methods
    def add_label(self, save_to="", label="", frm=None, to=None):
        """
        A Labeling Helper

        [What is this?]

        This function helps users to create labels

        This function uses "inject" function to make a label.
        For more information, please refer to "inject" function.

        Keyword Arguments:
            save_to {str} -- [destination sensor name] (default: {""})
            label {str, int, float} -- [desired label] (default: {""})
            frm {int} -- [range parameter] (default: {None})
            to {int} -- [range parameter] (default: {None})
        
        Returns:
            [pandas.DataFrame] -- [returns changed data table]
        """

        self.__check_data_existency()
        frm, to = self.__check_frm_to(frm, to)
        index = np.arange(frm, to, 1)

        self.inject(value=label, sensor=save_to, index=index)
        
        return self.data

    def add_index(self, indices=[]):
        """
        A Data Table Index Helper

        [What is this?]

        This function takes "indices", which should be a list of unique index,
        and sets "indices" as the index of data table.
        
        Keyword Arguments:
            indices {list} -- [a list of unique index] (default: {[]})
        
        Raises:
            ValueError -- [raise if not all indices are unique]
            ValueError -- [raise if the number of index is not equal to data table's sample size]
        """

        self.__check_data_existency()
        if len(set(indices)) != len(indices):
            raise ValueError("each index must be unique!")
        if len(indices) != self.sample_size:
            out = "incorrect length of indices!"
            out += " indices must have the same length as sample_size!"
            out += "\n indices: "+str(len(indices))
            out += "\n sample_size: "+str(self.sample_size)
            raise ValueError(out)
        
        self.data.index = indices
        
        return self.data

    def add_time(self, year=2017, month=12, date=21, time_interval=[]):
        """
        A Time Stamp Generator

        [What is this?]

        This function sets date indices to the data table.
        The time interval of the indices can be specified in two ways:
        1. a single integer/float:      ex) time_interval = 0.5 (secs)
        2. an array of integers/floats: ex) np.random.uniform(0, 1, dg.sample_size)
                                        w/ the array option, the number of elements in the array
                                        must be equal to the sample size of the data generator.
        
        This function uses "add_index" function.
        For more information, please refer to "add_index" function.


        Keyword Arguments:
            year {int} -- [start year] (default: {2017})
            month {int} -- [start month] (default: {12})
            date {int} -- [start date] (default: {21})
            time_interval {start int} -- [time interval of data table indices] (default: {1})
        
        Returns:
            [pandas.DataFrame] -- [returns changed data table]
        """

        self.__check_data_existency()
        step = None
        step_is_array = False
        if isinstance(time_interval, int):
            step = datetime.timedelta(seconds=time_interval)
        else:
            time_interval = np.array(time_interval)
            if time_interval.shape[0] != self.sample_size:
                out = "incorrect length of time_interval!"
                out += " time_interval must have the same length as sample_size!"
                out += "\n time_interval: "+str(time_interval.shape[0])
                out += "\n sample_size: "+str(self.sample_size)
                raise ValueError(out)
            else:
                step_is_array = True
        
        time_stamp = list()
        dt = datetime.datetime(year, month, date)
        for i in range(self.sample_size):
            time_stamp.append(dt.strftime('%Y-%m-%d %H:%M:%S'))
            if step_is_array:
                ti = int(time_interval[i])
                step = datetime.timedelta(seconds=ti)
            dt += step

        return self.add_index(indices=time_stamp)

    def copy(self):
        """
        A SensorDataGenerator Object Copy Helper

        [What is this?]

        This function copies a whole SensorDataGenerator object (itself).

        Returns:
            [SensorDataGenerator object] -- [an independent deep copy of SensorDataGenerator object]
        """

        return copy.deepcopy(self)

    def sensor_names(self):
        """
        Method To Get All Sensor Names

        [What is this?]

        This function returns all sensor names.

        Returns:
            [list] -- [returns list of all sensor names]
        """

        self.__check_data_existency()
        return list(self.data.columns)

    def load_from_csv(self, file_name=""):
        """
        A .csv Data File Loader

        [What is this?]

        This function loads data from .csv file.

        Keyword Arguments:
            file_name {str} -- [filename to load] (default: {""})
        
        Returns:
            [pandas.DataFrame] -- [returns loaded data table]
        """

        self.data = pd.read_csv(file_name)
        self.sample_size = self.data.shape[0]
        return self.data

    def load_from_excel(self, file_name="", sheet_name=""):
        """
        A .xlsx Data File Loader

        [What is this?]

        This function loads data from .xlsx file.

        Keyword Arguments:
            file_name {str} -- [filename to load] (default: {""})
            sheet_name {str} -- [sheet name to load] (default: {""})
        
        Returns:
            [pandas.DataFrame] -- [returns loaded data table]
        """

        self.data = pd.read_excel(io=open(file_name, 'rb'), sheet_name=sheet_name)
        self.sample_size = self.data.shape[0]
        return self.data

    def plot_data(self, file_name=None, sensors=[], just_save=False):
        """
        A Data Ploting Helper

        [What is this?]

        This function plots sensor data of specified sensor(s).

        Keyword Arguments:
            file_name {[type]} -- [filename to be used to save the plot] (default: {None})
            sensors {list} -- [list of sensor names] (default: {[]})
            just_save {bool} -- [True iff users want to save plot, False if not] (default: {False})
        """

        self.__check_data_existency()
        if len(sensors) == 0:
            sensors = self.sensor_names()
        fig, ax = plt.subplots()
        fig.set_size_inches(27, 4.5)
        fig.suptitle("Generated Numbers Plot", fontsize=16)
        plt.ylabel('numbers')
        for sname in sensors: 
            plt.plot(self.data[sname], label=sname, marker=".")

        ax.legend(loc='center right', bbox_to_anchor=(1.12, 0.5), shadow=True, ncol=1)
        num_data = len(self.data)
        amount = int(0.2*self.data.shape[0])
        loc = [i for i in range(num_data) if i%amount == 0 and i >= 0 and i < num_data or i == num_data-1]
        stamp = [self.data.index.values[i] for i in range(num_data) if i%amount == 0 and i >= 0 and i < num_data or i == num_data-1]
        plt.xticks(loc, stamp)
        if file_name is not None: 
            file_name = file_name+".png"
            savefig(file_name)
            print("  result saved as:",file_name)

        if not just_save: plt.show()

    def shape(self):
        """
        A Shape Information Helper

        [What is this?]

        This function returns the shape of data table.
        
        Returns:
            [tuple] -- [returns (row, col)]
        """

        return self.data.shape

    def save_as_csv(self, file_name="hello_world", frm=None, to=None):
        """
        Data Export Helper [.csv]
        
        Keyword Arguments:
            file_name {str} -- [filename to be used to save data table] (default: {"hello_world"})
            frm {[type]} -- [range parameter] (default: {None})
            to {[type]} -- [range parameter] (default: {None})
        """

        self.__check_data_existency()
        frm, to = self.__check_frm_to(frm, to)
        file_name += ".csv"
        self.data[frm:to].to_csv(file_name, sep=',', encoding='utf-8')
        print("  data saved as:",file_name)

    def save_as_excel(self, file_name="hello_world", sheet_name="data", frm=None, to=None):
        """
        
        Keyword Arguments:
            file_name {str} -- [filename to be used to save data table] (default: {"hello_world"})
            frm {[type]} -- [range parameter] (default: {None})
            to {[type]} -- [range parameter] (default: {None})
        """

        self.__check_data_existency()
        frm, to = self.__check_frm_to(frm, to)
        file_name += ".xlsx"
        writer = pd.ExcelWriter(file_name)
        self.data.to_excel(writer, sheet_name)
        writer.save()
        print("  data saved as:",file_name)

    def seed(self, number=None):
        """
        Setting Seed for Generation

        [What is this?]

        This function sets the seeds to Numpy, Scipy, Random based generation methods.
        To specify the seed (integer),
            set parameter "seed=DesiredInteger".

        By default, pseudo-random integer will be used.
        
        Keyword Arguments:
            number {int} -- [the seed number to be used] (default: {None})

        Raises:
            ValueError -- [raise if the seed input is not an integer]
        """

        if number is None:
            number = random.randint(0, self.MAX_INT)
        
        np.random.seed(number)


    # Generation Methods
    def generate(self, sample_size=1000, resize=False, seed=None):
        """
        The Main Generation Function

        [What is this?]

        This function generates numbers to specified sensors by generation input,
        and saves the numbers into the data table as pandas.DataFrame .

        Therefore, options must have been added to generation_input  
        through "generation_input.add_option" or other input options such as "copula"  
        in order to generate numbers with this function.

        Users can control the seed for random number generation  
        by specifying custom "seed" parameter.

        Once data table is generated,  
        users are allowed to resize data table with another generation.  
        This option is only applied iff the sample_size input is different from  
        previously used sample_size.
        
        Keyword Arguments:
            sample_size {int} -- [the number of samples to generate] (default: {1000})
            resize {bool} -- [True to resize data table w/ new sample_size, False if not] (default: {False})
            seed {[type]} -- [seed for generation] (default: {None})
        
        Raises:
            ValueError -- [raise if sample_size input is invalid]
            ValueError -- [raise if sample_size is undefined]
            ValueError -- [raise if users attempt to resize data table w/o generating data table beforehand]
        
        Returns:
            [pandas.DataFrame] -- [returns generated data table]
        """

        # error checking
        if self.sample_size is None:
            self.sample_size = sample_size
        if not isinstance(sample_size, int) or sample_size <= 0:
            raise ValueError("sample size must be an integer and greater than 0")
        if self.data is None and self.sample_size is None:
            raise ValueError("sample size must be defined for initial generation!")
        if resize and self.sample_size != sample_size:
            if self.data is None:
                raise ValueError("data table does not exist! please generate numbers first!")
            if self.sample_size > sample_size:
                self.data = self.data.iloc[:self.sample_size,]
            elif self.sample_size < sample_size:
                new_data_table_dict = {}
                additional_num_rows = sample_size-self.sample_size
                for col in list(self.data.columns):
                    col_data = np.array(self.data[col])
                    col_data = np.append(col_data, np.full(additional_num_rows, None))
                    new_data_table_dict[col] = col_data
                self.data = pd.DataFrame(new_data_table_dict)

            self.sample_size = sample_size

        # data generation
        sens_generator_mapping, sens_frm_to_mapping = self.__initialize(input=self.generation_input.options)
        sens_data_mapping = self.__generate_from_generator(sens_generator_mapping, seed)
        temp_data_table = {}
        for sn, data_list in sens_data_mapping.items():
            if self.data is None:
                if sn not in temp_data_table.keys():
                    temp_data_table[sn] = np.full(self.sample_size, None)
            else:
                if sn not in list(self.data.columns):
                    self.data[sn] = np.full(self.sample_size, None)
            for i, data in enumerate(data_list):
                frm = sens_frm_to_mapping[sn][i][0]
                to = sens_frm_to_mapping[sn][i][1]
                if self.data is None:
                    temp_data_table[sn][frm:to] = data
                else:
                    self.__assign_new_data(sn, frm, to, data)
        
        if self.data is None:
            self.data = pd.DataFrame(temp_data_table)

        return self.data

    def generate_covmat(self, dim=2, sigmas=None, corrs=None, eigs=None, seed=None):
        """
        A Covariance Matrix Generator

        [What is this?]

        This function generates a covariance matrix given sigmas and correlation matrix.
        By default, a 2x2 covariance matrix w/ stdev=1 and random eigenvalue is generated.
        
        Keyword Arguments:
            dim {int} -- [dimension of covariance matrix] (default: {2})
            sigmas {list} -- [a list of standard deviations] (default: {None})
            corrs {np.ndarray} -- [correlation matrix] (default: {None})
            seed {int} -- [seed for numpy.random] (default: {None})
        
        Raises:
            ValueError -- [raise if sigmas < 0]
            ValueError -- [raise if len(sigmas) != dim]
            ValueError -- [raise if correlation is not n by n square matrix]
            ValueError -- [raise if correlation value not in [-1.0, 1.0]]
        
        Returns:
            [numpy.array] -- [returns Dim x Dim covariance matrix]
        """

        if sigmas is not None:
            if (np.array(sigmas) < 0.0).all():
                raise ValueError("all sigmas must be greater than or equal to 0.")
            if len(sigmas) != dim:
                raise ValueError("the length of sigmas must be equal to \"dim\". length of sigmas:", len(sigmas))
        else:
            sigmas = np.full(dim, 1.0)
        if corrs is not None:
            if not isinstance(corrs, np.ndarray) and corrs.shape != (dim, dim):
                raise ValueError("corrs must be a \"dim\" x \"dim\" numpy matrix.")
            if (-1.0 > corrs).all() or (corrs > 1.0).all():
                raise ValueError("each correlation value must be in [-1.0, 1.0].")
        else:
            self.seed(seed)
            if eigs is None:
                eigs = np.random.randint(1, self.MAX_INT, dim)
                eigs = eigs/eigs.sum()
                eigs *= dim
            self.seed(seed)
            corrs = stats.random_correlation.rvs(eigs)

        return np.diag(sigmas).dot(corrs).dot(np.diag(sigmas))


    # Modification Methods
    def __assign_new_data(self, destination="", frm=None, to=None, new_data=None):
        """
        A Data Column Replacement/Relocation/Creation Helper

        [What is this?]

        This function safely creates/replaces data in the data table
        to "new_data" given frm & to range parameter.

        The range parameters are set to 0 and sample size respectively  
        by default if nothing is specified.
        
        Keyword Arguments:
            destination {str} -- [destination sensor name] (default: {""})
            frm {int} -- [range parameter] (default: {None})
            to {int} -- [range parameter] (default: {None})
            new_data {numpy.array} -- [data to be saved] (default: {None})
        """

        if frm is None and to is None:
            frm, to = self.__check_frm_to(frm, to, self.sample_size)
        if destination not in self.sensor_names():
            self.data[destination] = np.zeros(self.sample_size)
        destination_data_column = np.array(self.data[destination])
        destination_data_column[frm:to] = new_data
        self.data[destination] = destination_data_column


    def drop_sensors(self, sensors=[]):
        """
        A Sensor Dropper

        [What is this?]

        This function drops specified sensors in "sensors"
        from the data table.
        
        Keyword Arguments:
            sensors {list} -- [list of sensor names] (default: {[]})
        
        Returns:
            [pandas.DataFrame] -- [returns changed data table]
        """

        self.__check_data_existency()
        self.data = self.data.drop(columns=sensors)
        return self.data

    def duplicate(self, sensor=None, save_to=None, frm=None, to=None):
        """
        A Sensor Data Duplication Helper

        [What is this?]

        This function duplicates selected sensor data to destination sensor.
        The range parameters "frm" and "to" are set to 0 and sample_size  
        if users do not specify. 
        
        Keyword Arguments:
            sensor {str} -- [name of source sensor] (default: {None})
            save_to {str} -- [name of destination sensor] (default: {None})
            frm {int} -- [range parameter] (default: {None})
            to {int} -- [range parameter] (default: {None})
        """

        self.__check_data_existency()
        self.__check_sensor(sensors=[sensor])
        frm, to = self.__check_frm_to(frm, to, self.sample_size)
        if save_to is None:
            save_to = sensor+"DUP"

        new_data = copy.deepcopy(self.data[sensor])
        self.__assign_new_data(save_to, frm, to, new_data)
        return self.data

    def inject(self, value=None, sensor=None, index=-1):
        """
        A Value Injection Helper

        [What is this?]

        This function takes a sensor name and index/indices to set new value as "value" parameter.

        Keyword Arguments:
            value {any value} -- [desired value] (default: {None})
            sensor {str} -- [destination sensor] (default: {None})
            index {int, np.array} -- [desired index/indices] (default: {-1})
        
        Raises:
            ValueError -- [raise if index is out of range]
            ValueError -- [raise if some indices are out of range]
        """

        self.__check_data_existency()
        self.__check_sensor(sensors=[sensor])
        if isinstance(index, (int, np.int_)):
            if index < 0 or index >= self.sample_size:
                raise ValueError("index must be: 0 <= index < sample size")
        else:
            index = np.array(index)
            if (index < 0).any() or (index >= self.sample_size).any():
                raise ValueError("indices must be: 0 <= index < sample size")

        new_data = np.array(self.data[sensor])
        new_data[index] = value
        self.__assign_new_data(destination=sensor, new_data=new_data)

    def multinomial_process(self, sensors=[], pval=[], sample_wise=False, save_to="MP_SENSOR", keep_old=True, seed=None, frm=None, to=None):
        """
        A Multinomial Sensor Data Selector

        [What is this?]

        This function selects a desired value for each iteration of selected sensor data
        by using probability value for each sensor data column w/ multinomial process and
        saves into a destination sensor "save_to" to corresponding index.
        
        Keyword Arguments:
            sensors {list} -- [list of source sensors] (default: {[]})
            pval {list, np.array} -- [list or 2D array of pvals for each sensor] (default: {[]})
            sample_wise {bool} -- [True if pval changes by sample, False if not] (default: {False})
            save_to {str} -- [name of destination sensor] (default: {"MP_SENSOR"})
            keep_old {bool} -- [True to keep processed sensor data, False if not] (default: {True})
            seed {[type]} -- [seed for numpy.random] (default: {None})
            frm {[type]} -- [range parameter] (default: {None})
            to {[type]} -- [range parameter] (default: {None})
        
        Raises:
            ValueError -- [raise if the number of pvals is insufficient or overly provided (not sample_wise)]
            ValueError -- [raise if the sum of pvals is not equal to 1]
            ValueError -- [raise if the number of pvals is insufficient or overly provided within a single time step (sample_wise)]
            ValueError -- [raise if the number of pvals samples is insufficient or overly provided]
            ValueError -- [raise if the sum of pvals within a single time step is not equal to 1]
        
        Returns:
            [pandas.DataFrame] -- [returns changed data table]
        """

        self.__check_data_existency()
        self.__check_sensor(sensors)
        frm, to = self.__check_frm_to(frm, to)
        desired_size = to-frm

        if not sample_wise:
            if len(sensors) != len(pval):
                raise ValueError("incorrect number of sensor name and probability input:", sensors, pval)
            if np.sum(pval) != 1:
                raise ValueError("incorrect probability input:", pval)
        else:
            pval = np.array(pval)
            if pval.shape[1] != len(sensors):
                raise ValueError("incorrect number of sensor name and probability input:", sensors, pval)
            if pval.shape[0] != desired_size:
                raise ValueError("the number of pval samples is not equal to the desired modification size!", pval.shape[0], desired_size)
            for pv in pval:
                if np.sum(pv) != 1:
                    raise ValueError("incorrect probability input:", pval)

        self.seed(seed)
        new_data = np.zeros(desired_size)
        for i in range(desired_size):
            pv = pval if not sample_wise else pval[i]
            for index, value in enumerate(np.random.multinomial(1, pv)):
                if value == 1:
                    new_data[i] = self.data[sensors[index]][i]

        if not keep_old:
            if save_to in sensors:
                sensors.remove(save_to)
            self.drop_sensors(sensors)

        self.__assign_new_data(save_to,frm,to,new_data)
        
        return self.data

    def randomly_inject_null(self, sensor=[], seed=None):
        """
        A Random Null Injector

        [What is this?]
        
        Keyword Arguments:
            sensor {list} -- [list of sensor names] (default: {[]})
            seed {int} -- [seed for random generation] (default: {None})
        
        Returns:
            [pandas.DataFrame] -- [returns changed data talbe]
        """

        self.__check_data_existency()
        if seed is None:
            seed = random.randint(0,self.MAX_INT)
        random.seed(seed)

        if len(sensor) == 0:
            col = random.randint(0, self.MAX_INT)%len(self.data.columns)
            sensor = self.data.columns[col]
        idx = random.randint(0, self.MAX_INT)%self.sample_size
        out =  "Null Injection to sensor: "+str(sensor)
        out += "\n                at index: "+str(idx)
        print(out)

        return self.inject(value=None, sensor=sensor, index=idx)

    def reorder(self):
        """
        An Alphabetic Column Sorter

        [What is this?]

        This function reorders sensors by alphanumeric order.
        """

        # Reorder Columns in Alphabetic Order
        self.__check_data_existency()
        sorted_snames = self.sensor_names()
        sorted_snames.sort()
        self.data = self.data[sorted_snames]

    def replace(self, source=[], destination=[], frm=None, to=None):
        """
        A Data Replacement Helper

        [What is this?]

        This function replaces sensor data to destination data
        
        Keyword Arguments:
            source {list} -- [list of sensor names] (default: {[]})
            destination {list} -- [list of destination sensor names] (default: {[]})
            frm {[type]} -- [range parameter] (default: {None})
            to {[type]} -- [range parameter] (default: {None})
        """

        self.__check_data_existency()
        frm, to = self.__check_frm_to(frm,to)
        len_copy_pattern = len(source)
        len_destination = len(destination)

        if len_copy_pattern > len_destination:
            print("Warning: the number of destination sensor(s) is(are) smaller than the number of source sensor(s)!")

        for i in range(len_destination):
            index = len_copy_pattern%i
            self.__assign_new_data(destination[i], frm, to, self.data[source[index]][frm:to])

    def reverse(self, sensors=[], axis=0, frm=None, to=None):
        """
        A Data Reverse Ordering Helper

        [What is this?]

        This function reorders each sensor data in reverse order.
        
        Keyword Arguments:
            sensor {list} -- [list of sensor names] (default: {[]})
            axis {int} -- [0: reversed sample order, 1: reversed column order] (default: {0})
        
        Raises:
            ValueError -- [raise if axis is not 0 or 1]
        
        Returns:
            [pandas.DataFrame] -- [returns changed data table]
        """

        self.__check_data_existency()
        self.__check_sensor(sensors)
        frm, to = self.__check_frm_to(frm, to)
        frm_idx,to_idx = frm, to
        index = list(self.data.index.values)
        frm, to = index[frm], index[to-1]
        sensors = list(sensors)
        if len(sensors) == 0:
            sensors = self.sensor_names()

        if axis == 0:
            if frm_idx == 0 and to_idx == self.sample_size:
                self.data = self.data.iloc[::-1]
            else:
                
                self.data.loc[frm:to,sensors] = self.data.loc[frm:to,sensors][::-1].values
                index_list=self.data.index.to_list()
                index_list[frm_idx:to_idx] = index_list[frm_idx:to_idx][::-1]
                self.data.index = index_list
                
        elif axis == 1:
            sensors.reverse()
            self.data = self.data[sensors]
        else:
            raise ValueError("axis must be either 0 or 1")

        return self.data

    def weighted_sum(self, sensors=[], weight=[], sample_wise=False, save_to=None, frm=None, to=None):
        """
        A Weighted Sum Calculation

        [What is this?]

        This function calculates weighted sums among different sensors and saves to "save_to" sensor.
        
        Keyword Arguments:
            sensors {list} -- [list of source sensor names] (default: {[]})
            weight {list} -- [1D or 2D list of weights for each sensor/sample] (default: {[]})
            sample_wise {bool} -- [specifying if some or all weights are different with respect to each sample] (default: {False})
            save_to {str} -- [destination sensor name] (default: {None})
            frm {int} -- [range parameter] (default: {None})
            to {int} -- [range parameter] (default: {None})
        
        Raises:
            ValueError -- [raise if sample_wise weight has incorrect shape]
            ValueError -- [raise if some weights are missing for some sensors]
        
        Returns:
            [pandas.DataFrame] -- [returns changed data table]
        """

        self.__check_data_existency()
        self.__check_sensor(sensors)
        frm, to = self.__check_frm_to(frm, to)
        weight = np.array(weight)

        if sample_wise:
            if weight.shape[0] != int(to-frm) or weight.shape[1] != len(sensors):
                raise ValueError("shape of weight must be:", frm-to, len(sensors))
        else:
            if len(sensors) != len(weight):
                raise ValueError("all sensors must have its own weight assigned!")

        destination = sensors[0] if save_to is None else save_to
        new_data = np.multiply(np.array(self.data[sensors][frm:to]), weight).sum(axis=1)
        self.__assign_new_data(destination, frm, to, new_data)

        return self.data


    # Scaler
    def normalize(self, option="l2"):
        """
        A Normalizer for Generated Samples

        [What is this?]

        This function normalized each sensor column into [l1, l2, or max] scale.
        For the normalization process, The "sklearn.preprocessing Normalizer" is used.
        
        Keyword Arguments:
            option {str} -- [normalization option] (default: {"l2"})
        
        Returns:
            [pandas.DataFrame] -- [returns changed data table]
        """

        self.__check_data_existency()
        col = self.data.columns
        index = self.data.index.values

        scaler = Normalizer(norm=option)
        new_data = scaler.fit_transform(self.data)
        self.data = pd.DataFrame(data=new_data, index=index, columns=col)

        return self.data

    def standardize(self):
        """
        A Standardizer for Generated Samples

        [What is this?]

        This function standardizes each sensor column into z-score based scaler.
        For the standardization process, The "sklearn.preprocessing StandardScaler" is used.
        """

        self.__check_data_existency()
        col = self.data.columns
        index = self.data.index.values

        scaler = StandardScaler()
        new_data = scaler.fit_transform(self.data)
        self.data = pd.DataFrame(data=new_data, index=index, columns=col)

        return self.data
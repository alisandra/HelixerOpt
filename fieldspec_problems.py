__author__ = 'Alisandra Denton'

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem


### FieldSpec problems ###
class FieldSpecProblem(problem.Problem):
    """Base Problem for playing with FieldSpec reflectance data and auto encoders

    Takes input as folder of csvs with wave_length,sample01_reflectance,sample02_reflectance,... in each row"""

    # location and shape of input data
    @property
    def directory_in(self):
        raise NotImplementedError

    @property
    def number_spectra(self):
        raise NotImplementedError

    @property
    def number_wavelengths(self):
        raise NotImplementedError

    # other stuff / todo
    @property
    def shuffle_me(self):
        return True

    def generate_data(self, data_dir, tmp_dir='/tmp/', task_id=-1):
        pass

    def generate_dataset(self):
        pass

    def dataset_generator(self):
        pass

    def example_reading_spec(self):
        pass

    def preprocess_example(self, example, mode, unused_hparams):
        pass

    def parser(self):
        pass




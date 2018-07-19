__author__ = 'Alisandra Denton'

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem

import numpy as np
import os

### FieldSpec problems ###
class FieldSpecProblem(problem.Problem):
    """Base Problem for playing with FieldSpec reflectance data and auto encoders

    Takes input as folder of csvs with wave_length,sample01_reflectance,sample02_reflectance,... in each row"""

    # location and shape of input data
    @property
    def directory_in(self):
        raise NotImplementedError

    @property
    def number_wavelengths(self):
        raise NotImplementedError

    # other stuff / todo
    @property
    def shuffle_me(self):
        return True

    def generate_data(self, data_dir, tmp_dir='/tmp/', task_id=-1):
        # todo prep file paths, other extras
        for dat_set in ['train', 'dev']:
            files_in = os.listdir(self.directory_in)
            files_in = [x for x in files_in if '_{}_'.format(dat_set) in x]
            data_in = []
            for fil in files_in:
                data_in.append(np.genfromtxt('{}/{}'.format(self.directory_in, fil), delimiter=',', skip_header=1))
            # once all in, check wavelengths match and join
            ready = merge_if_wavelengths_shared(data_in)

            # todo send this to generator to make individual 1D examples

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


def merge_if_wavelengths_shared(data_in):
    """merges numpy arrays in the data_in list by first column"""
    if not isinstance(data_in, list):
        raise ValueError('data_in must be a list of numpy arrays with wavelengths/other key in first column')
    sample_n = sum([x.shape[1] for x in data_in]) - len(data_in)  # bc all wavelength columns will be dropped
    at = 0
    out = np.zeros((data_in[0].shape[0], sample_n))
    for i in range(1, len(data_in), 1):
        assert np.array_equal(data_in[0][:, 0], data_in[i][:, 0])
        out[:, at:(at + data_in[i].shape[1] - 1)] = data_in[i][:, 1:]
    return out



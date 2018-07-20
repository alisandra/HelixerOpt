__author__ = 'Alisandra Denton'

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from genic_problems import serialize_from_numpy, DataExistsError

import tensorflow as tf
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

    # other stuff
    @property
    def shuffle_me(self):
        return True

    @property
    def num_shards(self):
        return 20

    # stuff I should probably but in a parent class for all my problems
    @property
    def num_shards_dev_test(self):
        return 4

    def make_meta(self, data_dir):
        return {'labels_shape': self.label_shape}

    @property
    def label_shape(self):
        return [self.number_wavelengths, 1]

    # data serializing
    def generate_data(self, data_dir, tmp_dir='/tmp/', task_id=-1):
        out_files = {'train': self.training_filepaths(data_dir=data_dir, num_shards=self.num_shards,
                                                      shuffled=not self.shuffle_me),
                     'dev': self.dev_filepaths(data_dir=data_dir, num_shards=self.num_shards_dev_test,
                                               shuffled=not self.shuffle_me)}
        # stop if data already exists
        finished = self.training_filepaths(data_dir=data_dir, num_shards=self.num_shards, shuffled=True)
        finished += self.dev_filepaths(data_dir=data_dir, num_shards=self.num_shards_dev_test, shuffled=True)
        for fin in finished:
            if os.path.exists(fin):
                raise DataExistsError("data already exists at (at least): {}, won't overwrite".format(fin))

        for dat_set in ['train', 'dev']:
            files_in = os.listdir(self.directory_in)
            files_in = [x for x in files_in if '_{}_'.format(dat_set) in x]
            data_in = []
            for fil in files_in:
                data_in.append(np.genfromtxt('{}/{}'.format(self.directory_in, fil), delimiter=',', skip_header=1))
            # once all in, check wavelengths match and join
            reflectance = merge_if_wavelengths_shared(data_in)
            self.generate_dataset(reflectance, out_files[dat_set])

        if self.shuffle_me:
            generator_utils.shuffle_dataset(out_files['train'] + out_files['dev'])

    def generate_dataset(self, reflectance, outfiles):
        generator_utils.generate_files(self.dataset_generator(reflectance), outfiles)

    @staticmethod
    def dataset_generator(reflectance):
        for i in range(reflectance.shape[1]):
            ref_flat = serialize_from_numpy(reflectance[:, i], float)
            yield {'inputs': ref_flat}

    def example_reading_spec(self):
        data_fields = {
            'inputs': tf.FixedLenFeature([self.number_wavelengths], tf.float32),
        }
        return data_fields, None  # idk what the None is a placeholder for in real t2t land

    def preprocess_example(self, example, mode, unused_hparams):
        example['inputs'] = tf.reshape(example['inputs'], [self.number_wavelengths, 1])
        # data is legit 1D this time, so not much to change
        if mode == tf.estimator.ModeKeys.TRAIN:
            # todo, linear transform on amplitude?
            # or perhaps addition of drifting baseline?
            pass
        return example

    def parser(self, serialized_example, mode, meta_info=None):
        if meta_info is None:
            meta_info = {}

        data_fields, _ = self.example_reading_spec()
        features = tf.parse_single_example(serialized_example,
                                           features=data_fields)
        example = self.preprocess_example(features, mode, meta_info)
        features = example['inputs']
        try:  # not necessarily there as potentially have a lot of unlabelled data
            labels = example['targets']
        except KeyError:
            labels = example['inputs']
        return features, labels


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



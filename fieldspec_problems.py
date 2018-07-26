__author__ = 'Alisandra Denton'

from tensor2tensor.data_generators import generator_utils
from mod_problem import ModProblem, serialize_from_numpy
from tensorflow.python.ops import parsing_ops
#from tensorflow.python.estimator.export import Serial
import tensorflow as tf
import numpy as np
import pandas as pd
import os


### FieldSpec problems ###
class FieldSpecProblem(ModProblem):
    """Base Problem for playing with FieldSpec reflectance data and auto encoders

    Takes input as folder of csvs with wave_length,sample01_reflectance,sample02_reflectance,... in each row"""

    # location and shape of input data
    @property
    def directory_in(self):
        raise NotImplementedError

    @property
    def number_wavelengths(self):
        raise NotImplementedError

    @property
    def num_shards(self):
        return 20

    @property
    def num_shards_dev_test(self):
        return 4

    @property
    def label_shape(self):
        return [self.number_wavelengths, 1]

    # data serializing
    def generate_data(self, data_dir, tmp_dir='/tmp/', task_id=-1):
        out_files = self.setup_file_paths(data_dir)

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

    def serving_input_fn_mod(self): # w/o 'mod' exists but seems to be a) completely generic, and b) need parameter
        """for serving export, also  (AKA?) for setting up hub export"""
        def serving_input_fn():
            serialized_eg = tf.placeholder(tf.string, [None], 'tensor_in')
            receiver_tensors = {'inputs': serialized_eg}
            feature_spec, _ = self.example_reading_spec()
            features = parsing_ops.parse_example(serialized_eg, feature_spec)
            #features = self.parser(features, tf.estimator.ModeKeys.PREDICT)
            return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)
            #return self.serving_input_fn(None)

        return serving_input_fn

    def serving_input_receiver_fn(self):
        feature_spec, _ = self.example_reading_spec()
        return tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)()



class FieldSpecLabelledProblem(ModProblem):
    """Problem for playing with FieldSpec reflectance data when it has labels

    Takes input as folder of csvs with all labels to the left, of all reflectance data
    The column 'sets' should contain xval/test/train designations"""

    # location and shape of input data
    @property
    def file_in(self):
        raise NotImplementedError

    @property
    def number_labels(self):
        raise NotImplementedError

    def get_label_columns(self, df):
        # this should be code to get from a pandas dataframe the actual columns for the labels of interest
        raise NotImplementedError

    def get_one_label(self, labels, i):
        raise NotImplementedError  # todo, tmp hack to get around subsetting in changing dimensionality

    @property
    def number_wavelengths(self):
        raise NotImplementedError

    @property
    def num_shards(self):
        return 20

    @property
    def num_shards_dev_test(self):
        return 4

    @property
    def label_shape(self):
        return [self.number_labels]

    # data serializing
    def generate_data(self, data_dir, tmp_dir='/tmp/', task_id=-1):
        out_files = self.setup_file_paths(data_dir)
        name_fix = {'xval': 'dev', 'test': 'test', 'train': 'train'}

        df = pd.read_csv(self.file_in)

        for dat_set in name_fix.keys():
            targ_df = df[df['sets'] == dat_set]
            x_df = targ_df.iloc[:, (targ_df.shape[1] - self.number_wavelengths):]
            x_df = np.array(x_df)
            y_df = self.get_label_columns(targ_df)
            self.generate_dataset(reflectance=x_df, labels=y_df, outfiles=out_files[name_fix[dat_set]])
        if self.shuffle_me:
            generator_utils.shuffle_dataset(out_files['train'] + out_files['dev'])

    def generate_dataset(self, reflectance, labels, outfiles):
        generator_utils.generate_files(self.dataset_generator(reflectance, labels), outfiles)

    def dataset_generator(self, reflectance, labels):
        for i in range(reflectance.shape[0]):
            ref_flat = serialize_from_numpy(reflectance[i, :], float)
            lab_flat = serialize_from_numpy(self.get_one_label(labels, i), float)
            yield {'inputs': ref_flat, 'targets': lab_flat}

    def example_reading_spec(self):
        data_fields = {
            'inputs': tf.FixedLenFeature([self.number_wavelengths], tf.float32),
            'targets': tf.FixedLenFeature([self.number_labels], tf.float32)
        }
        return data_fields, None  # idk what the None is a placeholder for in real t2t land

    def preprocess_example(self, example, mode, unused_hparams):
        #example['inputs'] = tf.reshape(example['inputs'], [self.number_wavelengths, 1])
        # target data is legit 1D this time, so not much to change
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
        labels = example['targets']
        return features, labels

    # todo, needs it's own import
    # todo, an easy way to make subclasses to indicate from same file, which label to use

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



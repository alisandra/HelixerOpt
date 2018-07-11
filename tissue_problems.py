__author__ = 'Alisandra Denton'

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators.gene_expression import generate_shard_args

from builtins import range

import tensorflow as tf
import numpy as np

import random
import csv
import os
import json
import re
import subprocess
import itertools
import time
import multiprocessing as mp

from dna_helpers import seq_to_row, row_to_array


### gene expression problems ###
class TissueExpressionProblem(problem.Problem):
    """Base Problem for tissue specific expression. That is: Gene Exp in formats that are possibly produced"""
    TRAIN = 'train'
    TEST = 'test'
    XVAL = 'val'

    SETS = [TRAIN, TEST, XVAL]

    # todo, boot these to subclass? NotImplementedError
    # todo, symmetry between len_x, and len_y and other naming would be nice! x_x
    @property
    def len_x(self):
        return 4096 * 4

    @property
    def middle_x(self):
        # set this lower to take a (centered) subset of each x example
        return self.len_x

    @property
    def len_y(self):
        return 128

    @property
    def middle_y(self):
        # set this lower to take a (centered)  subset of each y example
        return self.len_y

    @property
    def number_y(self):
        return 30

    @property
    def label_shape(self):
        return [self.middle_y, self.number_y]

    @property
    def directory_in(self):
        raise NotImplementedError

    def y_info_file(self, set_name):
        return '{}_Yinfos_{}.csv'.format(self.name, set_name)

    @property
    def shuffle_me(self):
        return True

    @property
    def x_pfx(self):
        x_files = os.listdir(self.x_direc)
        x_file_pfx = os.path.commonprefix(x_files)
        return self.x_direc + x_file_pfx

    @property
    def x_direc(self):
        return self.directory_in + '/x/'

    @property
    def y_direc(self):
        return self.directory_in + '/y/'

    @property
    def num_shards(self):
        return 100

    @property
    def num_shards_dev_test(self):
        return 10

    @property
    def target_log1p(self):
        return False

    @property
    def target_subtract_mean(self):
        return False

    @property
    def target_divide_sd(self):
        return False

    def y_stats_pfx(self, set_name, data_dir):
        # note, this used to be .format(self.directory_in, ...) won't be compatible with old locations
        return '{}/{}_{}'.format(data_dir, self.name, set_name)

    def generate_data(self, data_dir, tmp_dir='/tmp/', task_id=-1):

        # track for later
        paths_to_shuffle = []
        processes = []
        stat_files = []

        datasets = {}  # still awkward... todo: does t2t's problem not have an inherrent train/test/dev mode???
        filepaths = [self.training_filepaths, self.test_filepaths, self.dev_filepaths]  # same sort order as SETS
        for key in TissueExpressionProblem.SETS:
            datasets[key] = [filepaths.pop(0), self.num_shards_dev_test, key]

        datasets[TissueExpressionProblem.TRAIN][1] = self.num_shards  # more files dedicated to training
        # similarity between prefix and setname to conform more common 'val' for x-validation to 'dev' for t2t standards
        datasets[TissueExpressionProblem.XVAL][2] = 'dev'  # compensate for previous inconsistency in naming TODO fix

        shuffled = not self.shuffle_me  # because if we don't want to shuffle it, we'll assume it is 'shuffled' (enough)

        for key in TissueExpressionProblem.SETS:
            num_examples = system_wc('{}{}.csv'.format(self.x_pfx, key))
            file_namer, num_shards, set_name = datasets[key]

            outfiles = file_namer(data_dir, num_shards, shuffled=shuffled)
            paths_to_shuffle += outfiles
            for start, end, outfile in generate_shard_args(
                    outfiles, num_examples):
                new_stat_file = '{}_from{}.json'.format(self.y_stats_pfx(set_name, data_dir), start)
                p = mp.Process(target=self.generate_dataset,
                               args=(key + '.csv', [outfile], start, end, new_stat_file))
                processes.append(p)
                stat_files.append(new_stat_file)

        # All the multi threading
        threads = 8
        # start a set of processes
        running = []
        for i in range(min(threads, len(processes))):
            p = processes.pop()
            p.start()
            running.append(p)

        while running:
            time.sleep(0.5)
            # rm finished
            for i in range(len(running) - 1, -1, -1):
                if not running[i].is_alive():
                    running.pop(i)
            # add more
            if processes:
                try:
                    for i in range(threads - len(running)):
                        p = processes.pop()
                        p.start()
                        running.append(p)
                except IndexError:
                    continue
        # Shuffle
        if self.shuffle_me:
            generator_utils.shuffle_dataset(paths_to_shuffle)

        # combine all y_stats files into one per set
        # could simply delete non-train, but also might help catch a non-representative training set
        for set_name in [self.TRAIN, 'dev', self.TEST]:
            set_pfx = self.y_stats_pfx(set_name, data_dir)
            targ_files = [x for x in stat_files if set_pfx in x]
            y_tracker = Y_normalizer()
            y_log_tracker = Y_normalizer()
            for tf in targ_files:
                with open(tf) as f:
                    sub_infos = json.load(f)
                y_tracker.add_subset_from_dict(sub_infos['raw'])
                y_log_tracker.add_subset_from_dict(sub_infos['log1p'])
                os.remove(tf)
            y_stats = {
                'raw': y_tracker.final_export(),
                'log1p': y_log_tracker.final_export()
            }
            fileout = '{}_stats.json'.format(self.y_stats_pfx(set_name, data_dir))
            with open(fileout, 'w') as f:
                json.dump(y_stats, f)

    def example_reading_spec(self):
        data_fields = {
            "targets": tf.FixedLenFeature([self.number_y * self.middle_y], tf.float32),
            "inputs": tf.FixedLenFeature([self.middle_x], tf.float32),
        }
        return data_fields, None

    def preprocess_example(self, example, mode, unused_hparams):

        # restore original shapes
        example["targets"] = tf.reshape(example["targets"],
                                        [self.number_y, self.middle_y])
        example["targets"] = tf.transpose(example["targets"])

        example["inputs"] = tf.reshape(example["inputs"], (self.middle_x // 4, 4))

        # adjust training examples to have artificially more
        if mode == tf.estimator.ModeKeys.TRAIN:
            example = self.maybe_reverse_complement(example)
            # todo, mask input regions in train mode
            #example = self.maybe_0_out(example)

        if self.target_log1p:
            meta = unused_hparams['log1p']
            example['targets'] = tf.log1p(example['targets'])
        else:
            meta = unused_hparams['raw']
        if self.target_subtract_mean:
            example['targets'] = tf.subtract(example['targets'], meta['mean'])
        if self.target_divide_sd:
            example['targets'] = tf.divide(example['targets'], meta['sd'])

        return example

    @staticmethod
    def maybe_reverse_complement(example, prob=0.5):
        print('expecting (length, # samples), e.g. (128, 30): {}'.format(example['targets'].get_shape()))
        if random.random() <= prob:
            # three dimensions, for reverse, complement, and anyways length of 1 (so doesn't matter)
            # todo, please test this!
            example['inputs'] = tf.reverse(example['inputs'], [0, 1])
            example['targets'] = tf.reverse(example['targets'], [0])

        return example

    @staticmethod
    def maybe_0_out(example, prob=0.6, length=32):
        while random.random() <= prob:
            start_at = random.choice(list(range(example['inputs'].get_shape()[0])))
            example['inputs'][start_at:(start_at + length)] = 0
            # todo, please test this!
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

    def make_meta(self, data_dir):
        f_path = self.y_stats_pfx('train', data_dir) + '_stats.json'
        with open(f_path) as f:
            out = json.load(f)
        return out

    def generate_dataset(self,
                         set_ending,
                         outfiles,
                         start_idx=None,
                         end_idx=None,
                         stat_file=None):

        generator_utils.generate_files(
            self.dataset_generator(set_ending, start_idx, end_idx, stat_file),
            outfiles)

    def dataset_generator(self,
                          set_ending,
                          start_idx=None,
                          end_idx=None,
                          stat_file=None):
        y_direc = self.y_direc
        x_pfx = self.x_pfx
        middle_y = self.middle_y
        number_y = self.number_y
        len_x = self.len_x
        len_y = self.len_y
        middle_x = self.middle_x
        # subset x coordinates
        buffer_x = (len_x - middle_x) // 2
        assert buffer_x % 2 == 0  # this must be an even number
        end_x = buffer_x + middle_x
        # subset y coordinates
        buffer_y = (len_y - middle_y) // 2
        assert buffer_y % 2 == 0  # this too must be an even number
        end_y = buffer_y + middle_y
        print('end_y {}'.format(end_y))
        print('buffer_y {}'.format(buffer_y))
        # x and y are already encoded as such
        yfiles = os.listdir(y_direc)
        yfiles = [x for x in yfiles if set_ending in x]
        x_handle = open(x_pfx + set_ending)
        x_reader = shard_csv_reader(x_handle, start_idx, end_idx)
        #x_reader = open_csv_reader(full_path=x_pfx + set_ending)
        y_handles = [open(y_direc + '/' + x) for x in yfiles]
        y_readers = [shard_csv_reader(x, start_idx, end_idx) for x in y_handles]
        #y_readers = [open_csv_reader_2files(directory=y_direc, file_name=x) for x in yfiles]

        # tracker to ultimately allow calculation of mean & sd across all data
        y_tracker = Y_normalizer()
        y_log_tracker = Y_normalizer()

        for row in x_reader:
            # setup array to hold y results (with -1 to notice failure)
            # prep y
            y_array = np.full((len(y_readers), middle_y), -1, dtype=np.float64)
            for j in range(len(y_readers)):
                y_array[j, :] = row_to_array(next(y_readers[j]))[buffer_y:end_y]
                if any(np.isinf(y_array[j, :])):
                    print("-----Infinity at or around-----")
                    print(y_array[j, :])
            y_width = y_array.shape[1]
            y_height = y_array.shape[0]
            # prep x
            row[0] = self.mod_unprocessed_x(row[0], y_array)
            xrow = seq_to_row(row[0], spread_evenly=False)  # x data to array
            xrow = xrow.flatten()[buffer_x:end_x]
            x = [float(w) for w in xrow]  # TFRecords need explicit / unfailing data types
            ex_dict = {
                'inputs': x,
                'x_width': [4],
                'x_height': [middle_x // 4],
                'targets': list(y_array.flatten()),
                'y_width': [y_width],
                'y_height': [y_height]
            }
            y_tracker.add_data(y_array)
            y_log_tracker.add_data(np.log1p(y_array))
            yield ex_dict
        # check (occasionally, once should do) that sizes are as expected
        # y_width should throw an error above on replacement sizes
        assert y_height == number_y
        # total x needs checking
        assert middle_x == len(x)

        x_handle.close()
        [y_handle.close() for y_handle in y_handles]
        # export mean, sum squared differences, and length for both y, and log(y + 1)
        # so that y can be transformed to z-space readily for training (and to z-space and back for predict)
        y_stats = {
            'raw': y_tracker.export_subset(),
            'log1p': y_log_tracker.export_subset()
        }
        fileout = stat_file  # '{}_from{}.json'.format(y_stats_pfx, start_idx)
        with open(fileout, 'w') as f:
            json.dump(y_stats, f)

    def mod_unprocessed_x(self, x, y_array):
        return x


class ScrambledExpressionProblem(TissueExpressionProblem):
    def preprocess_example(self, example, mode, unused_hparams):
        example = super(ScrambledExpressionProblem, self).preprocess_example(example, mode, unused_hparams)
        e = tf.transpose(example['inputs'], perm=[1, 0, 2])
        e = tf.random_shuffle(e)
        e = tf.transpose(e, perm=[1, 0, 2])
        example['inputs'] = e
        return example


class SpikedExpressionProblem(TissueExpressionProblem):
    def __init__(self, was_reversed=False, was_copy=False):
        super(SpikedExpressionProblem, self).__init__(was_reversed=was_reversed, was_copy=was_copy)
        self._motifs = None

    @property
    def motif_length(self):
        return 32

    @property
    def pfx_seed(self):
        # final seed will be `self.pfx_seed + self.len_y`, for deterministically random motif selection
        return 'puma'

    @property
    def motifs(self):
        if self._motifs is None:
            random.seed('{}{}'.format(self.pfx_seed, self.len_y))
            motifs = []
            for _ in range(self.number_y):
                motifs.append(SpikedExpressionProblem.make_motif(self.motif_length))
            self._motifs = motifs
            for i in range(len(self.motifs)):
                print(motifs[i])
        return self._motifs

    @staticmethod
    def make_motif(length):
        bps = ['C', 'A', 'T', 'G']
        out = ''
        for _ in range(length):
            out += random.choice(bps)
        return out

    def motifs_to_add(self, y_array):
        # np.full((len(y_readers), middle_y), -1, dtype=np.float64)
        # (number samples) x (length each sample)
        mean_coverage = np.mean(y_array, axis=1)
        log_cov = np.log2(mean_coverage + 1)
        n_to_add = log_cov.astype(np.int)
        assert len(n_to_add) == self.number_y
        motifs = self.motifs
        motifs_to_add = []
        for i in range(len(n_to_add)):
            motifs_to_add += [motifs[i]] * n_to_add[i]
        return motifs_to_add

    def spike_sequence(self, sequence, y_array):
        motifs_to_add = self.motifs_to_add(y_array)
        for motif in motifs_to_add:
            i = random.randint(0, len(sequence))
            sequence = self.overwrite_substring(sequence, motif, i)
        return sequence

    @staticmethod
    def overwrite_substring(stringy, newsub, start_at):
        n = len(stringy)
        out = stringy[:start_at] + newsub
        out += stringy[start_at + len(newsub):]
        out = out[:n]
        return out

    def mod_unprocessed_x(self, x, y_array):
        out = self.spike_sequence(x, y_array)
        return out


def system_wc(filepath, par='-l'):
    raw_out = subprocess.check_output(['wc', par, filepath])
    out = raw_out.decode()
    out = re.sub(' .*\n', '', out)
    out = int(out)
    return out


def shard_csv_reader(file_handle, start_idx, end_idx):
    reader = csv.reader(file_handle)
    for line in itertools.islice(reader, start_idx, end_idx):
        yield line


class Y_normalizer:
    def __init__(self):
        self.sum_sq_diffs = float(0)
        self.len_total = int(0)
        self.sum = float(0)

    # for adding raw data
    def add_data(self, y_array):
        add_sum = np.sum(y_array)
        if np.isinf(add_sum):
            print('largest: {}, min {}, mean {}'.format(np.max(y_array), np.min(y_array), np.mean(y_array)))
            raise ValueError('Cannot handle inf here')
        if (y_array < 0).any():
            print('largest: {}, min {}, mean {}'.format(np.max(y_array), np.min(y_array), np.mean(y_array)))
            raise ValueError('How did you get negative coverage anyways?')
        self.sum += float(np.sum(y_array))
        self.sum_sq_diffs += float(self.sum_squared_differences(y_array))
        self.len_total += int(np.prod(y_array.shape))

    def export_subset(self):
        return {'sum': float(self.sum),
                'sum_sq_diffs': float(self.sum_sq_diffs),
                'n': int(self.len_total)}

    # for combining data from e.g. files or threads
    def add_subset(self, partial_sum, sum_sq_diffs, n):
        self.sum += float(partial_sum)
        self.sum_sq_diffs += float(sum_sq_diffs)
        self.len_total += int(n)

    def add_subset_from_dict(self, dct):
        partial_sum = dct['sum']
        ssds = dct['sum_sq_diffs']
        n = dct['n']
        self.add_subset(partial_sum, ssds, n)

    def final_export(self):
        try:
            mean = self.sum / self.len_total
        except ZeroDivisionError:
            mean = None
        return {
            'mean': mean,
            'sd': np.sqrt(self.sum_sq_diffs) / (self.len_total - 1)
        }

    @staticmethod
    def sum_squared_differences(np_data):
        return np.sum((np_data - np.mean(np_data)) ** 2)




__author__ = 'Alisandra Denton'

import os
from tensor2tensor.data_generators import problem


class ModProblem(problem.Problem):
    """Base Problem for all the custom problems started here with controller.py"""

    @property
    def shuffle_me(self):
        return True

    @property
    def num_shards(self):
        return 100

    @property
    def num_shards_dev_test(self):
        return 10

    def make_meta(self, data_dir):
        return {'labels_shape': self.label_shape}

    @property
    def label_shape(self):
        return NotImplementedError

    def setup_file_paths(self, data_dir):
        out_files = {'train': self.training_filepaths(data_dir=data_dir, num_shards=self.num_shards,
                                                      shuffled=not self.shuffle_me),
                     'dev': self.dev_filepaths(data_dir=data_dir, num_shards=self.num_shards_dev_test,
                                               shuffled=not self.shuffle_me),
                     'test': self.test_filepaths(data_dir=data_dir, num_shards=self.num_shards_dev_test,
                                                 shuffled=not self.shuffle_me)}
        # stop if data already exists
        finished = self.training_filepaths(data_dir=data_dir, num_shards=self.num_shards, shuffled=True)
        finished += self.dev_filepaths(data_dir=data_dir, num_shards=self.num_shards_dev_test, shuffled=True)
        finished += self.test_filepaths(data_dir=data_dir, num_shards=self.num_shards_dev_test, shuffled=True)
        for fin in finished:
            if os.path.exists(fin):
                raise DataExistsError("data already exists at (at least): {}, won't overwrite".format(fin))
        return out_files


class DataExistsError(Exception):
    pass


def serialize_from_numpy(a_numpy_array, fn=float):
    x = a_numpy_array.flatten()
    x = list(x)
    x = [fn(w) for w in x]
    return x

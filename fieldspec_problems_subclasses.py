import numpy as np
from fieldspec_problems import FieldSpecProblem, FieldSpecLabelledProblem
from tensor2tensor.utils import registry

@registry.register_problem
class FieldSpecProblemTest(FieldSpecProblem):

    @property
    def directory_in(self):
        return 'test'

    @property
    def number_wavelengths(self):
        return 2151


@registry.register_problem
class FieldSpecProblemLabelledTest(FieldSpecLabelledProblem):
    @property
    def file_in(self):
        return 'labelled.csv'

    @property
    def number_wavelengths(self):
        return 2151

    @property
    def number_labels(self):
        return 1

    def get_label_columns(self, df):
        labels = np.array(df['Condition'])
        labels[labels == 'control'] = 0
        labels[labels == 'Drought'] = 1
        labels = labels.reshape((labels.shape[0], 1))
        return labels

    def get_one_label(self, labels, i):
        return labels[i]
__author__ = 'Alisandra Denton'

from tissue_problems import TissueExpressionProblem, SpikedExpressionProblem, ScrambledExpressionProblem
from tensor2tensor.utils import registry


@registry.register_problem
class TissueExpressionAt01(TissueExpressionProblem):
    @property
    def directory_in(self):
        return '/a/directory/with/coverage/data/for/30/tissues'

    @property
    def number_y(self):
        return 30


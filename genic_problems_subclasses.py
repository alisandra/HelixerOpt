__author__ = 'Alisandra Denton'

from tensor2tensor.utils import registry

from genic_problems import GeneCallingProblem

@registry.register_problem
class GeneCallingProblemTest(GeneCallingProblem):
    @property
    def directory_in(self):
        return 'test/arabidopsis'

    @property
    def gff_ending(self):
        return '.gff3'


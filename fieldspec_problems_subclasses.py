from fieldspec_problems import FieldSpecProblem
from tensor2tensor.utils import registry

@registry.register_problem
class FieldSpecProblemTest(FieldSpecProblem):

    @property
    def directory_in(self):
        return 'test'

    @property
    def number_wavelengths(self):
        return 2151


from fieldspec_problems import FieldSpecProblem


class FieldSpecProblemTest(FieldSpecProblem):

    @property
    def directory_in(self):
        return 'test'

    @property
    def number_wavelengths(self):
        return 2151


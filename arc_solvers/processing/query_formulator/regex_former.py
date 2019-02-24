from arc_solvers.processing.query_formulator.formulator import Formulator


class RegexFormulator(Formulator):

    def formulate(self, sample, **kwargs):
        super().does_arg_exists('rule', **kwargs)
        # TODO: check weather the kwargs has rules apply rules based on that
        sample['question']['stem'] = "hehehehehe!!"
        return sample

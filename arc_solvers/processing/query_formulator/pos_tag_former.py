from arc_solvers.processing.query_formulator.formulator import Formulator


class POSTagFormulator(Formulator):

    def formulate(self, sample, **kwargs):
        super().does_arg_exists('rule', **kwargs)
        # have to lookup how to do this
        return sample

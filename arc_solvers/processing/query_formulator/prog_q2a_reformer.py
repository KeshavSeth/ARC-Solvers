from arc_solvers.processing.convert_to_entailment import *
from arc_solvers.processing.query_formulator.formulator import Formulator


class ProgrammaticalQ2AReformer(Formulator):

    def formulate(self, sample, **kwargs):
        fitb = get_fitb_from_question(sample['question']['stem'])
        choices = sample['question']['choices']
        for idx in range(len(choices)):
            choices[idx]['hypothesis'] = create_hypothesis(fitb, choices[
                idx]['text'])
        return sample

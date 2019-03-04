"""
This class serves is one stop for all reformulations.
We can write our own new query former like RegexFormulator and register it 
manually in method __register_reformer

While using this class, we just need to pass the formulation type.
For example, if formulation type = 'regex'
the form_query will automatically formulate query using RegexFormulator.
Note: while using with notebook, if you add a new formulator class, you have 
restart notebooks kernel.

The correctness of rule passed is user dependent. However, we can add basic 
check in specific formulators.

Also, for python notebook rules will come from the user. However, when we 
start using these formulations on end-to-end basis, we would need to add a 
Rule supplier (a file which give formulator all the rules)

JSONL format of files
 input
 {
    "id":"Mercury_SC_415702",
    "question": {
       "stem":"George wants to warm his hands quickly by rubbing them. Which skin surface will
               produce the most heat?",
       "choices":[
                  {"text":"dry palms","label":"A"},
                  {"text":"wet palms","label":"B"},
                  {"text":"palms covered with oil","label":"C"},
                  {"text":"palms covered with lotion","label":"D"}
                 ]
    },
    "answerKey":"A"
  }
  
  
  output
  {
   "id": "Mercury_SC_415702",
   "question": {
      "stem": "..."
      "choice": {"text": "dry palms", "label": "A"},
      "new_query": "..."
      "support": {
                "text": "...",
                "type": "sentence",
                "ir_pos": 0,
                "ir_score": 2.2,
            }
    },
    "answerKey":"A"
  }
  ...
  {
   "id": "Mercury_SC_415702",
   "question": {
      "stem": "..."
      "choice": {"text":"palms covered with lotion","label":"D"}
      "new_query": "..."
      "support": {
                "text": "...",
                "type": "sentence",
                "ir_pos": 1,
                "ir_score": 1.8,
            }
     "answerKey":"A"
  }
  
  Every formulator will add a new_query related to the choice at the same 
  scope level
"""
import json
import os
import sys
from typing import List, Dict

from allennlp.common.util import JsonDict
from tqdm._tqdm import tqdm

from arc_solvers.processing.add_retrieved_text import filter_hits_across_choices, create_output_dict
from arc_solvers.processing.es_search import EsSearch, EsHit
from arc_solvers.processing.query_formulator.pos_tag_former import \
    POSTagFormulator
from arc_solvers.processing.query_formulator.prog_q2a_reformer import \
    ProgrammaticalQ2AReformer
from arc_solvers.processing.query_formulator.regex_former import RegexFormulator


MAX_HITS = 8 
es_search = EsSearch(es_client="node008", max_hits_per_choice=MAX_HITS, max_hits_retrieved=100)


class ReformulatorBase:

    @staticmethod
    def __register_reformers():
        return {
            'regex': RegexFormulator(),
            'prog_q2a': ProgrammaticalQ2AReformer(),
            'pos-tag': POSTagFormulator()
        }

    def __init__(self):
        self.reformers = self.__register_reformers()

    def form_query(self, sample, formulation_type, rule=None):
        ''' Return format:
        {
            "id": "Mercury_7175875"
            "question": {
                "stem": "...", 
                "choices": [
                    {
                        "text": 'Planetary density will decrease.', 
                        'label': 'A', 
                        'new_query': '...'
                    }, 
                    ..., 
                    {
                        'text': 'Planetary gravity will become stronger.', 
                        'label': 'D', 
                        'new_query': '...'
                    }
                ]
            }
            "answerKey": "C"
        }
        '''
        if formulation_type not in self.reformers.keys():
            raise ValueError(f"Formulation type : {formulation_type} is not "
                             f"currently supported!!\n Supported types are : "
                             f"{self.reformers.keys()}")
        return self.reformers.get(formulation_type).formulate(sample, rule=rule)

    def reformulate_query(self, qa_file, output_file):
        with open(output_file, 'w') as reform_qa, open(qa_file, 'r') as origin_qa:
            print("Writing to {} from {}".format(output_file, qa_file))
            line_tqdm = tqdm(origin_qa, dynamic_ncols=True)
            for line in line_tqdm:
                json_line = json.loads(line)
                num_reform = 0
                for output_dict in self.reform_query_to_qajson(json_line):
                    reform_qa.write(json.dumps(output_dict) + "\n")
                    num_reform += 1
                line_tqdm.set_postfix(hits=num_reform)

    def reform_query_to_qajson(self, qa_json: JsonDict):
        rule = [
            '[wW]hat.*?','[hH]ow.*?'
            ]
        new_qa = self.form_query(qa_json, formulation_type='prog_q2a', rule=rule)
        hits_per_choice = es_search.get_hits_for_question(None, new_qa["question"]["choices"])
        filter_hits_across_choices(hits_per_choice, MAX_HITS)
        output_dicts_per_question = []
        for choice in new_qa["question"]["choices"]:
            choice_text = choice["text"]
            hits = hits_per_choice[choice_text]
            for hit in hits:
                output_dict_per_hit = create_output_dict(new_qa, choice, hit)
                output_dicts_per_question.append(output_dict_per_hit)
        return output_dicts_per_question


def count(filepath):
    import re
    wh_count = 0
    with open(filepath, 'r') as f:
        for idx, line in enumerate(f):
            json_line = json.loads(line)
            question_text = json_line["question"]["stem"]

            wh_words = ["which", "what", "where", "when", "how", "who", "why"]
            for wh in wh_words:
                m = re.search(wh + "\?[^\.]*[\. ]*$", question_text.lower())
                if m:
                    wh_count += 1
                    break
                else: # Otherwise, find the wh-word in the last sentence
                    m = re.search(wh + "[ ,][^\.]*[\. ]*$", question_text.lower())
                    if m:
                        wh_count += 1

    print(f'questions with wh_words/all questions: {wh_count}/{idx+1}')
    
if __name__ == "__main__":
    QA_FILE_PATH = "data/ARC-V1-Feb2018/ARC-Challenge/ARC-Challenge-Test.jsonl"
    output_filepath = "data/ARC-V1-Feb2018/ARC-Challenge/ARC-Challenge-Test_with_hits_reformqa.jsonl"

#    reformulator = ReformulatorBase()
#    reformulator.reformulate_query(QA_FILE_PATH, output_filepath)
    count(QA_FILE_PATH)
    



from arc_solvers.processing.query_formulator.pos_tag_former import \
    POSTagFormulator
from arc_solvers.processing.query_formulator.prog_q2a_reformer import \
    ProgrammaticalQ2AReformer
from arc_solvers.processing.query_formulator.regex_former import RegexFormulator

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
      "hypothesis": {}
    },
     "answerKey":"A"
  }
  ...
  {
   "id": "Mercury_SC_415702",
   "question": {
      "stem": "..."
      "choice": {"text":"palms covered with lotion","label":"D"}
      "hypothesis": {}
     "answerKey":"A"
  }
  
  Every formulator will add a hypothesis related to the choice at the same 
  scope level
"""
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
        if formulation_type not in self.reformers.keys():
            raise ValueError(f"Formulation type : {formulation_type} is not "
                             f"currently supported!!\n Supported types are : "
                             f"{self.reformers.keys()}")
        return self.reformers.get(formulation_type).formulate(sample, rule=rule)

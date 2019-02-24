import copy
from random import randint as ri

from arc_solvers.processing.add_retrieved_text import *
from arc_solvers.processing.query_formulator.formulator_base import \
    ReformulatorBase

es_search = EsSearch(max_hits_per_choice=8, max_hits_retrieved=100)
formBase = ReformulatorBase()


def qa_to_jsonl(qa_file):
    qa_file = os.getcwd() + "/../../../" + qa_file
    json_dataset = []
    with open(qa_file, 'r') as qa_file:
        for line in qa_file:
            json_dataset.append(json.loads(line))
    return json_dataset


def get_sample(jsonl, id=None):
    if id is None:
        return copy.deepcopy(jsonl[ri(0, len(jsonl))])
    for sample in jsonl:
        if sample["id"] == id:
            # We need deepcopy in here as we will change sample afterwards.
            return copy.deepcopy(sample)
    raise ValueError(f"Could not find the id : {id}")


def get_reformulated_sample(sample, formulation_type, rule):
    return formBase.form_query(sample, formulation_type, rule)


# this function is too similar to add_hit_to_qajson in add_retrieved_text
# however for only the purpose of notebook duplication is okay. Need to write
# something clean if go forward with symbolic approach
# this is just hacky
def get_support_text(sample, label='A'):
    sample['question']['choices'] = [choice for choice in sample['question'][
        'choices'] if choice['label'] == label]
    hits_per_choice = es_search.get_hits_for_question(
        sample['question']['stem'], sample['question']['choices'])
    output_dicts_per_question = []
    filter_hits_across_choices(hits_per_choice, MAX_HITS)
    for choice in sample["question"]["choices"]:
        choice_text = choice["text"]
        hits = hits_per_choice[choice_text]
        for hit in hits:
            output_dict_per_hit = create_output_dict(sample, choice, hit)
            output_dicts_per_question.append(output_dict_per_hit)
    return output_dicts_per_question


def print_supporting_text(es_result):
    for res in es_result:
        support = res['question']['support']
        print(f"Score : {support['ir_score']} | Text : {support['text']}")


def print_sample(sample):
    # print(sample)
    # print("============================================================")
    print(f"id : {sample['id']}")
    print("")
    print("Question")
    print(sample['question']['stem'])
    print("")
    print("Choices")
    for choice in sample['question']['choices']:
        print(f"{choice['label']} : {choice['text']} ")
    print("")
    print(f"Correct option : {sample['answerKey']}")

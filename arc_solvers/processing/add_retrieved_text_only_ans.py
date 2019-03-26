"""
# This code is modified from add_retrieved_text.py for the purpose of building training data for multi-step retriever write.
# Only retrieve support texts for the correct answer.

# Change: 
#   EsSearch: 
#     1. change hits_length limit of character to limit of words
#     2. add min_hits_length
#

Script to retrieve HITS for each answer choice and question
USAGE:
 python scripts/add_retrieved_text.py qa_file output_file

JSONL format of files
 1. qa_file:
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

 2. output_file:
  {
   "id": "Mercury_SC_415702",
   "question": {
      "stem": "..."
      "choice": {"text": "dry palms", "label": "A"},
      "support": {
                "text": "...",
                "type": "sentence",
                "ir_pos": 0,
                "ir_score": 2.2,
            }
    },
     "answerKey":"A"
  }
"""

import json
import os
import sys
import argparse
from typing import List, Dict

from allennlp.common.util import JsonDict
from tqdm._tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))))
from arc_solvers.processing.es_search import EsSearch, EsHit

def add_retrieved_text(args):
    es_search = EsSearch(
                    es_client = "node008", 
                    max_hits_retrieved = args.num_retrieve*2,
                    min_hit_length = 5,
                    max_hit_length = 100,
                    max_hits_per_choice = args.num_retrieve
                )

    with open(args.output_file, 'w') as output_handle, open(args.input_file, 'r') as qa_handle:
        print("Writing to {} from {}".format(args.output_file, args.input_file))
        line_tqdm = tqdm(qa_handle, dynamic_ncols=True)
        for line in line_tqdm:
            json_line = json.loads(line)
            num_hits = 0
            for output_dict in add_hits_to_qajson(es_search, json_line, args.num_retrieve):
                output_handle.write(json.dumps(output_dict) + "\n")
                num_hits += 1
            line_tqdm.set_postfix(hits=num_hits)


def add_hits_to_qajson(es_search, qa_json: JsonDict, num_retrieve: int):
    question_text = qa_json["question"]["stem"]
    #choices = [choice["text"] for choice in qa_json["question"]["choices"]]
    choices = [choice["text"] for choice in qa_json["question"]["choices"] if choice["label"] == qa_json["answerKey"]]
    hits_per_choice = es_search.get_hits_for_question(question_text, choices)
    output_dicts_per_question = []
    filter_hits_across_choices(hits_per_choice, num_retrieve)
    for choice in qa_json["question"]["choices"]:
        if choice["label"] != qa_json["answerKey"]: 
            continue
        choice_text = choice["text"]
        hits = hits_per_choice[choice_text]
        for hit in hits:
            output_dict_per_hit = create_output_dict(qa_json, choice, hit)
            output_dicts_per_question.append(output_dict_per_hit)
    return output_dicts_per_question


def filter_hits_across_choices(hits_per_choice: Dict[str, List[EsHit]],
                               top_k: int):
    """
    Filter the hits from all answer choices(in-place) to the top_k hits based on the hit score
    """
    # collect ir scores
    ir_scores = [hit.score for hits in hits_per_choice.values() for hit in hits]
    # if more than top_k hits were found
    if len(ir_scores) > top_k:
        # find the score of the top_kth hit
        min_score = sorted(ir_scores, reverse=True)[top_k - 1]
        # filter hits below this score
        for choice, hits in hits_per_choice.items():
            hits[:] = [hit for hit in hits if hit.score >= min_score]


# Create the output json dictionary from the QA file json, answer choice json and retrieved HIT
def create_output_dict(qa_json: JsonDict, choice_json: JsonDict, hit: EsHit):
    output_dict = {
        "id": qa_json["id"],
        "question": {
            "stem": qa_json["question"]["stem"],
            "choice": choice_json,
            "support": {
                "text": hit.text,
                "type": hit.type,
                "ir_pos": hit.position,
                "ir_score": hit.score,
            }
        },
        "answerKey": qa_json["answerKey"]
    }
    return output_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', default=None, type=str)
    parser.add_argument('--output_file', default=None, type=str)
    parser.add_argument('--num_retrieve', default=8, type=int)

    args = parser.parse_args()
    add_retrieved_text(args)

"""
Input data format: each line is a json object
{
    "id": "Mercury_SC_415702", 
    "question": {
        "stem": "George wants to warm his hands quickly by rubbing them. Which skin surface will produce the most heat?", 
        "choice": {"text": "dry palms", "label": "A"}, 
        "support": {
            "text": "Motion produces heat when we rub the palms of our hands briskly together.", 
            "type": "sentence", "ir_pos": 0, "ir_score": 31.004463
        }
    }, 
    "answerKey": "A", 
    "premise": "Motion produces heat when we rub the palms of our hands briskly together.", 
    "hypothesis": "George wants to warm his hands quickly by rubbing them. Dry palms skin surface will produce the most heat.", 
    "hypothesisStructure": "George<>wants<>to warm his hands quickly by rubbing them$$$George<>wants to warm quickly<>by rubbing them$$$George<>wants to warm quickly by rubbing<>them$$$Dry palms skin surface<>will produce<>the most heat", 
    "score": 0.4067254066467285
}
...

"""
import argparse
import json

from tqdm._tqdm import tqdm

def filter_entailment(args):
    with open(args.output_file, 'w') as outfile, open(args.input_file, 'r') as infile:
        print("Writing to {} from {}".format(args.output_file, args.input_file))
        line_tqdm = tqdm(infile, dynamic_ncols=True)

        filtered_entailment = {}
        for line in line_tqdm:
            qa_json = json.loads(line)
            num_q = 0
            # Filter entailment score
            if qa_json['score'] > args.min_entail_score:
                filtered_entailment.setdefault(qa_json["id"], []).append(qa_json)
            #line_tqdm.set_postfix(hits=num_q)
        
        for qid, qa_list in filtered_entailment.items():
            print('\r q', num_q, end="")
            # Filter number of entailment texts
            if len(qa_list) <= args.max_entail_docs:
                continue
            qa_list = sorted(qa_list, key=lambda qa: qa['score'], reverse=True)
            qa_list = qa_list[:args.max_entail_docs]
            
            # Write to outfile
            output_dict = create_output_dict(qa_list)
            outfile.write(json.dumps(output_dict) + '\n')
            num_q += 1
        print()

def create_output_dict(qa_list):

    qa_object = qa_list[0]
    output_dict = {
        "id": qa_object["id"],
        "question": {
            "stem": qa_object["question"]["stem"],
            "choice": qa_object["question"]["choice"]
        },
        "answerKey": qa_object["answerKey"],
        "support": []
    }
    for qa in qa_list:
        output_dict["support"].append({
            "text": qa["question"]["support"]["text"],
            "type": qa["question"]["support"]["type"],
            "ir_pos": qa["question"]["support"]["ir_pos"],
            "ir_score": qa["question"]["support"]["ir_score"],
            "entail_score": qa["score"],
            "entail_hypo": qa["hypothesis"]
        })

    return output_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default=None, type=str)
    parser.add_argument("--output_file", default=None, type=str)
    parser.add_argument("--max_entail_docs", default=5, type=int)
    parser.add_argument("--min_entail_score", default=0.3, type=float)

    args = parser.parse_args()
    filter_entailment(args)

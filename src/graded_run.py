import os
import json
import argparse
import torch
import ollama

# Argument parser for year and model_name input
parser = argparse.ArgumentParser(description="Process data for a specific year and model.")
parser.add_argument("--year", type=str, required=True, help="Year to process (e.g., '19', '20', '21')")
parser.add_argument("--model_name", type=str, required=True, help="Model name (e.g., 'llama3.2')")
args = parser.parse_args()

year = args.year
model_name = args.model_name

device = torch.device("cuda:1")

# Load prompts
with open('prompts.json', 'r') as f:
    prompts = json.load(f)

errors = open(f'errors_{model_name}_{year}.txt', 'w')
for userid in prompts:
    if userid != '32':
        continue

    queries = {}
    for line in open(f'data/topics.filtered.dl{year}.txt', 'r').readlines():
        qid, querytext = line.split('\t')
        queries[qid] = querytext

    text_col = {}
    for line in open(f'data/qrels_text.dl{year}', 'r').readlines():
        docid = line.split('\t')[0]
        doctext = line.replace(docid, '').replace('\t', '')
        text_col[docid] = doctext

    qrels_col = {}
    out_path = f'outputs/{model_name}/graded/graded_dl{year}_{userid}.txt'
    done = {}
    if f'graded_dl{year}_{userid}.txt' in os.listdir(f'outputs/{model_name}/graded/'):
        for line in open(out_path, 'r').readlines():
            try:
                qid = line.split()[0]
                docid = line.split()[2] 
                done[(qid, docid)] = ''
            except:
                pass

    output = open(out_path, 'a')
    qcounter = []
    for line in open(f'data/qrels.dl{year}-passage.txt', 'r').readlines():
        qid = line.split()[0]
        docid = line.split()[2] 
        if qid not in qcounter:
            qcounter.append(qid)
        
        if (qid, docid) in done:
            continue

        sys_msg = prompts[userid]['graded']['system']
        user_msg = prompts[userid]['graded']['user']
        
        try:
            if '{query}' not in user_msg or '{document}' not in user_msg:
                print(f"Error: query or document not found in {userid} user message")
                errors.write(f"Error: query or document not found in {userid} graded user message\n")    
        except Exception as e:
            print(f"Error: {e}")
            errors.write(f"Error: {e}\n")
            continue
        
        user_msg = user_msg.replace('{query}', queries[qid]).replace('{document}', text_col[docid])

        try:
            prompt_answer = ollama.generate(model=model_name, prompt=user_msg, system=sys_msg,
                                            options={"temperature": 0, "seed": 42})
        except Exception as e:
            print(f"Error: {e}")
            errors.write(f"Error: {e}\n")
            continue

        rate = prompt_answer.response
        print(year, "graded", "user", userid, model_name, round(len(qcounter) / len(queries), 2), docid, rate)
        output.write(f"{qid} 0 {docid} {rate}\n")
    
    output.close()
errors.close()

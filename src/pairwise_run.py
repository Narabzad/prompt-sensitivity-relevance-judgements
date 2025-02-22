import os
import json
import re
from ollama import chat
from ollama import ChatResponse
import ollama
import argparse

# Argument parser for year input
parser = argparse.ArgumentParser(description="Process data for a specific year.")
parser.add_argument("--year", type=str, required=True, help="Year to process (e.g., '20', '21')")
parser.add_argument("--model", type=str, required=True, help="Model name (e.g., 'llama3.2', 'mistral')")

args = parser.parse_args()

year = args.year
model_name = args.model
import torch
device = torch.device("cuda:1")

with open('prompts.json', 'r') as f:
    prompts = json.load(f)
    
errors=open('errors/errors_pairwise_'+model_name+'_'+year,'a')
for userid in prompts:
    if userid != '7' and userid != '9':
        continue
    
    out_path= f'judgments/{model_name}/pairwise/pairwise_dl{year}_{userid}.txt'
    output = open(out_path,'a')
    query_list=[]
    queries={}
    for line in open(f'data/topics/topics.filtered.dl{year}.txt','r').readlines():
        qid,querytext = line.split('\t')
        queries[qid]=querytext

    text_col = {}
    for line in open(f'data/qrels/qrels_text.dl{year}','r').readlines():
        docid = line.split('\t')[0]
        doctext = line.replace(docid,'').replace('\t','')
        text_col[docid]=doctext
        
    done = {}
    if f'pairwise_dl{year}_{userid}.txt' in os.listdir(f'outputs/{model_name}/pairwise/'):
        for line in open(out_path,'r').readlines():
            try:
                if 'regular' in line:
                    qid = line.split()[0]
                    docid1 = line.split()[4]
                    docid2 = line.split()[5] 
                    done[(qid,docid1,docid2)]=''
            except:
                pass
    for line in open(f'data/pairwise/pairs_{year}','r').readlines():
        qid,l1,l2,docid1,docid2 = line.rstrip().strip().split('\t')
        
        if (qid,docid1,docid2) in done:
            continue
        sys_msg = prompts[userid]['pairwise']['system']
        user_msg = prompts[userid]['pairwise']['user']
        
        try:
            if '{query}' not in user_msg or '{documentA}' not in user_msg or '{documentB}' not in user_msg:
                print(f"Error: query or document not found in {userid} user message")
                errors.write(f"Error: query or document not found in {userid} pairwise user message\n")    
        except Exception as e:
            print(f"Error: {e}")
            errors.write(f"Error: {e}\n")
            continue
        
        user_msg1 = user_msg.replace('{query}', queries[qid]).replace('{documentB}', text_col[docid2]).replace('{documentA}', text_col[docid1])
        user_msg2 = user_msg.replace('{query}', queries[qid]).replace('{documentB}', text_col[docid1]).replace('{documentA}', text_col[docid2])

        try:
            prompt_answer = ollama.generate(model=model_name, prompt=user_msg1, system=sys_msg,\
            options={"temperature":0, "seed":42})
            
        except Exception as e:
            print(f"Error: {e}")
            errors.write(f"Error: {e}\n")
            continue

        rate = prompt_answer.response
        print(year,  model_name , "pairwise - user", userid  ,docid,rate)
        output.write(f"{qid}\t{l1}\t{l2}\tregular\t{docid1}\t{docid2}\t{rate}\n")
        
        
        try:
            prompt_answer = ollama.generate(model=model_name, prompt=user_msg2, system=sys_msg,\
            options={"temperature":0, "seed":42})
            
        except Exception as e:
            print(f"Error: {e}")
            errors.write(f"Error: {e}\n")
            continue

        rate = prompt_answer.response
        print(year,  model_name , "pairwise - user", userid  ,docid,rate)
        output.write(f"{qid}\t{l1}\t{l2}\tswapped\t{docid2}\t{docid1}\t{rate}\n")
    output.close()
errors.close()
    

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
    
errors=open('errors/errors_binary_'+model_name+'_'+year,'a')
for userid in prompts:
    if userid != '7' and userid != '9':
        continue
    
    out_path= f'judgments/{model_name}/binary/binary_dl{year}_{userid}.txt'
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
    if f'binary_dl{year}_{userid}.txt' in os.listdir(f'outputs/{model_name}/binary/'):
        for line in open(out_path,'r').readlines():
            try:
                if 'regular' in line:
                    qid = line.split()[0]
                    docid1 = line.split()[4]
                    docid2 = line.split()[5] 
                    done[(qid,docid1,docid2)]=''
            except:
                pass
    qcounter=[]
    for line in open(f'data/qrels/qrels.dl{year}-passage.txt','r').readlines():
        
        qid,_,docid,_ = line.split()
        if qid not in qcounter:
            qcounter.append(qid)
        
        if (qid,docid) in done:
            output.write(f"{qid} 0 {docid} {done[(qid,docid)]}\n")
            continue
        sys_msg = prompts[userid]['binary']['system']
        user_msg = prompts[userid]['binary']['user']

        # Check for the presence of '{query}' and '{document}' in user_msg
        
        if '{query}' not in user_msg or '{document}' not in user_msg:
            print(f"Error: query or document not found in {userid} user message")
            errors.write(f"Error: query or document not found in {userid} binary user message\n")    
        user_msg = user_msg.replace('{query}', queries[qid]).replace('{document}', text_col[docid])
        messages = [
                    {
                        "role": "system",
                        "content": sys_msg
                    },
                    {
                        "role": "user",
                        "content":   user_msg

                    }
                    ]
                    # Send request to Azure OpenAI

        try:
            response: ChatResponse = chat(model=model_name, 
                              messages=messages, 
                              options={"seed": 666, "temperature": 0})
            
        except Exception as e:
            print(f"Error: {e}")
            errors.write(f"Error: {e}\n")
            continue

        rate = response.message.content
        print(rate)
        print(year, "binary","user", userid , model_name, round(len(qcounter)/len(queries),2),docid,rate)
        output.write(f"{qid} 0 {docid} {rate}\n")
    output.close()
errors.close()
    

    

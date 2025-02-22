# Prompt Sensitivity in LLM-Based Relevance Judgments
This repository contains data and scripts for analyzing prompt sensitivity in LLM-based relevance judgments.
It collects 72 prompts from both human experts and LLMs, applying them to three different relevance judgement methods (binary, graded, and pairwise) using GPT-4o, Mistral, and LLaMA as judges. The study compares LLM judgments with TREC Deep Learning 2020 & 2021 labels, analyzing prompt robustness across models and LLMs as prompt generators. 

[Prompt collection Guideline](https://drive.google.com/file/d/1mBn58tj2EZn3NvnW1s1Gn3gUjRotNvDq/view)
## 📂 Repository Structure
```
prompt-sensitivity-relevance-judgements/
├── prompts.json             # The prompts used for LLM-based relevance judgment
├── data/
│   ├── topics/
│   │   ├── topics.filtered.dl20.txt    # Queries for DL 2020
│   │   └── topics.filtered.dl21.txt    # Queries for DL 2021
│   ├── qrels/
│   │   ├── qrels.dl20-passage.txt       # Human labels for DL 2020
│   │   └── qrels.dl21-passage.txt       # Human labels for DL 2021
│   ├── pairwise/
│   │   ├── pairs_20.txt                 # Selected passage pairs for DL 2020
│   │   └── pairs_21.txt                 # Selected passage pairs for DL 2021
│── judgments/
|   ├── gpt-4o/
|   │   ├── binary/                  # Binary relevance judgments by GPT-4o
|   │   ├── graded/                  # Graded relevance judgments by GPT-4o
|   │   └── pairwise/                # Pairwise relevance judgments by GPT-4o 
|   ├── mistral/
|   |   │   ├── binary/                  # Binary relevance judgments by Mistral
|   |   │   ├── graded/                  # Graded relevance judgments by Mistral
|   |   │   └── pairwise/                # Pairwise relevance judgments by Mistral
|   └── llama/
|           ├── binary/                  # Binary relevance judgments by LLaMA
|           ├── graded/                  # Graded relevance judgments by LLaMA
|           └── pairwise/                # Pairwise relevance judgments by LLaMA
└── src/                     # Scripts for running LLM-based judgments
     ├── pairwise_run.py # Runs pairwise judgments
     ├── binary_run.py # Runs binary judgments
     └── graded_run.py # Runs graded judgments 
```


## Judgment File Naming Convention

Each relevance judgment file is named following this pattern: ```[model_name]_[judgment_type]dl[year][participant_id].txt```


Where:
- `[model_name]` → The LLM used for judgments (`gpt-4o`, `mistral`, `llama`).
- `[judgment_type]` → The type of relevance judgment:
  - `binary` → Binary (0/1) judgments.
  - `graded` → Graded relevance (0-3 scale).
  - `pairwise` → Pairwise comparison between two passages.
- `dl[year]` → The dataset year (`dl20` for 2020, `dl21` for 2021).
- `[participant_id]` → The ID of the prompt used for the judgment.

** Example Filenames:**
- `mistral_graded_dl20_1.txt` → Graded relevance judgments by Mistral on DL 2020, using prompt #1.
- `gpt-4o_binary_dl21_5.txt` → Binary judgments by GPT-4o on DL 2021, using prompt #5.

## Prompt Processing Details

- Total Prompts: Initially, there were 32 prompts in `prompts.json`.  2 were test prompts and excluded. 30 prompts were used in the experiments.
- Final Processed Prompts: After filtering, only 12 prompts from humans and 12 prompts from LLMs were considered valid for generating judgments.
- Some judgment files may contain fewer lines than others. This happens because some LLMs failed to judge certain passages or returned incomplete results.

##  Running the Judgements with Ollama

You can run the scripts inside the `src/` folder to generate relevance judgments for different LLMs.

```
python src/binary_run.py --year [20 or 21] --model [llama3.2 or mistral]
python src/pairwise_run.py --year [20 or 21] --model [llama3.2 or mistral]
python src/graded_run.py --year [20 or 21] --model [llama3.2 or mistral]
```
Note: If you choose to run GPT-4o, you must modify the script to add your OpenAI or Azure OpenAI key and correctly format the model output.

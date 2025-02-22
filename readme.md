# Prompt Sensitivity in LLM-Based Relevance Judgments
This repository contains data and scripts for analyzing prompt sensitivity in LLM-based relevance judgments.

## Repository Structure

prompt-sensitivity-relevance-judgements/ │── prompts.json # The prompts used for LLM-based relevance judgment │── data/ │ ├── topics/ │ │ ├── topics.filtered.dl20.txt # Queries for DL 2020 │ │ ├── topics.filtered.dl21.txt # Queries for DL 2021 │ ├── qrels/ │ │ ├── qrels.dl20-passage.txt # Human labels for DL 2020 │ │ ├── qrels.dl21-passage.txt # Human labels for DL 2021 │ ├── pairwise/ │ │ ├── pairs_20.txt # Selected passage pairs for DL 2020 │ │ ├── pairs_21.txt # Selected passage pairs for DL 2021 │ ├── judgments/ │ │ ├── gpt-4o/ │ │ │ ├── binary/ # Binary relevance judgments by GPT-4o │ │ │ ├── graded/ # Graded relevance judgments by GPT-4o │ │ │ ├── pairwise/ # Pairwise relevance judgments by GPT-4o │ │ ├── mistral/ │ │ │ ├── binary/ # Binary relevance judgments by Mistral │ │ │ ├── graded/ # Graded relevance judgments by Mistral │ │ │ ├── pairwise/ # Pairwise relevance judgments by Mistral │ │ ├── llama/ │ │ │ ├── binary/ # Binary relevance judgments by LLaMA │ │ │ ├── graded/ # Graded relevance judgments by LLaMA │ │ │ ├── pairwise/ # Pairwise relevance judgments by LLaMA 

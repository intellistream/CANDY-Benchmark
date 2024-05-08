# Notebook for Fancy downstream tasks

## basic_search.ipynb

Just to show how CANDY works under python, no other dependencies

## rag.ipynb

### Overview

This is a toy example of RAG that supports online ingestion of knowledge. Specifically, we argument llama with
knowledge from WarThunder Wiki (see warthunder.py) in an online manner. Eventually, llama knows a very strange plane
named F-16A_ADF thanks to the RAG powered by CANDY.

### How it works

We craft this RAG toy by putting CANDY into a DPR retriever (see CANDYRetriever.py), while the rest part of RAG follows
the llama_cpp example from https://github.com/IntelLabs/fastRAG.git

### Dependencies

Please first make sure you have installed the fastRAG package

## rag_evaluation.ipynb
More advanced than rag.ipynb, where we can get the bleu and rouge scores in an unified way and save into csv.

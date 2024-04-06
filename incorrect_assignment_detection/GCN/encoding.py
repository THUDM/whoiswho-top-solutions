import json as js
import torch
import os
import pickle as pk
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default="dataset/pid_to_info_all.json")
parser.add_argument("--save_path", type = str, default = "dataset/roberta_embeddings.pkl")
args = parser.parse_args()

with open(args.path, "r", encoding="utf-8") as f:
    papers = js.load(f)

batch_size = 5000
device = torch.device("cuda:0")

# Initialize RoBERTa tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('roberta-base')
model = AutoModel.from_pretrained('roberta-base').to(device)
dic_paper_embedding = {}
papers = [[key, value] for key,value in papers.items()]
for ii in tqdm(range(0, len(papers), batch_size), total=len(papers)//batch_size):
    
    batch_papers = papers[ii: ii + batch_size]
    texts = [paper[1]["title"] for paper in batch_papers]
    
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=30)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    tt = 0
    for jj in range(ii, ii+len(batch_papers)):
        paper_id = papers[jj][0]
        paper_vec = embedding[tt]
        tt+=1
        dic_paper_embedding[paper_id] = paper_vec

with open(args.save_path, "wb") as f:
    pk.dump(dic_paper_embedding, f)

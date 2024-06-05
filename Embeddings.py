# Word Embeddings

import torch
import pandas as pd
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from utils import batch

# LLM
MODEL_NAME = 'allenai/longformer-base-4096'

# DATA
df = pd.read_csv('') # Dataframe with 'text' column of text data
df['text']=df['text'].apply(lambda x : x.strip()) # Remove any blankspace after text

# GPU processing
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

# LLM model config
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)



# Batching and Epochs -- tokenizer settings for longformer
outer=[]
for i in tqdm(range(0,len(df),32)):
    corpus=df['text'][i:i+32]
    inner=[]
    for epoch in batch(corpus, 16):
        tokenized_train = tokenizer(epoch.values.tolist(), 
                                    padding=True, 
                                    pad_to_multiple_of=512,
                                    truncation = True,
                                    max_length=4096, 
                                    return_tensors='pt')
        tokenized_train = {k:v.detach().clone().to(device) for k,v in tokenized_train.items()}
        with torch.no_grad():
            hidden_train = model(**tokenized_train) #dim : [batch_size(nr_sentences), tokens, emb_dim]


            # get only the [CLS] embeddings
            cls_train = hidden_train.last_hidden_state[:,0,:]
            inner.append(cls_train.to('cpu'))
    outer.append(torch.cat(inner))

CLS_hidden_states=torch.cat(outer)
torch.save(CLS_hidden_states, '') # output location
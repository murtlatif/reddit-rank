import random
import torch
import torch.nn as nn
from torchtext import data
import torchtext
from filters import update_title
import re
from datetime import datetime
import spacy
import pandas as pd

TITLE = data.Field(sequential=True, lower=True,
                    tokenize='spacy', include_lengths=True)

total_data = data.TabularDataset(path='cgi-bin/all.csv', format='csv', skip_header=True, fields=[('title', TITLE)])

TITLE.build_vocab(total_data)
TITLE.vocab.load_vectors(torchtext.vocab.GloVe(name='6B', dim=100))
vocab = TITLE.vocab

def tokenizer(text):
    spacy_en = spacy.load('en')
    return [tok.text for tok in spacy_en(text)]

def is_serious(title):
    if re.match(r'[\[\(]serious[\)\]]', title, flags=re.IGNORECASE) is not None:
        return 1
    return 0

def onehot_hour(hour):
    return [1 if hour == i else 0 for i in range(24)]

def onehot_weekday(weekday):
    return [1 if weekday == i else 0 for i in range(7)]

def evaluate_input(title, nsfw, classifier_type):

    model = torch.load(f'cgi-bin/model_{classifier_type}.pt', map_location=torch.device('cpu'))
    model.eval()

    serious_flair = is_serious(title)
    over_18 = 1 if nsfw else 0

    curr_time = datetime.now()
    hour = curr_time.hour
    weekday = curr_time.weekday()

    onehot_hr = onehot_hour(hour)
    onehot_wd = onehot_weekday(weekday)

    title = update_title(title)

    tokens = tokenizer(title)
    if len(tokens) < 4:
        return 'The input sentence must contain at least 4 words.'

    token_ints = [vocab.stoi[tok] for tok in tokens]

    token_tensor = torch.LongTensor(token_ints).view(-1, 1)
    lengths = torch.tensor([len(token_ints)])

    context = torch.tensor([serious_flair, over_18] + onehot_hr + onehot_wd).float().unsqueeze(0)
    
    prediction = model(token_tensor, context, lengths)
    activated_prediction = torch.argmax(prediction)
    output_data = {
        'prediction': int(activated_prediction),
        'serious': serious_flair,
        'over_18': over_18,
        'hour': hour,
        'weekday': weekday,
        'title': title,
        'test': prediction
    }
    return output_data

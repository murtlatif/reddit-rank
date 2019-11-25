import random
import torch
import torch.nn as nn
from filters import update_title
import re

model = torch.load('cgi-bin/model.pt', map_location=torch.device('cpu'))

def class_to_text(num):
    scores = ['zero', 'low', 'decent', 'high', 'viral']
    return scores[num]

def clean_title(title):
    update_title(title)

def is_serious(title):
    if re.match(r'[\[\(]serious[\)\]]', title) is not None:
        return 1
    return 0

def evaluate_input(title, nsfw):
    serious_flair = is_serious(title)
    over_18 = nsfw
    
    evaluation = random.randint(0, 4)
    return class_to_text(evaluation)
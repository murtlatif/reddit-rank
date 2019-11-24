import random

def class_to_text(num):
    scores = ['zero', 'low', 'decent', 'high', 'viral']
    return scores[num]
    
def tester(title):
    evaluation = random.randint(0, 4)
    return class_to_text(evaluation)
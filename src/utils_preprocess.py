import re

def clean_text(dataset):
    dataset = dataset.map(lambda example: {'text': re.sub('[@#]\S+', '', example['text'])})
    dataset = dataset.map(lambda example: {'text': re.sub('\s{2,}', ' ', example['text'])})
    return dataset
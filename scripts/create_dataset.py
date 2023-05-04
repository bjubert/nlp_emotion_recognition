from datasets import concatenate_datasets, load_dataset, Dataset, ClassLabel
import os
import pandas as pd
import re

def clean_text(dataset):
    dataset = dataset.map(lambda example: {'text': re.sub('[@#]\S+', '', example['text'])})
    dataset = dataset.map(lambda example: {'text': re.sub('\s+', '', example['text'])})
    return dataset

def import_dair_ai():
    dataset = load_dataset('dair-ai/emotion', split=None)
    ds = concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])
    #sadness (0), joy (1), love (2), anger (3), fear (4), surprise (5).
    #affect values to emotions and remove line if key is not in emotions
    emotions = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
    #Filter the dataset to keep only the emotions in the emotions dictionary
    ds = ds.filter(lambda example: example['label'] in emotions.keys())
    #Map the values of the emotion to the corresponding emotion
    def map_emotions(example):
        example['emotion'] = emotions[example['label']]
        del example['label']
        return example
    ds = ds.map(map_emotions)
    #Renommer la colonne emotion en label
    ds = ds.rename_column('emotion', 'label')
    #ds.map(lambda example: {'label': emotions[example['label']]})
    return ds

def import_kaggle_emotion():
    with open('data/raw_datasets/kaggle_emotion/train.txt', 'r') as f:
        train = f.readlines()
        train = [line.strip().split(';') for line in train]
        #create dataset huggingface
        train = pd.DataFrame(train, columns=['text', 'label'])
        ds = Dataset.from_pandas(train)
        return ds


def import_sentiment_analysis_in_text():   
    df = pd.read_csv('https://query.data.world/s/nzgta6r4vf4sl7nmiwudkglbberul4?dws=00000')
    df = df[['content', 'sentiment']]
    df = df.rename(columns={'sentiment': 'label'})
    df = df.rename(columns={'content': 'text'})
    ds = Dataset.from_pandas(df)
    return ds

def import_daily_dialog():
    #import dataset from huggingface
    dataset = load_dataset('daily_dialog', split=None)  
    #concatenate train, validation and test
    ds = concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])
    #rename columns
    ds = ds.rename_column('emotion', 'label')
    ds = ds.rename_column('dialog', 'text')
    #remove columns
    ds = ds.remove_columns(['act'])
    #For each sentence in the list of sentences (text variable), create a row for each sentence
    #and add the corresponding emotion label
    sentences = []
    labels = []
    for text, label in zip(ds['text'], ds['label']):
        sentences.extend(text)
        labels.extend(label)
    #create dataset huggingface
    ds = Dataset.from_dict({'text': sentences, 'label': labels})
    #affect values to emotions and remove line if key is not in emotions
    emotions = {0: 'neutral', 1: 'anger', 2: 'disgust', 3: 'fear', 4: 'happiness'}
    #Filter the dataset to keep only the emotions in the emotions dictionary
    ds = ds.filter(lambda example: example['label'] in emotions.keys())
    ds = ds.map(lambda example: {'label': emotions[example['label']]})
    return ds

def create_dataset():
    dsk = [import_dair_ai(), import_kaggle_emotion(), import_sentiment_analysis_in_text(), import_daily_dialog()]
    for ds in dsk:
        ds = clean_text(ds)
    dsk = concatenate_datasets(dsk)
    return dsk

def save_dataset(ds):
    ds.save_to_disk('data/dataset')

ds = create_dataset()
save_dataset(ds)



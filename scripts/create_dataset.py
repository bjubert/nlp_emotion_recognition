from datasets import concatenate_datasets, load_dataset, Dataset, ClassLabel
import pandas as pd
import re
import sys
sys.path.append('/app')
from src import utils_preprocess as up

def import_dair_ai():
    dataset = load_dataset('dair-ai/emotion', split=None)
    ds = concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])
    #sadness (0), joy (1), love (2), anger (3), fear (4), surprise (5).
    #affect values to emotions and remove line if key is not in emotions
    emotions = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
    #Filter the dataset to keep only the emotions in the emotions dictionary
    ds = ds.filter(lambda example: example['label'] in emotions.keys())
    #Map the values of the emotion to the corresponding emotion. Workaround because map function doesn't work.
    def map_emotions(example):
        example['emotion'] = emotions[example['label']]
        del example['label']
        return example
    ds = ds.map(map_emotions)
    #ds.map(lambda example: {'label': emotions[example['label']]})

    #Renommer la colonne emotion en label
    ds = ds.rename_column('emotion', 'label')
    print(set(ds['label']))
    return ds

def import_kaggle_emotion():
    with open('data/raw_datasets/kaggle_emotion/train.txt', 'r') as f:
        train = f.readlines()
        train = [line.strip().split(';') for line in train]
        #create dataset huggingface
        train = pd.DataFrame(train, columns=['text', 'label'])
        ds = Dataset.from_pandas(train)
        print(set(ds['label']))
        return ds

def import_kaggle_tasks():
    with open('data/raw_datasets/kaggle_tasks/training.csv', 'r') as f:
        train = f.readlines()
        train = [line.strip().split(',') for line in train]
        #create dataset huggingface
        train = pd.DataFrame(train, columns=['text', 'label'])
        ds = Dataset.from_pandas(train)
        # 0 : sadness, 1 : joy, 2 : love, 3 : anger, 4 : fear 
        #affect values to emotions and remove line if key is not in emotions
        emotions = {'0': 'sadness', '1': 'joy', '2': 'love', '3': 'anger', '4': 'fear'}
        #Filter the dataset to keep only the emotions in the emotions dictionary
        ds = ds.filter(lambda example: example['label'] in emotions.keys())
        ds = ds.map(lambda example: {'label': emotions[example['label']]})
        print(ds)
        print(set(ds['label']))
        return ds
    

def import_sentiment_analysis_in_text():   
    df = pd.read_csv('https://query.data.world/s/nzgta6r4vf4sl7nmiwudkglbberul4?dws=00000')
    df = df[['content', 'sentiment']]
    df = df.rename(columns={'sentiment': 'label'})
    df = df.rename(columns={'content': 'text'})
    ds = Dataset.from_pandas(df)
    print(set(ds['label']))
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
    print(set(ds['label']))
    return ds

def create_dataset():
    ds_list = []
    for ds in [import_dair_ai(), import_kaggle_emotion(), import_kaggle_tasks(), import_sentiment_analysis_in_text(), import_daily_dialog()]:
        ds = up.clean_text(ds)
        ds_list.append(ds)
    datasets = concatenate_datasets(ds_list)
    #replace happiness by joy
    datasets = datasets.map(lambda example: {'label': 'joy' if example['label'] == 'happiness' else example['label']})
    # replace worry by fear
    datasets = datasets.map(lambda example: {'label': 'fear' if example['label'] == 'worry' else example['label']})
    # remove empty labels
    datasets = datasets.filter(lambda example: example['label'] != 'empty')
    return datasets

def save_dataset(ds):
    ds.save_to_disk('data/dataset')

ds = create_dataset()
save_dataset(ds)




"""Data Processing and Feature engineering  ."""
import argparse
import logging
import pathlib

import boto3
import numpy as np
import pandas as pd

import os
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

if __name__ == "__main__":
    
    logger.info('Installing packages.')
    install('torch')
    install('transformers')
    install('datasets[s3]')
    
    logger.info("Starting preprocessing.")
    
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--input-data", type=str, required=True)
#     args = parser.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    # tokenizer used in preprocessing
    tokenizer_name = 'distilbert-base-uncased'

    # dataset used
    dataset_name = 'imdb'

    # load dataset
    dataset = load_dataset(dataset_name)

    # download tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # tokenizer helper function
    def tokenize(batch):
        return tokenizer(batch['text'], padding='max_length', truncation=True)

    logger.info("Loading data.")
    # load dataset
    train_dataset, test_dataset = load_dataset('imdb', split=['train', 'test'])
    test_dataset = test_dataset.shuffle().select(range(1000)) # smaller the size for test dataset to 1k 

    # tokenize dataset
    train_dataset = train_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)

    # set format for pytorch
    train_dataset =  train_dataset.rename_column("label", "labels")
    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    test_dataset = test_dataset.rename_column("label", "labels")
    test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    logger.info("Save preprocessing data for training locally.")
    
    train_dataset.save_to_disk('/opt/ml/processing/train')
    test_dataset.save_to_disk('/opt/ml/processing/test')

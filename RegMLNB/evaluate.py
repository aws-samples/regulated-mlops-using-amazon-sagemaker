
"""Evaluation script"""

import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, set_seed
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, fbeta_score, confusion_matrix, precision_recall_curve, auc
from datasets import load_from_disk
from torch.nn import CrossEntropyLoss, Softmax
import random
import logging
import sys
import argparse
import os
import torch
import pandas as pd
import time
import numpy as np
from numpy import argmax
import json
import logging
import pathlib
import tarfile


logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

if __name__ == "__main__":
    logger.debug("Starting evaluation.")
    model_path = "/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path="./hf_model")
    print('Logger Debug..')
    logger.debug(os.listdir('./hf_model'))
    
    test_dir = "/opt/ml/processing/test/"
    test_dataset = load_from_disk(test_dir)
    
    
    model = AutoModelForSequenceClassification.from_pretrained('./hf_model')
    tokenizer = AutoTokenizer.from_pretrained('./hf_model')

    trainer = Trainer(model=model)
    
    eval_result = trainer.evaluate(eval_dataset=test_dataset)

    with open('./hf_model/evaluation.json') as f:
        eval_result = json.load(f)
           
    logger.debug(eval_result)
    output_dir = "/opt/ml/processing/evaluation"

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        
    # writes eval result to file which can be accessed later in s3 ouput
    with open(os.path.join(output_dir, "eval_results.txt"), "w") as writer:
        print(f"***** Eval results *****")
        for key, value in sorted(eval_result.items()):
            writer.write(f"{key} = {value}\n")

    
    ##Predictions on each row of the test dataset
    print('--- Inference For Downstream Analysis---')
    
    arr_text = [tokenizer.decode(ele["input_ids"]) for ele in test_dataset]
    predictions = trainer.predict(test_dataset)
    
    softmax = Softmax(dim=1)
    softmax_preds = softmax(torch.Tensor(predictions.predictions))
    prob_class0 = softmax_preds[:, 0]
    prob_class1 = softmax_preds[:, 1]
    
    arr_preds = predictions.predictions.argmax(-1)
    arr_labels = predictions.label_ids
    acc = accuracy_score(arr_labels, arr_preds)

    # Data to plot precision - recall curve
    precision, recall, thresholds = precision_recall_curve(arr_labels, arr_preds)
    
    # Use AUC function to calculate the area under the curve of precision recall curve
    pr_auc = auc(recall, precision)
    
    report_dict = {
        "binary_classification_metrics": {
            "accuracy": {
                "value": acc,
                "standard_deviation": "NaN",
            },
            "pr_auc": {
                "value": pr_auc,
                "standard_deviation": "NaN",
            },
        }
    }
    
    print("Classification report:\n{}".format(report_dict))    
    logger.info("Writing out evaluation report with accuracy: %f", acc)
    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))

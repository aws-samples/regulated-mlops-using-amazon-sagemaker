"""Model Quality Monitor script"""

import os
import time
import boto3
import json
import sys
import argparse
import subprocess
import logging
import random
import time
from datetime import datetime

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

if __name__ == "__main__":
    logger.info('Installing packages.')
    install('sagemaker==2.59.1.post0')
    
    logger.info("Starting model quality monitor.")

    # Parse argument variables passed via the DeployModel processing step
    parser = argparse.ArgumentParser()
    parser.add_argument('--endpoint-name', type=str)
    parser.add_argument('--baseline-dataset-uri', type=str)
    parser.add_argument('--ground-truth-upload-path', type=str)
    parser.add_argument('--role', type=str)
    parser.add_argument('--region', type=str)
    args = parser.parse_args()

    # Configure AWS session
    os.environ["AWS_DEFAULT_REGION"] = args.region

    import sagemaker
    from sagemaker.transformer import Transformer
    from sagemaker.s3 import S3Downloader, S3Uploader
    
    ### Monitoring Step ###

    region = args.region
    role = args.role
    endpoint_name = args.endpoint_name

    NUM_INVOKES=10

    def ground_truth_with_id(inference_id):
        random.seed(inference_id)
        rand = random.random()
        return {
            "groundTruthData": {
                "data": "1" if rand < 0.5 else "0",  # randomly generate positive labels 50% of the time
                "encoding": "CSV",
            },
            "eventMetadata": {
                "eventId": str(inference_id),
            },
            "eventVersion": "0",
        }

    def upload_ground_truth(records, upload_time):
        fake_records = [json.dumps(r) for r in records]
        data_to_upload = "\n".join(fake_records)
        target_s3_uri = f"{args.ground_truth_upload_path}/{upload_time:%Y/%m/%d/%H/%M%S}.jsonl"
        print(f"Uploading {len(fake_records)} records to", target_s3_uri)
        S3Uploader.upload_string_as_file_body(data_to_upload, target_s3_uri)

    def generate_fake_ground_truth():
        fake_records = [ground_truth_with_id(i) for i in range(NUM_INVOKES)]
        upload_ground_truth(fake_records, datetime.utcnow())

    generate_fake_ground_truth()
    logger.info("Generating fake ground truth data to demonstrate model quality monitor.")

    # Model Quality Monitor
    from sagemaker import get_execution_role, session, Session
    from sagemaker.model_monitor import ModelQualityMonitor
    from sagemaker.model_monitor.dataset_format import DatasetFormat

    model_quality_monitor = ModelQualityMonitor(
        role=role,
        instance_count=1,
        instance_type='ml.m5.xlarge',
        volume_size_in_gb=20,
        max_runtime_in_seconds=1800
        #sagemaker_session=sagemaker_session
    )

    baseline_job_name = f"MyBaseLineJob-{datetime.now():%Y-%m-%d-%H-%M-%S}"

    job = model_quality_monitor.suggest_baseline(
        job_name=baseline_job_name,
        baseline_dataset=args.baseline_dataset_uri, # The S3 location of the validation dataset.
        dataset_format=DatasetFormat.csv(header=True),
        #output_s3_uri = baseline_results_uri, # The S3 location to store the results.
        problem_type='BinaryClassification',
        inference_attribute= "prediction", # The column in the dataset that contains predictions.
        probability_attribute= "probability", # The column in the dataset that contains probabilities.
        ground_truth_attribute= "label" # The column in the dataset that contains ground truth labels.
    )
    job.wait()

    from sagemaker.model_monitor import CronExpressionGenerator
    from sagemaker.model_monitor import EndpointInput
    from time import gmtime, strftime

    mon_schedule_name = endpoint_name + '-quality-monitoring-schedule'

    # Create an EnpointInput
    endpointInput = EndpointInput(
        endpoint_name=endpoint_name,
        destination="/opt/ml/processing/input_data",
        #inference_attribute= "" # The column in the dataset that contains predictions.
        probability_attribute= "score", # The column in the dataset that contains probabilities.
        probability_threshold_attribute = 0.7 # Threshold value to distinguish between binary classes
    )

    # Create an hourly model quality monitoring schedule
    model_quality_monitor.create_monitoring_schedule(
        monitor_schedule_name=mon_schedule_name,
        endpoint_input=endpointInput,
        problem_type="BinaryClassification",
        ground_truth_input=args.ground_truth_upload_path,
        #statistics=my_default_monitor.baseline_statistics(),
        constraints=model_quality_monitor.suggested_constraints(),
        schedule_cron_expression=CronExpressionGenerator.hourly(),
        enable_cloudwatch_metrics=True,
    )
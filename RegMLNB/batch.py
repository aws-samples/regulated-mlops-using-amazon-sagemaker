import argparse
import os
import subprocess
import sys
import logging
import boto3

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

if __name__ == "__main__":
    logger.info('Installing packages.')
    install('sagemaker==2.59.1.post0')
    
    logger.info("Starting batch transform.")
    import sagemaker
    from sagemaker.transformer import Transformer

    # Parse argument variables passed via the DeployModel processing step
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str)
    parser.add_argument('--batch-input-uri', type=str)
    parser.add_argument('--region', type=str)
    args = parser.parse_args()

    # Configure AWS session
    os.environ["AWS_DEFAULT_REGION"] = args.region

    transformer = Transformer(model_name=args.model_name,
                            instance_count=1,
                            instance_type="ml.m5.xlarge",
                            accept="application/json",
                            assemble_with="None"
                            )

    transformer.transform(args.batch_input_uri,
                        data_type = 'S3Prefix',
                        content_type='application/json',
                        split_type='None',
                        #input_filter="$.batch_inputs",
                        #output_filter="$.inputs",
                        join_source="Input"
                        )

    print('Waiting for transform job: ' + transformer.latest_transform_job.job_name)
    transformer.wait()
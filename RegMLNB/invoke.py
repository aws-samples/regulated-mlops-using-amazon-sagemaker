"""
This Lambda function invokes the endpoint 10 times to demonstrate the model quality monitor.
"""

import boto3
import random 
import json
import time
from datetime import datetime
import uuid


def lambda_handler(event, context):
    """ """
    client = boto3.client('sagemaker-runtime')

    # The name of the endpoint to be invoked
    endpoint_name = event["endpoint_name"]
    ground_truth_upload_path = event["ground_truth_upload_path"]
   
    NUM_INVOKES = 10

    # Invoke endpoint for model quality monitor example
    for i in range(NUM_INVOKES):
        payload = generate_payload()
        response = client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Body=payload,
            InferenceId=str(i), # unique ID per row
        )["Body"].read()
        time.sleep(1)

    # Re-generate fake ground truth data for model quality monitor example
    generate_fake_ground_truth(ground_truth_upload_path)

    # Set hourly cloudwatch event trigger for Lambda function
    event_client = boto3.client('events')
    
    # Create a rule that's scheduled to run every 1 hour
    rslt = event_client.put_rule(Name='TRIGGER_ENDPOINT_HOURLY', ScheduleExpression='rate(1 hour)', State='ENABLED')

    # Add this lambda function as a target
    rslt = event_client.put_targets(Rule='TRIGGER_ENDPOINT_HOURLY',
                                    Targets=[{'Arn': context.invoked_function_arn,
                                        'Id': 'RANDOM_ID',
                                        'Input': '{"endpoint_name": "' + endpoint_name +'", "ground_truth_upload_path": "' + ground_truth_upload_path + '"}',
                                    }])
    
    # Get event ARN for lambda trigger
    event_arn = event_client.describe_rule(Name='TRIGGER_ENDPOINT_HOURLY')['Arn']
    
    # Add permissions for this lambda to be invoked by event trigger
    lambda_clnt = boto3.client('lambda')
    print(str(uuid.uuid4()))
    rslt = lambda_clnt.add_permission(FunctionName=context.invoked_function_arn,
                                      StatementId=str(uuid.uuid4()),
                                      Action='lambda:InvokeFunction',
                                      Principal='events.amazonaws.com',
                                      SourceArn=event_arn)

    return {
        "statusCode": 200,
        "body": json.dumps("Endpoint Invoked!"),
    }

#Invoke the endpoint with fake calls to allow model quality monitor to run
def generate_payload():
    payload = {
        "inputs": "Hello. This is an amazing test run from Python " + str(random.random())
    }
    return json.dumps(payload)

# Ground truth generator function
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

# Upload ground truth to s3 function
def upload_ground_truth(records, upload_time, ground_truth_upload_path):
    fake_records = [json.dumps(r) for r in records]
    data_to_upload = "\n".join(fake_records)
    target_s3_uri = f"{ground_truth_upload_path}/{upload_time:%Y/%m/%d/%H/%M%S}.jsonl"
    print(f"Uploading {len(fake_records)} records to", target_s3_uri)
    
    s3 = boto3.resource('s3')
    bucket, key = split_s3_path(target_s3_uri)
    s3.Object(bucket,key).put(Body=data_to_upload)

def generate_fake_ground_truth(ground_truth_upload_path):
    NUM_INVOKES = 10
    
    fake_records = [ground_truth_with_id(i) for i in range(NUM_INVOKES)]
    upload_ground_truth(fake_records, datetime.utcnow(), ground_truth_upload_path)
    
def split_s3_path(s3_path):
    path_parts=s3_path.replace("s3://","").split("/")
    bucket=path_parts.pop(0)
    key="/".join(path_parts)
    return bucket, key
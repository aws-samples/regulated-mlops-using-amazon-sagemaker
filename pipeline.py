import os
import time
import boto3
import sagemaker
import json
import sys

from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput

from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    Processor,
    ScriptProcessor,
)

from sagemaker import Model
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.huggingface import HuggingFace, HuggingFaceModel

from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, CacheConfig
from sagemaker.workflow.lambda_step import (
    LambdaStep,
    LambdaOutput,
    LambdaOutputTypeEnum,
)
from sagemaker.workflow.step_collections import CreateModelStep
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.condition_step import (
    ConditionStep,
    JsonGet,
)
from sagemaker.lambda_helper import Lambda
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.pipeline import Pipeline

region = sagemaker.Session().boto_region_name
sm_client = boto3.client("sagemaker")
s3_client = boto3.client('s3', region_name=region)
boto_session = boto3.Session(region_name=region)
sagemaker_session = sagemaker.session.Session(
    boto_session=boto_session, sagemaker_client=sm_client
)
s3_prefix = "RegMLNB"
model_package_group_name = f"RegMLNBModelPackageGroupName"

# Opening JSON file with CDK outputs
f = open('./cdk-outputs.json')

data = json.load(f)
f.close()

# Pull role arn key from the json list of CDK outputs
regml_output = data['regml-stack']
role_arn_key = list(regml_output.keys())[0]
role = data['regml-stack'][role_arn_key]

print('Role:', role)
lambda_role = role
default_bucket = sagemaker_session.default_bucket()
output_destination = "s3://{}/{}/data".format(default_bucket, s3_prefix)

from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)

# processing step parameters
processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
processing_instance_type = ParameterString(
    name="ProcessingInstanceType", default_value="ml.c5.2xlarge"
)

# training step parameters
training_instance_type = ParameterString(
    name="TrainingInstanceType", default_value="ml.p3.2xlarge"
)
training_instance_count = ParameterInteger(name="TrainingInstanceCount", default_value=1)

# model approval status
model_approval_status = ParameterString(
    name="ModelApprovalStatus",
    default_value="PendingManualApproval",  # ModelApprovalStatus can be set to a default of "Approved" if you don't want manual approval.
)
# cache configuration
cache_config = CacheConfig(enable_caching=True, expire_after="30d")

########## Data Processing ##########
print('Data Processing Step ..')
sklearn_processor = SKLearnProcessor(
    framework_version="0.23-1",
    instance_type=processing_instance_type,
    instance_count=processing_instance_count,
    base_job_name="RegMLNB-preprocessing",
    role=role,
)

step_process = ProcessingStep(
    name="Processing",
    processor=sklearn_processor,
    outputs=[
        ProcessingOutput(
            output_name="train",
            destination="{}/train".format(output_destination),
            source="/opt/ml/processing/train",
        ),
        ProcessingOutput(
            output_name="test",
            destination="{}/test".format(output_destination),
            source="/opt/ml/processing/test",
        ),
        ProcessingOutput(
            output_name="validation",
            destination="{}/test".format(output_destination),
            source="/opt/ml/processing/validation",
        ),
    ],
    code="./RegMLNB/preprocessing.py",
    cache_config=cache_config,
)

########## Training ##########

print('Training Step ..')

container = '763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:1.7.1-transformers4.6.1-gpu-py36-cu110-ubuntu18.04' 

hyperparameters = {
"model_name": "distilbert-base-uncased",
"train_batch_size": 32,
"epochs": 1,
}

estimator = HuggingFace(
    image_uri=container,
    entry_point="train.py",
    source_dir='./RegMLNB',
    base_job_name="RegMLNB" + "/training",
    instance_type=training_instance_type,
    instance_count=training_instance_count,
    role=role,
    transformers_version="4.6.1",
    pytorch_version="1.7.1",
    py_version="py36",
    hyperparameters=hyperparameters,
    sagemaker_session=sagemaker_session,
)

step_train = TrainingStep(
name="TrainHuggingFaceModel",
estimator=estimator,
inputs={
    "train": TrainingInput(
        s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
            "train"
        ].S3Output.S3Uri,
        #content_type="text/csv"
    ),
    "test": TrainingInput(
        s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
            "test"
        ].S3Output.S3Uri,
        #content_type="text/csv"
    ),
},
cache_config=cache_config,
)

########## Model Evaluation ##########

print('Evaluation Step ..')

# Processing step for evaluation
script_eval = ScriptProcessor(
    image_uri=container,
    command=["python3"],
    instance_type=processing_instance_type,
    instance_count=1,
    base_job_name=f"script-RegMLNB-eval",
    sagemaker_session=sagemaker_session,
    role=role,
)

evaluation_report = PropertyFile(
    name="RegMLNBEvaluationReport",
    output_name="evaluation",
    path="evaluation.json",
)
step_eval = ProcessingStep(
    name="RegMLNBEval",
    processor=script_eval,
    inputs=[
        ProcessingInput(
            source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
            destination="/opt/ml/processing/model",
        ),
        ProcessingInput(
            source=step_process.properties.ProcessingOutputConfig.Outputs[
                "test"
            ].S3Output.S3Uri,
            destination="/opt/ml/processing/test",
        ),
    ],
    outputs=[
        ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation", destination = f"{output_destination}/evaluation_report",
                        ),
    ],
    code="./RegMLNB/evaluate.py",
    property_files=[evaluation_report],
    cache_config=cache_config,
)

########## MODEL REGISTRATION AND APPROVAL STEP ##########
print('Model registration Step ..')

model = HuggingFaceModel(
    model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
    role=role,
    transformers_version="4.6.1",
    pytorch_version="1.7.1",
    py_version="py36",
    sagemaker_session=sagemaker_session,
)

from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.workflow.step_collections import RegisterModel

model_metrics = ModelMetrics(
    model_statistics=MetricsSource(
        s3_uri="{}/evaluation.json".format(
            step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
        ),
        content_type="application/json",
    )
)
step_register = RegisterModel(
    name="RegMLNBRegisterModel",
    model=model,
    content_types=["application/json"],
    response_types=["application/json"],
    inference_instances=["ml.g4dn.xlarge", "ml.m5.xlarge"],
    transform_instances=["ml.g4dn.xlarge", "ml.m5.xlarge"],
    model_package_group_name=model_package_group_name,
    approval_status= 'Approved', #model_approval_status
    model_metrics=model_metrics,
)

########## LAMBDA STEP FOR ENDPOINT CREATION ##########
print('Lambda Step ..')

current_time = time.strftime("%m-%d-%H-%M-%S", time.localtime())
model_name = "RegMLNB-model" + current_time
endpoint_config_name = "RegMLNB-endpoint-config" + current_time
endpoint_name = "RegMLNB-endpoint-" + current_time
function_name = "RegMLNB-lambda-step" + current_time

# Lambda helper class can be used to create the Lambda function
func = Lambda(
    function_name=function_name,
    execution_role_arn=lambda_role,
    script="lambda_deployer.py",
    handler="lambda_deployer.lambda_handler",
    timeout=600,
    memory_size=10240,
)

# The dictionary retured by the Lambda function is captured by LambdaOutput, each key in the dictionary corresponds to a
# LambdaOutput

output_param_1 = LambdaOutput(output_name="statusCode", output_type=LambdaOutputTypeEnum.String)
output_param_2 = LambdaOutput(output_name="body", output_type=LambdaOutputTypeEnum.String)
output_param_3 = LambdaOutput(output_name="other_key", output_type=LambdaOutputTypeEnum.String)

# The inputs provided to the Lambda function can be retrieved via the `event` object within the `lambda_handler` function
# in the Lambda
step_deploy_lambda = LambdaStep(
    name="LambdaStepRegMLNBDeploy",
    lambda_func=func,
    inputs={
        "model_name": model_name,
        "endpoint_config_name": endpoint_config_name,
        "endpoint_name": endpoint_name,
        "model_package_arn": step_register.steps[0].properties.ModelPackageArn,
        "role": role,
        "data_capture_destination": "{}/datacapture".format(output_destination)
    },
    outputs=[output_param_1, output_param_2, output_param_3],
)

########## BATCH INFERENCE STEP ##########
print('Batch Inference Step ..')

# Upload batch.py script to S3
s3_client.upload_file(Filename='./RegMLNB/batch.py', Bucket=default_bucket, Key=f'{s3_prefix}/code/batch.py')
batch_script_uri = f's3://{default_bucket}/{s3_prefix}/code/batch.py'

# Upload example batch inputs data
s3_client.upload_file(Filename='./data/batch_inputs.json', Bucket=default_bucket, Key=f'{s3_prefix}/data/batch_inputs.json')
batch_input_uri = f's3://{default_bucket}/{s3_prefix}/data/batch_inputs.json'

# SKLearnProcessor to run batch inference job
batch_processor = SKLearnProcessor(
    framework_version='0.23-1',
    role=role,
    instance_type='ml.m5.xlarge',
    instance_count=1,
    volume_size_in_gb=60,
    base_job_name='batch-transformer',
    sagemaker_session=sagemaker_session)

# Processing step to execute batch inference script in processing container
step_batch_transform = ProcessingStep(
    name='BatchTransformStep',
    processor=batch_processor,
    job_arguments=[
        "--model-name", model_name,
        "--batch-input-uri", batch_input_uri,
        "--region", region,
    ],
    code=batch_script_uri,
    depends_on=[step_deploy_lambda])

########## MODEL QUALITY MONITOR STEP ##########
print("Model Quality Monitor step...")

# Model Quality Monitor
from sagemaker import get_execution_role, session, Session
from sagemaker.model_monitor import ModelQualityMonitor
from sagemaker.model_monitor.dataset_format import DatasetFormat
from datetime import datetime

# SKLearnProcessor to run model quality monitor job
model_quality_processor = SKLearnProcessor(
    framework_version='0.23-1',
    role=role,
    instance_type='ml.m5.xlarge',
    instance_count=1,
    volume_size_in_gb=60,
    base_job_name='model-quality',
    sagemaker_session=sagemaker_session)

# Upload fake_train_data.csv to fit baselining job
s3_client.upload_file(Filename='./data/fake_train_data.csv', Bucket=default_bucket, Key=f'{s3_prefix}/model-monitor/fake_train_data.csv')
baseline_dataset_uri = f's3://{default_bucket}/{s3_prefix}/model-monitor/fake_train_data.csv'

ground_truth_upload_path = (
    f"s3://{default_bucket}/{s3_prefix}/ground-truth-data"
)

# Processing step to execute batch inference script in processing container
step_model_quality = ProcessingStep(
    name='ModelQualityScheduleStep',
    processor=model_quality_processor,
    job_arguments=[
        "--endpoint-name", endpoint_name,
        "--baseline-dataset-uri", baseline_dataset_uri,
        "--ground-truth-upload-path", ground_truth_upload_path,
        "--role", role,
        "--region", region,
    ],
    code="./RegMLNB/monitor.py",
    depends_on=[step_deploy_lambda])

########## LAMBDA STEP TO INVOKE ENDPOINT ##########
print('Lambda Step ..')

# Lambda helper class can be used to create the Lambda function
invoke_func = Lambda(
    function_name="RegMLNB-lambda-step-invoke" + current_time,
    execution_role_arn=lambda_role,
    script="./RegMLNB/invoke.py",
    handler="invoke.lambda_handler",
    timeout=600,
    memory_size=10240,
)

# The dictionary retured by the Lambda function is captured by LambdaOutput, each key in the dictionary corresponds to a
# LambdaOutput
output_param_1 = LambdaOutput(output_name="statusCode", output_type=LambdaOutputTypeEnum.String)
output_param_2 = LambdaOutput(output_name="body", output_type=LambdaOutputTypeEnum.String)

# The inputs provided to the Lambda function can be retrieved via the `event` object within the `lambda_handler` function
# in the Lambda
step_lambda_invoke = LambdaStep(
    name="LambdaStepEndpointInvoke",
    lambda_func=invoke_func,
    inputs={
        "endpoint_name": endpoint_name,
        "ground_truth_upload_path": ground_truth_upload_path
    },
    outputs=[output_param_1, output_param_2],
    depends_on=[step_model_quality]
)

########## CONDITION STEP TO CHECK MODEL ACCURACY AND CONDITIONALLY CREATE A MODEL AND REGISTER IN MODEL REGISTRY ##########

print('Conditional Step ..')

# Condition step for evaluating model accuracy and branching execution
cond_accuracy = ConditionGreaterThanOrEqualTo(  # You can change the condition here
    left=JsonGet(
        step=step_eval,
        property_file=evaluation_report,
        json_path="binary_classification_metrics.accuracy.value",  # This should follow the structure of your report_dict defined in the evaluate.py file.

    ),
    right=0.60,  # You can change the threshold here
)

# Condition step for evaluating PRAUC (more robust measure than accuracy for imbalanced classification)
cond_prauc = ConditionGreaterThanOrEqualTo(  # You can change the condition here
    left=JsonGet(
        step=step_eval,
        property_file=evaluation_report,
        json_path="binary_classification_metrics.pr_auc.value",  # This should follow the structure of your report_dict defined in the evaluate.py file.

    ),
    right=0.70,  # You can change the threshold here
)

step_cond = ConditionStep(
    name="CheckRegMLNBEvalAccuracy",
    conditions=[cond_accuracy, cond_prauc],
    if_steps=[step_register,step_deploy_lambda,step_batch_transform,step_model_quality,step_lambda_invoke], #
    else_steps=[],
)

########## DEFINE PIPELINE ##########

try: 
    pipeline_name = f"RegMLNBPipeline"
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            processing_instance_count,
            training_instance_type,
            training_instance_count,
            model_approval_status,
    #            input_data,
        ],
        steps=[step_process,step_train,step_eval,step_cond], 
        sagemaker_session=sagemaker_session,
    )

    definition = json.loads(pipeline.definition())

    #submit the pipeline to sagemaker and start execution
    upsert_response = pipeline.upsert(role_arn=role) 
    print("\n###### Created/Updated SageMaker Pipeline: Response received:")
    print(upsert_response)

    execution = pipeline.start()
    print(f"\n###### Execution started with PipelineExecutionArn: {execution.arn}")

    print("Waiting for the execution to finish...")
    execution.wait() # wait
    print("\n#####Execution completed. Execution step details:")
    print(execution.list_steps())

    
except Exception as e:  
    print(f"Exception: {e}")
    sys.exit(1)




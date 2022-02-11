from aws_cdk import (
    aws_lambda as _lambda,
    aws_s3 as s3,
    aws_s3_deployment as s3deploy,
    aws_sagemaker as sagemaker,
    aws_iam as iam,
    core
)

from .role_stack import RegMLRoleStack

class RegMLStack(core.Stack):

    def __init__(self, scope: core.Construct, construct_id: str) -> None:
        super().__init__(scope, construct_id)
        
        RegMLRoleStack(self, 'RegMLRole')

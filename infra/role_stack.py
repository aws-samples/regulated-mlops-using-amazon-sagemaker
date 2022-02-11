from aws_cdk import (
    aws_sagemaker as sagemaker,
    aws_iam as iam,
    core
)
import os


class RegMLRoleStack(core.Construct):

    def __init__(self, scope: core.Construct, construct_id: str) -> None: #, bucket_name: str, **kwargs
        super().__init__(scope, construct_id)

        self._PREFIX = construct_id

        # Create Role for the SageMaker Backend
        self._service_role = iam.Role(
            self, f'{self._PREFIX}-ServiceRole',
            role_name=f'{self._PREFIX}-ServiceRole',
            assumed_by=iam.CompositePrincipal(
                iam.ServicePrincipal("sagemaker.amazonaws.com"),
                iam.ServicePrincipal("lambda.amazonaws.com"),
                iam.ServicePrincipal("s3.amazonaws.com")
            )
        )
        self._service_role.add_managed_policy(iam.ManagedPolicy.from_aws_managed_policy_name('AWSCodeCommitPowerUser'))
        self._service_role.add_managed_policy(iam.ManagedPolicy.from_aws_managed_policy_name('AmazonSageMakerFullAccess'))
        self._service_role.add_managed_policy(iam.ManagedPolicy.from_aws_managed_policy_name('AmazonS3FullAccess'))
        self._service_role.add_managed_policy(iam.ManagedPolicy.from_aws_managed_policy_name('AWSLambda_FullAccess'))
        self._service_role.add_managed_policy(iam.ManagedPolicy.from_aws_managed_policy_name('CloudWatchEventsFullAccess'))

        core.CfnOutput(
            self,
            id="ServiceRoleARN",
            value=self._service_role.role_arn,
            description="ARN for service role created by RegML Stack"
        )
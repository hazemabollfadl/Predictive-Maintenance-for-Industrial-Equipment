"""
Create or update the SageMaker endpoint.

- If the endpoint doesn't exist yet: creates it from scratch.
- If it already exists: updates it in-place (zero-downtime blue/green swap).

Run: python deploy/scripts/deploy.py
Requires env vars: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION,
                   S3_BUCKET, SAGEMAKER_ROLE_ARN
"""

import os

import boto3
import yaml
from sagemaker.pytorch import PyTorchModel


def main():
    with open("deploy/config.yaml") as f:
        cfg = yaml.safe_load(f)

    s3_bucket = os.environ["S3_BUCKET"]
    role_arn = os.environ["SAGEMAKER_ROLE_ARN"]
    region = os.environ["AWS_REGION"]

    model_uri = f"s3://{s3_bucket}/{cfg['s3']['prefix']}/model.tar.gz"
    endpoint_name = cfg["sagemaker"]["endpoint_name"]

    model = PyTorchModel(
        model_data=model_uri,
        role=role_arn,
        framework_version=cfg["model"]["framework_version"],
        py_version=cfg["model"]["python_version"],
        entry_point="inference.py",
    )

    sm_client = boto3.client("sagemaker", region_name=region)
    existing_endpoints = [
        e["EndpointName"]
        for e in sm_client.list_endpoints()["Endpoints"]
    ]

    if endpoint_name in existing_endpoints:
        print(f"Updating existing endpoint: {endpoint_name}")
        predictor = model.deploy(
            endpoint_name=endpoint_name,
            instance_type=cfg["sagemaker"]["instance_type"],
            initial_instance_count=cfg["sagemaker"]["initial_instance_count"],
            update_endpoint=True,
        )
    else:
        print(f"Creating new endpoint: {endpoint_name}")
        predictor = model.deploy(
            endpoint_name=endpoint_name,
            instance_type=cfg["sagemaker"]["instance_type"],
            initial_instance_count=cfg["sagemaker"]["initial_instance_count"],
        )

    print(f"Endpoint live: {predictor.endpoint_name}")


if __name__ == "__main__":
    main()

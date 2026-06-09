"""
Integration test — hits the live SageMaker endpoint.

Skipped automatically when SAGEMAKER_ENDPOINT_NAME is not set,
so it never blocks local development or forks without secrets.

Run locally:
    export SAGEMAKER_ENDPOINT_NAME=predictive-maintenance-rul
    export AWS_REGION=us-east-1
    pytest src/tests/integration -v
"""

import json
import os

import boto3
import numpy as np
import pytest


ENDPOINT_NAME = os.getenv("SAGEMAKER_ENDPOINT_NAME")


@pytest.fixture(scope="module")
def runtime_client():
    return boto3.client("sagemaker-runtime", region_name=os.environ["AWS_REGION"])


@pytest.mark.skipif(not ENDPOINT_NAME, reason="SAGEMAKER_ENDPOINT_NAME not set")
def test_healthy_engine_returns_low_probability(runtime_client):
    """A window of near-zero sensor drift should score below 0.5."""
    window = np.random.uniform(low=-0.1, high=0.1, size=(30, 15)).tolist()
    payload = json.dumps({"window": window})

    response = runtime_client.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=payload,
    )
    result = json.loads(response["Body"].read())

    assert "failure_probability" in result
    assert "alert" in result
    assert 0.0 <= result["failure_probability"] <= 1.0


@pytest.mark.skipif(not ENDPOINT_NAME, reason="SAGEMAKER_ENDPOINT_NAME not set")
def test_wrong_input_shape_returns_error(runtime_client):
    """Sending a (10, 15) window instead of (30, 15) must return a 4xx error."""
    window = np.random.rand(10, 15).tolist()   # wrong window size
    payload = json.dumps({"window": window})

    with pytest.raises(runtime_client.exceptions.ModelError):
        runtime_client.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Body=payload,
        )

"""
Lambda handler — two routes:
  GET  /baselines  → returns pre-computed sensor baselines
  POST /predict    → forwards sensor window to SageMaker, returns predicted RUL
"""

import json
import os
import boto3

with open(os.path.join(os.path.dirname(__file__), "baselines.json")) as f:
    BASELINES = json.load(f)

ENDPOINT_NAME = os.environ["SAGEMAKER_ENDPOINT_NAME"]
sagemaker = boto3.client("sagemaker-runtime")


def handler(event, context):
    method = event.get("requestContext", {}).get("http", {}).get("method", "")
    path   = event.get("rawPath", "")

    if method == "GET" and path == "/baselines":
        return _respond(200, BASELINES)

    if method == "POST" and path == "/predict":
        body = json.loads(event.get("body", "{}"))
        window = body.get("window")

        if not window:
            return _respond(400, {"error": "missing 'window' in request body"})

        response = sagemaker.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Body=json.dumps({"window": window}),
        )
        result = json.loads(response["Body"].read())
        return _respond(200, result)

    return _respond(404, {"error": "not found"})


def _respond(status_code, body):
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
        "body": json.dumps(body),
    }

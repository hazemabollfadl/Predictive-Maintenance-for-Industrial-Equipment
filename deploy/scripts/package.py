"""
Bundle model artifacts into model.tar.gz and upload to S3.

SageMaker expects a tar.gz with this flat structure:
  model.tar.gz
    ├── cnn_FD001.pt
    ├── scaler_FD001.pkl
    └── inference.py          ← entry point SageMaker looks for

Run: python deploy/scripts/package.py
Requires env vars: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, S3_BUCKET
"""

import os
import tarfile
import tempfile

import boto3
import yaml


def main():
    with open("deploy/config.yaml") as f:
        cfg = yaml.safe_load(f)

    weights_path = cfg["model"]["weights_path"]
    scaler_path = cfg["model"]["scaler_path"]
    s3_prefix = cfg["s3"]["prefix"]
    s3_bucket = os.environ["S3_BUCKET"]

    for path in (weights_path, scaler_path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. Train the model and save artifacts before deploying."
            )

    with tempfile.TemporaryDirectory() as tmp:
        archive_path = os.path.join(tmp, "model.tar.gz")

        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(weights_path, arcname="cnn_FD001.pt")
            tar.add(scaler_path, arcname="scaler_FD001.pkl")
            tar.add("deploy/inference.py", arcname="inference.py")

        s3_key = f"{s3_prefix}/model.tar.gz"
        boto3.client("s3").upload_file(archive_path, s3_bucket, s3_key)
        print(f"Uploaded model.tar.gz → s3://{s3_bucket}/{s3_key}")


if __name__ == "__main__":
    main()

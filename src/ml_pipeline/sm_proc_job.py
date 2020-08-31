# sm_proc_job.py
import sagemaker
from sagemaker.processing import ScriptProcessor, ProcessingInput
import boto3
import os
import sys
from time import gmtime, strftime
from airflow.models import Variable


def sm_proc_job(role, sess, bucket, spark_repo_uri, base_job_name, **context):
    timestamp_prefix = Variable.get("timestamp")

    prefix = "sagemaker/spark-preprocess/"
    input_prefix = prefix + "inputs/raw/abalone"
    input_preprocessed_prefix = prefix + \
        "/inputs/preprocessed/abalone/" + timestamp_prefix
    model_prefix = prefix + "model/spark/" + timestamp_prefix

    spark_processor = ScriptProcessor(base_job_name=base_job_name,
                                      image_uri=spark_repo_uri,
                                      command=["/opt/program/submit"],
                                      role=role,
                                      sagemaker_session=sagemaker.Session(
                                          sess),
                                      instance_count=2,
                                      instance_type="ml.r5.xlarge",
                                      max_runtime_in_seconds=1200,
                                      env={"mode": "python"})
    code_uri = "s3://"+bucket+"/code/smprocpreprocess.py"

    spark_processor.run(code=code_uri, arguments=["s3_input_bucket", bucket, "s3_input_key_prefix",
                                                  input_prefix, "s3_output_bucket",
                                                  bucket, "s3_output_key_prefix",
                                                  input_preprocessed_prefix, "s3_model_bucket",
                                                  bucket, "s3_model_prefix", model_prefix], logs=True)

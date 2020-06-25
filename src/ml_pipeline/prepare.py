from airflow.models import Variable
from time import gmtime, strftime
import boto3


def start(bucket, keys, file_paths):
    timestamp_prefix = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    Variable.set("timestamp", timestamp_prefix)

    s3 = boto3.client('s3')
    input_key = keys[0]
    input_file = file_paths[0]
    preproc_key = keys[1]
    preproc_file = file_paths[1]
    s3.upload_file(Filename=input_file, Bucket=bucket, Key=input_key)
    s3.upload_file(Filename=preproc_file, Bucket=bucket, Key=preproc_key)

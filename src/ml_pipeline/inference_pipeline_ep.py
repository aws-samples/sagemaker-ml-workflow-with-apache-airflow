import sagemaker
from sagemaker.pipeline import PipelineModel
from sagemaker.sparkml.model import SparkMLModel
from sagemaker.model import Model
import boto3
import os
import sys
from . import schema_utils
from sagemaker.amazon.amazon_estimator import get_image_uri
from airflow.models import Variable


def inference_pipeline_ep(role, sess, spark_model_uri, region, **context):
    timestamp_prefix = Variable.get("timestamp")
    # sm = boto3.client('sagemaker', region_name=region)
    # s3client = boto3.client('s3', region_name=region)

    s3_sparkml_data_uri = spark_model_uri

    # s3_xgb_objects = s3client.list_objects_v2(Bucket=bucket, StartAfter='sagemaker/spark-preprocess/model/xgboost/')
    # obj_list = s3_xgb_objects['Contents']
    # obj_list.sort(key = lambda x:x['LastModified'], reverse=False)
    # xgboost_model_latest = obj_list[-1]['Key']
    #s3_xgboost_model_uri = 's3://' + bucket + '/' + xgboost_model_latest

    s3_xgboost_model_uri = context['task_instance'].xcom_pull(
        task_ids='xgboost_model_training')['Training']['ModelArtifacts']['S3ModelArtifacts']

    xgb_container = get_image_uri(
        sess.region_name, 'xgboost', repo_version='0.90-1')

    schema_json = schema_utils.abalone_schema()

    sparkml_model = SparkMLModel(model_data=s3_sparkml_data_uri,  role=role,
                                 sagemaker_session=sagemaker.session.Session(
                                     sess),
                                 env={'SAGEMAKER_SPARKML_SCHEMA': schema_json})

    xgb_model = Model(model_data=s3_xgboost_model_uri, role=role,
                      sagemaker_session=sagemaker.session.Session(sess), image=xgb_container)

    pipeline_model_name = 'inference-pipeline-spark-xgboost-' + timestamp_prefix

    sm_model = PipelineModel(name=pipeline_model_name,
                             role=role,
                             sagemaker_session=sagemaker.session.Session(
                                 sess),
                             models=[sparkml_model, xgb_model])

    endpoint_name = 'inference-pipeline-endpoint-' + timestamp_prefix

    sm_model.deploy(initial_instance_count=1,
                    instance_type='ml.c4.xlarge', endpoint_name=endpoint_name)

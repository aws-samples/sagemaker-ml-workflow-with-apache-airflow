import sagemaker
from sagemaker.pipeline import PipelineModel
from sagemaker.sparkml.model import SparkMLModel
from sagemaker.model import Model
import boto3
import os
import sys
import schema_utils
from sagemaker.amazon.amazon_estimator import get_image_uri
from airflow.models import Variable

sm = boto3.client('sagemaker', region_name='us-east-1')
#timestamp_prefix = Variable.get("timestamp")


def inference_pipeline_ep(role, sess, spark_model_uri, bucket, **context):
    timestamp_prefix = Variable.get("timestamp")
    s3_sparkml_data_uri = spark_model_uri
    s3_xgboost_model = sm.list_training_jobs(MaxResults=1, StatusEquals='Completed', SortBy='CreationTime',
                                             NameContains='training-job-', SortOrder='Descending')['TrainingJobSummaries'][0]['TrainingJobName']

    s3_xgboost_model_uri = 's3://'+bucket+'/sagemaker/spark-preprocess/model/xgboost/' + \
        s3_xgboost_model+'/output/model.tar.gz'

    xgb_container = get_image_uri(
        sess.region_name, 'xgboost', repo_version='0.90-1')

    schema_json = schema_utils.abalone_schema()

    sparkml_model = SparkMLModel(model_data=s3_sparkml_data_uri,  role=role, sagemaker_session=sagemaker.session.Session(
        sess), env={'SAGEMAKER_SPARKML_SCHEMA': schema_json})

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

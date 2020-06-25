from __future__ import print_function
import json
import requests
from datetime import datetime

# airflow operators
import airflow
from airflow.models import DAG, Variable
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.python_operator import PythonOperator

# airflow sagemaker operators
from airflow.contrib.operators.sagemaker_training_operator \
    import SageMakerTrainingOperator
from airflow.contrib.operators.sagemaker_tuning_operator \
    import SageMakerTuningOperator
from airflow.contrib.operators.sagemaker_transform_operator \
    import SageMakerTransformOperator
from airflow.contrib.operators.sagemaker_model_operator \
    import SageMakerModelOperator
from airflow.contrib.operators.sagemaker_endpoint_operator \
    import SageMakerEndpointOperator

from airflow.contrib.hooks.aws_hook import AwsHook

# sagemaker sdk
import boto3
import sagemaker
from sagemaker.amazon.amazon_estimator import get_image_uri
from sagemaker.estimator import Estimator
from sagemaker.model import Model
from sagemaker.transformer import Transformer
from sagemaker.pipeline import PipelineModel
from sagemaker.sparkml.model import SparkMLModel

# airflow sagemaker configuration
from sagemaker.workflow.airflow import training_config
from sagemaker.workflow.airflow import model_config
from sagemaker.workflow.airflow import transform_config


# ml workflow specific
from ml_pipeline import inference_pipeline_ep, sm_proc_job, prepare
from time import gmtime, strftime
import config as cfg

# =============================================================================
# functions
# =============================================================================


def get_sagemaker_role_arn(role_name, region_name):
    iam = boto3.client('iam', region_name=region_name)
    response = iam.get_role(RoleName=role_name)
    return response["Role"]["Arn"]


def create_s3_input(s3_data):
    data = sagemaker.session.s3_input(
        s3_data, distribution='FullyReplicated',  content_type='text/csv', s3_data_type='S3Prefix')
    return data


# =============================================================================
# setting up training, tuning and transform configuration
# =============================================================================


# read config file
config = cfg.config

# set configuration for tasks
hook = AwsHook(aws_conn_id='airflow-sagemaker')
region = config["job_level"]["region_name"]
sess = hook.get_session(region_name=region)
role = get_sagemaker_role_arn(
    config["train_model"]["sagemaker_role"],
    sess.region_name)

# create XGB estimator
xgb_container = get_image_uri(
    sess.region_name, 'xgboost', repo_version="0.90-1")

xgb_estimator = Estimator(
    image_name=xgb_container,
    role=role,
    sagemaker_session=sagemaker.session.Session(sess),
    **config["train_model"]["estimator_config"]
)

# train_config specifies SageMaker training configuration

train_data = create_s3_input(
    config['train_model']['inputs']['train'])
validation_data = create_s3_input(
    config['train_model']['inputs']['validation'])
data_channels = {'train': train_data, 'validation': validation_data}

train_config = training_config(
    estimator=xgb_estimator,
    inputs=data_channels)

# Batch inference

xgb_transformer = Transformer(
    model_name=config['batch_transform']['model_name'],
    sagemaker_session=sagemaker.session.Session(sess),
    **config['batch_transform']['transformer_config']
)

transform_config = transform_config(
    transformer=xgb_transformer,
    **config['batch_transform']['transform_config']
)

# =============================================================================
# define airflow DAG and tasks
# =============================================================================
# define airflow DAG
args = {
    'owner': 'airflow',
    'start_date': airflow.utils.dates.days_ago(2)
}

dag = DAG(
    'sagemaker-ml-pipeline',
    default_args=args,
    schedule_interval=None,
    concurrency=1,
    max_active_runs=1,
    user_defined_filters={'tojson': lambda s: json.JSONEncoder().encode(s)}
)

# Set the tasks in the DAG

# Start operator
init = PythonOperator(
    task_id='start_job',
    dag=dag,
    provide_context=False,
    python_callable=prepare.start,
    op_kwargs={'bucket': config['bucket'],
               'keys': config['keys'], 'file_paths': config['file_paths']})

# SageMaker processing job task
sm_proc_job_task = PythonOperator(
    task_id='sm_proc_job',
    dag=dag,
    provide_context=True,
    python_callable=sm_proc_job.sm_proc_job,
    op_kwargs={'role': role, 'sess': sess, 'bucket': config['bucket'], 'spark_repo_uri': config['spark_repo_uri']})

# Train xgboost model task
train_model_task = SageMakerTrainingOperator(
    task_id='xgboost_model_training',
    dag=dag,
    config=train_config,
    aws_conn_id='airflow-sagemaker',
    wait_for_completion=True,
    check_interval=30
)

# Inference pipeline endpoint task
inference_pipeline_task = PythonOperator(
    task_id='inference_pipeline',
    dag=dag,
    python_callable=inference_pipeline_ep.inference_pipeline_ep,
    op_kwargs={'role': role, 'sess': sess,
               'spark_model_uri': config['inference_pipeline']['inputs']['spark_model'], 'bucket': config['bucket']}
)

# launch sagemaker batch transform job and wait until it completes
batch_transform_task = SageMakerTransformOperator(
    task_id='batch_predicting',
    dag=dag,
    config=transform_config,
    aws_conn_id='airflow-sagemaker',
    wait_for_completion=True,
    check_interval=30)

# Cleanup task
cleanup_task = DummyOperator(
    task_id='cleaning_up',
    dag=dag)


init.set_downstream(sm_proc_job_task)
sm_proc_job_task.set_downstream(train_model_task)
train_model_task.set_downstream(inference_pipeline_task)
inference_pipeline_task.set_downstream(batch_transform_task)
batch_transform_task.set_downstream(cleanup_task)

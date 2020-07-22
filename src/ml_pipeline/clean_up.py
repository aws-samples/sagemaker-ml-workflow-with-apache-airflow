# clean_up.py
# Delete SageMaker endpoints
import boto3


def clean_up(region, bucket):
    sm = boto3.client('sagemaker', region_name=region)
    s3 = boto3.client('s3', region_name=region)

    # Get list of inference endpoints
    endpoints_list = sm.list_endpoints(
        NameContains='inference-pipeline-endpoint')

    # Delete sagemaker endpoints
    for endpoint in endpoints_list['Endpoints']:
        sm.delete_endpoint(EndpointName=endpoint['EndpointName'])

    # Get s3 key for xbgoost model artifact
    delete_obj_list = []
    s3_xgb_objects = s3.list_objects_v2(
        Bucket=bucket, StartAfter='sagemaker/spark-preprocess/model/xgboost/')['Contents']

    for obj in s3_xgb_objects:
        delete_obj_list.append({
            'Key': obj['Key']
        })

    s3_spark_objects = s3.list_objects_v2(
        Bucket=bucket, StartAfter='sagemaker/spark-preprocess/model/spark/'
    )['Contents']

    for obj in s3_spark_objects:
        delete_obj_list.append({
            'Key': obj['Key']
        })

    # Delete ALL xgboost and spark model artifacts
    s3.delete_objects(
        Bucket=bucket,
        Delete={
            'Objects': delete_obj_list
        }
    )

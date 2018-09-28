import boto3
import re
import os
import wget
import time
from time import gmtime, strftime
import sys
import json
import sagemaker

start = time.time()

role = sys.argv[1]

from sagemaker.mxnet import MXNet
from sagemaker.tuner import IntegerParameter, CategoricalParameter, ContinuousParameter, HyperparameterTuner

region = boto3.Session().region_name
train_data_location = 's3://sagemaker-sample-data-{}/mxnet/mnist/train'.format(region)
test_data_location = 's3://sagemaker-sample-data-{}/mxnet/mnist/test'.format(region)

estimator = MXNet(entry_point='mnist.py',
                  role=role,
                  train_instance_count=1,
                  train_instance_type='ml.m4.xlarge',
                  sagemaker_session=sagemaker.Session(),
                  base_job_name='DEMO-hpo-mxnet',
                  hyperparameters={'batch_size': 100})

hyperparameter_ranges = {'optimizer': CategoricalParameter(['sgd', 'Adam']),
                         'learning_rate': ContinuousParameter(0.01, 0.2),
                         'num_epoch': IntegerParameter(10, 50)}

objective_metric_name = 'Validation-accuracy'
metric_definitions = [{'Name': 'Validation-accuracy',
                       'Regex': 'Validation-accuracy=([0-9\\.]+)'}]

tuner = HyperparameterTuner(estimator,
                            objective_metric_name,
                            hyperparameter_ranges,
                            metric_definitions,
                            max_jobs=9,
                            max_parallel_jobs=3)

tuner.fit({'train': train_data_location, 'test': test_data_location})

print(tuner)

sagemaker = boto3.client(service_name='sagemaker')
status = sagemaker.describe_hyper_parameter_tuning_job(
    HyperParameterTuningJobName=tuner.latest_tuning_job.job_name)['HyperParameterTuningJobStatus']

print('HPO job current status: {}'.format(status))

try:
    # wait for the job to finish and report the ending status
    sagemaker.get_waiter('tuning_job_completed_or_stopped').wait(HyperParameterTuningJobName=tuner.latest_tuning_job.job_name)
    status = sagemaker.describe_hyper_parameter_tuning_job(
        HyperParameterTuningJobName=tuner.latest_tuning_job.job_name)['HyperParameterTuningJobStatus']
    print("Training job ended with status: " + status)
except:
    print('Training failed to start')
     # if exception is raised, that means it has failed
    message = sagemaker.describe_hyper_parameter_tuning_job(
        HyperParameterTuningJobName=tuner.latest_tuning_job.job_name)['FailureReason']
    print('Training failed with the following error: {}'.format(message))

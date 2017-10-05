#!/usr/bin/env bash

export BASE_JOB_NAME=mltest_sonar_20171005_193554
export JOB_NAME=mltest_sonar_20171005_193554

export GCS_JOB_DIR=JOB_DIR=gs://dextest/$JOB_NAME

gcloud ml-engine models create $BASE_JOB_NAME --regions us-central1

export sampleJson=gs://dextest/mltest_sonar/sampleOutput.sampledata.R.json


export MODEL_BINARIES=gs://dextest/mltest_sonar_20171005_193554/export
PACKAGE_STAGING_LOCATION=gs://dextest
read -rsp $'Press enter to continue...\n'
gsutil ls  $MODEL_BINARIES

echo create model
gcloud ml-engine versions create v1 --model $BASE_JOB_NAME --origin $MODEL_BINARIES --runtime-version 1 --staging-bucket $PACKAGE_STAGING_LOCATION


echo run prediction
gcloud ml-engine predict --model $BASE_JOB_NAME --version v1 --json-instances $sampleJson
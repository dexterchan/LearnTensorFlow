#!/usr/bin/env bash

export BASE_JOB_NAME=sonar_prediction
now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="$BASE_JOB_NAME$now"



export GCS_JOB_DIR=gs://dextest/mltest_sonar
export BASE_JOB_NAME=mltest_sonar_20171005_193554


export sampleJson=gs://dextest/mltest_sonar/sampleOutput.sampledata.R.json

#--runtime-version 1.2 \
gcloud ml-engine jobs submit prediction $JOB_NAME \
    --model $BASE_JOB_NAME \
    --version v1 \
    --data-format TEXT \
    --region us-central1 \
    --input-paths $sampleJson \
    --output-path $GCS_JOB_DIR/predictions

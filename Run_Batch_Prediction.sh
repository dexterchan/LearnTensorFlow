#!/usr/bin/env bash

export JOB_NAME=sonar_prediction

export GCS_JOB_DIR=gs://dextest/mltest_census
export BASE_JOB_NAME=census_20171005_163230

#--runtime-version 1.2 \
gcloud ml-engine jobs submit prediction $JOB_NAME \
    --model $BASE_JOB_NAME \
    --version v1 \
    --data-format TEXT \
    --region us-central1 \
    --input-paths gs://cloudml-public/testdata/prediction/census.json \
    --output-path $GCS_JOB_DIR/predictions

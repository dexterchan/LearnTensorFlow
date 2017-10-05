#!/usr/bin/env bash


TRAIN_FILE=gs://dextest/mltest_sonar/sampleOutput.train.csv
EVAL_FILE=gs://dextest/mltest_sonar/sampleOutput.test.csv




export OUTPUT_DIR=sonar_output

export GCS_JOB_DIR=gs://dextest/mltest_sonar_ex
export BASE_JOB_NAME=mltest_sonar
export TRAIN_STEPS=1000

now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="$BASE_JOB_NAME$now"
#export JOB_NAME=mltest_sonar

rm -rf $OUTPUT_DIR

#source activate tensorflow

gcloud ml-engine jobs submit training $JOB_NAME \
                                    --stream-logs \
                                    --runtime-version 1.2 \
                                    --job-dir $GCS_JOB_DIR \
                                    --module-name trainer.task \
                                    --package-path trainer/ \
                                    --region us-central1 \
                                    -- \
                                    --train-files $TRAIN_FILE \
                                    --eval-files $EVAL_FILE \
                                    --train-steps $TRAIN_STEPS \
                                    --eval-steps 100

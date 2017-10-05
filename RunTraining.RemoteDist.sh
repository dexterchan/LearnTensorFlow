#!/usr/bin/env bash

TRAIN_FILE=gs://dextest/mltest_sonar/sampleOutput.train.csv
EVAL_FILE=gs://dextest/mltest_sonar/sampleOutput.test.csv





#export GCS_JOB_DIR=gs://dextest/mltest_census_dist
#export GCS_JOB_PACK_DIR=gs://dextest/mltest_census_dist_pack
#export BASE_JOB_NAME=mltest_sonar
export TRAIN_STEPS=1000

#now=$(date +"%Y%m%d_%H%M%S")
#JOB_NAME="$BASE_JOB_NAME$now"



TRAINER_PACKAGE_PATH=trainer/
now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME=mltest_sonar_$now
MAIN_TRAINER_MODULE=trainer.task
JOB_DIR=gs://dextest/$JOB_NAME
PACKAGE_STAGING_LOCATION=gs://dextest
REGION=us-central1
RUNTIME_VERSION=1.2


#export TF_CONFIG={"cluster":}#--scale-tier $SCALE_TIER \
echo  $JOB_DIR
export SCALE_TIER=STANDARD_1

echo gcloud ml-engine jobs submit training $JOB_NAME \
                                    --stream-logs \
                                    --runtime-version $RUNTIME_VERSION \
                                    --job-dir $JOB_DIR \
                                    --module-name $MAIN_TRAINER_MODULE \
                                    --package-path $TRAINER_PACKAGE_PATH \
                                    --region $REGION \
                                    -- \
                                    --train-files $TRAIN_FILE \
                                    --eval-files $EVAL_FILE \
                                    --train-steps $TRAIN_STEPS \
                                    --eval-steps 100 \
                                    --scale-tier STANDARD_1 \
                                    --staging-bucket $PACKAGE_STAGING_LOCATION
#--scale-tier STANDARD_1 \
gcloud ml-engine jobs submit training $JOB_NAME \
                                    --stream-logs \
                                    --runtime-version $RUNTIME_VERSION \
                                    --job-dir $JOB_DIR \
                                    --module-name $MAIN_TRAINER_MODULE \
                                    --package-path $TRAINER_PACKAGE_PATH \
                                    --region $REGION \
                                    -- \
                                    --train-files $TRAIN_FILE \
                                    --eval-files $EVAL_FILE \
                                    --train-steps $TRAIN_STEPS \
                                    --eval-steps 100 \
                                    --staging-bucket $PACKAGE_STAGING_LOCATION


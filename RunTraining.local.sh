
#!/usr/bin/env bash

#https://storage.cloud.google.com/dextest/mltest_sonar/sonar.all-data.csv?_ga=2.171772389.-456602160.1501671706

TRAIN_FILE=gs://dextest/mltest_sonar/sampleOutput.train.csv
EVAL_FILE=gs://dextest/mltest_sonar/sampleOutput.test.csv




export TRAIN_STEPS=1000
export OUTPUT_DIR=sonar_output
rm -rf $OUTPUT_DIR

#source activate tensorflow
#Activate Python 2.7 tensorflow installed locally
source /Users/dexter/tensorflow_py2/bin/activate

#only run with Python 2.7
#gcloud ml-engine local train --package-path trainer \
#                           --module-name trainer.task \
#                           -- \
#                           --train-files $TRAIN_FILE \
#                           --eval-files $EVAL_FILE \
#                           --job-dir $OUTPUT_DIR \
#                           --train-steps $TRAIN_STEPS \
#                           --eval-steps 100


python trainer/task.py --train-files $TRAIN_FILE \
                       --eval-files $EVAL_FILE \
                       --job-dir $OUTPUT_DIR \
                       --train-steps $TRAIN_STEPS \
                       --eval-steps 1
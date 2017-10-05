from trainer.InputSchema import *
import model
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--output-files',
                        required=True,
                        #type=str,
                        default=None,
                        help='Training output file')



#from model import data_p

from model import prepareDataSet
parse_args, unknown = parser.parse_known_args()
trainfileName,testfileName=prepareDataSet("/Users/dexter/sandbox/LearnAI/LearnTensorFlow/sonar.all-data.csv",parse_args.output_files)

trainfileName=parse_args.output_files+".train.csv"
testfileName=parse_args.output_files+".test.csv"





numOfEpoch=10
trainArg={}

trainArg["train_files"]=[trainfileName]
trainArg["eval_files"]=[testfileName]
trainArg["job_dir"]="sonar_output"
trainArg["train_steps"]=numOfEpoch
trainArg["eval_steps"]=100
trainArg["reuse_job_dir"]=False
trainArg["train_batch_size"]=40
trainArg["eval_batch_size"]=40
trainArg["learning_rate"]=0.3 #0.03 #learning rate for gradient decent
trainArg["eval_frequency"]=50 #perform once evaluation per n steps
trainArg["first_layer_size"]=256
trainArg["num_layers"]=4
trainArg["scale_factor"]=0.25
trainArg["num_epochs"]=None
trainArg["export_format"]=model.JSON


from  trainer.task import *

tf.logging.set_verbosity("DEBUG")
# Set C++ Graph Execution level verbosity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(
    tf.logging.__dict__["DEBUG"] / 10)

dispatch(**trainArg)
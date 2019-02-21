# ROOT_PATH = '/home/danial/PycharmProjects/PoemNonPoemDiscriminator/'
import inspect

import os

ROOT_PATH = os.path.abspath(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) + '/../') + '/'

DATA_PATH = ROOT_PATH + 'data/'
BERT_PATH = DATA_PATH + "bert_models/uncased_L-12_H-768_A-12/"
CH_BERT_PATH = DATA_PATH + "bert_models/chinese_L-12_H-768_A-12/"
MODEL_PATH = DATA_PATH + 'temp_models/'
EXPORT_PATH = DATA_PATH + 'exports/'
DATASET_PATH = DATA_PATH + 'dataset/'
OBJ_fILES_PATH = DATA_PATH + 'obj_files/'
LOGS_PATH = DATA_PATH + 'temp_logs/'
TEMP_PATH = DATA_PATH + 'temp_files/'

for path in [DATA_PATH, MODEL_PATH, EXPORT_PATH, DATASET_PATH, OBJ_fILES_PATH, LOGS_PATH, TEMP_PATH]:
    if not os.path.exists(path):
        os.mkdir(path)

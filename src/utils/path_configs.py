# ROOT_PATH = '/home/danial/PycharmProjects/PoemNonPoemDiscriminator/'
import inspect
import platform
import os
from dotenv import load_dotenv

load_dotenv(verbose=True)

__bert_path = os.getenv("MONGO_DB")
assert __bert_path != ''

COMPUTER_NAME = platform.node()

ROOT_PATH = os.path.abspath(os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe()))) + '/../../') + '/'

DATA_PATH = ROOT_PATH + 'data/'
BERT_PATH = __bert_path
MODEL_PATH = DATA_PATH + 'models/'
EXPORT_PATH = DATA_PATH + 'exports/'
TABLE_EXPORT_PATH = DATA_PATH + 'exports/tables/'
FIG_EXPORT_PATH = DATA_PATH + 'exports/figs/'
DATASET_PATH = DATA_PATH + 'dataset/'
OBJ_fILES_PATH = DATA_PATH + 'obj_files/'
LOGS_PATH = DATA_PATH + 'temp_logs/'
TEMP_PATH = DATA_PATH + 'temp_files/'

for path in [DATA_PATH, MODEL_PATH, EXPORT_PATH, TABLE_EXPORT_PATH, FIG_EXPORT_PATH,
             DATASET_PATH, OBJ_fILES_PATH, LOGS_PATH, TEMP_PATH]:
    if not os.path.exists(path):
        os.mkdir(path)

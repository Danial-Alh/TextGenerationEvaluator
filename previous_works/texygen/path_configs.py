import inspect
import os

ROOT_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
DATA_PATH = ROOT_PATH + '/data/'
MODEL_PATH = DATA_PATH + 'saved_models/'

for p in [DATA_PATH, MODEL_PATH]:
    if not os.path.exists(p):
        os.mkdir(p)

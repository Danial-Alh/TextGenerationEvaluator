import os

from ...path_configs import MODEL_PATH

SAVING_PATH = MODEL_PATH + 'pg_bleu/'
if not os.path.exists(SAVING_PATH):
    os.mkdir(SAVING_PATH)

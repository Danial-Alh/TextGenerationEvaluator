import json
import os

from metrics.fbd import FBD
from utils.path_configs import EXPORT_PATH

res = {}

# export text file from sub dir
# https://stackoverflow.com/questions/1228466/how-to-filter-files-when-using-scp-to-copy-dir-recursively
validation_dir = "/tmp/validation/"
generated_dir = "/tmp/generated/"

possible_base_list = ["bleu3", "bleu4", "bleu5", "last_iter"]
possible_model_list = ["leakgan2", "Maligan", "Mle", "Rankgan", "Seqgan"]
for dataset_name in ["coco60", "emnlp60"]:
    res[dataset_name] = {}
    for k in range(3):
        res[dataset_name][k] = {}
        validation_path = os.path.join(validation_dir, "%s-valid-k%d_parsed.txt" % (dataset_name, k))
        validation_text = open(validation_path).read().split("\n")
        assert 19000 < len(validation_text) < 21000, len(validation_text)
        fbd = FBD(validation_text, 64, "data/bert_models/uncased_L-12_H-768_A-12/")
        for model in possible_model_list:
            res[dataset_name][k][model] = {}
            for base in possible_base_list:
                generated_path = os.path.join(generated_dir, dataset_name, "%s_k%d" % (model, k),
                                              "%s_based_samples.txt" % base)
                generated_text = open(generated_path).read().split("\n")
                assert 19000 < len(generated_text) < 21000, len(generated_text)
                fbd_score = fbd.get_score(generated_text)
                res[dataset_name][k][model][base] = fbd_score

export_dir = EXPORT_PATH
export_path = os.path.join(export_dir, "FBD.json")
with open(export_path, 'w') as fp:
    json.dump(res, fp)

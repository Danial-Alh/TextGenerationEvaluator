import json
import os

from export_utils.evaluation_utils import TblGenerator, change_model_name, change_metric_name
from utils.path_configs import EXPORT_PATH


def make_caption(dataset, base):
    dataset_name_dict = {"coco60": "COCO Captions", "emnlp60": "EMNLP2017 WMT News"}
    good_dataset_name = dataset_name_dict[dataset]
    if base == "last_iter":
        suffix = "maximum number of iterations"
    else:
        suffix = "best %s" % change_metric_name(base)
    caption = "Performance of models (using different measures) on %s when training termination criterion is based on the %s." % (
        good_dataset_name, suffix)
    # caption = "Models tarined on \"%s\" data. training stop criterion based on %s." % (good_dataset_name, suffix)
    return caption


possible_base_list = ["bleu3", "bleu4", "bleu5", "last_iter"]

evaluation_dir = EXPORT_PATH
res = {}
fbd_path = os.path.join(evaluation_dir, "FBD.json")
fbd = json.load(open(fbd_path))
for dataset_name in ["coco60", "emnlp60"]:
    res[dataset_name] = {}
    for k in range(3):
        res[dataset_name][k] = {}
        evaluation_path = os.path.join(evaluation_dir,
                                       "evaluations_3k-fold_k%d_%s-train_nsamp20000.json" % (k, dataset_name))
        with open(evaluation_path) as file:
            data = json.load(file)[str(k)]
            for base_type in possible_base_list:
                res[dataset_name][k][base_type] = {}
                for model_name in data[base_type + " restore type"]:
                    good_model_name = change_model_name(model_name)
                    res[dataset_name][k][base_type][good_model_name] = {}
                    for metric_name in data[base_type + " restore type"][model_name]:
                        good_metric_name = change_metric_name(metric_name)
                        elem = data[base_type + " restore type"][model_name][metric_name]
                        res[dataset_name][k][base_type][good_model_name][good_metric_name] = elem

                for model_name in ["leakgan2", "Maligan", "Mle", "Rankgan", "Seqgan"]:
                    good_model_name = change_model_name(model_name)
                    elem = fbd[dataset_name][str(k)][model_name][base_type]
                    res[dataset_name][k][base_type][good_model_name]["FBD"] = elem

metric_names = ["NLL", "FBD"]
metric_names += ["MSJ-%d" % i for i in range(2, 6)]
metric_names += ["BLEU-%d" % i for i in range(2, 6)]
metric_names += ["SBLEU-%d" % i for i in range(2, 6)]

import numpy as np

models_list = ["MLE", "SeqGAN", "RankGAN", "MaliGAN", "LeakGAN"]
for dataset_name in ["coco60", "emnlp60"]:
    for base_type in possible_base_list:
        best_mask = {x: None for x in metric_names}
        best_value = {x: None for x in metric_names}

        for model_name in models_list:
            for metric_name in metric_names:
                tmp = [res[dataset_name][k][base_type][model_name][metric_name] for k in range(3)]
                mu, sigma = np.mean(tmp), np.std(tmp)
                # nll is ll!
                if metric_name == "FBD" or metric_name.startswith("SBLEU"):
                    mu *= -1

                vl = best_value[metric_name]
                if vl is None or mu > vl:
                    best_value[metric_name] = mu
                    best_mask[metric_name] = model_name

        caption = make_caption(dataset_name, base_type)
        tbl = TblGenerator(["Model"] + metric_names, caption, "scriptsize",
                           "table:%s:%s" % (dataset_name, "last" if base_type == "last_iter" else base_type))
        for model_name in models_list:
            metric_results = []
            for metric_name in metric_names:
                tmp = [res[dataset_name][k][base_type][model_name][metric_name] for k in range(3)]
                # metric_results.append("$%.2f$ \\newline $\\pm %.2f$" % (np.mean(tmp), np.std(tmp)))
                mu, sigma = np.mean(tmp), np.std(tmp)

                if metric_name == "NLL":
                    mu *= -1
                s = ""
                if best_mask[metric_name] == model_name:
                    assert abs(best_value[metric_name]) == abs(mu), (best_value[metric_name], mu)
                    s = " \\begin{tabular}{@{}c@{}} $\mathbf{%.3f}$ \\\\ $\mathbf{\pm %.2f}$\\end{tabular} "
                else:
                    s = " \\begin{tabular}{@{}c@{}} $%.3f$ \\\\ $\pm %.2f$\\end{tabular} "

                metric_results.append(s % (mu, sigma))

            tbl.add_row([model_name] + metric_results)
        print(str(tbl))

import json
import os

from export_utils.evaluation_utils import TblGenerator, change_model_name, change_metric_name
from path_configs import EXPORT_PATH

possible_base_list = ["-nll_oracle", "last_iter"]

evaluation_dir = EXPORT_PATH
res = {}

for k in range(3):
    res[k] = {}
    evaluation_path = os.path.join(evaluation_dir,
                                   "oracle_evaluations_3k-fold_k%d_oracle37.5-train_nsamp12500.json" % k)
    with open(evaluation_path) as file:
        data = json.load(file)[str(k)]
        for base_type in possible_base_list:
            res[k][base_type] = {}
            for model_name in data[base_type + " restore type"]:
                good_model_name = change_model_name(model_name)
                res[k][base_type][good_model_name] = {}
                for metric_name in data[base_type + " restore type"][model_name]:
                    good_metric_name = change_metric_name(metric_name)
                    elem = data[base_type + " restore type"][model_name][metric_name]
                    res[k][base_type][good_model_name][good_metric_name] = elem

metric_names = ["NLL", "OracleNLL", "Bhattacharyya", "Jeffreys"]

import numpy as np

models_list = ["MLE", "SeqGAN", "RankGAN", "MaliGAN", "LeakGAN"]

for base_type in possible_base_list:
    best_mask = {x: None for x in metric_names}
    best_value = {x: None for x in metric_names}

    for model_name in models_list:
        for metric_name in metric_names:
            tmp = [res[k][base_type][model_name][metric_name] for k in range(3)]
            mu, sigma = np.mean(tmp), np.std(tmp)
            # nll is ll!
            if metric_name == "Jeffreys" or metric_name == "Bhattacharyya":
                mu *= -1

            vl = best_value[metric_name]
            if vl is None or mu > vl:
                best_value[metric_name] = mu
                best_mask[metric_name] = model_name

    if base_type == "last_iter":
        suffix = "maximum number of iterations"
    elif base_type == "-nll_oracle":
        suffix = "best NLLOracle"
    caption = "Performance of models (using different measures) on synthetic oracle when training termination criterion is based on the %s." % suffix
    # caption = "Models tarined on synthetic data. training stop criterion based on %s." % suffix
    tbl_label_suffix = "last" if base_type == "last_iter" else "nlloracle"
    tbl = TblGenerator(["Model"] + metric_names, caption, "small", "table:synthetic:%s" % tbl_label_suffix)
    for model_name in models_list:
        metric_results = []
        for metric_name in metric_names:
            tmp = [res[k][base_type][model_name][metric_name] for k in range(3)]
            # metric_results.append("$%.2f$ \\newline $\\pm %.2f$" % (np.mean(tmp), np.std(tmp)))
            mu, sigma = np.mean(tmp), np.std(tmp)

            if metric_name == "NLL" or metric_name == "OracleNLL":
                mu *= -1. / 20.
            s = ""
            if best_mask[metric_name] == model_name:
                # assert abs(best_value[metric_name]) == abs(mu), (best_value[metric_name], mu)
                s = " \\begin{tabular}{@{}c@{}} $\mathbf{%.3f}$ \\\\ $\mathbf{\pm %.2f}$\\end{tabular} "
            else:
                s = " \\begin{tabular}{@{}c@{}} $%.3f$ \\\\ $\pm %.2f$\\end{tabular} "

            metric_results.append(s % (mu, sigma))

        tbl.add_row([model_name] + metric_results)
    print(str(tbl))

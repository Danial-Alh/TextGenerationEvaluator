import numpy as np

from export_utils.evaluation_utils import TblGenerator, change_model_name, change_metric_name
from evaluator import Dumper
from models import all_models, create_model, out_side_trained_models

dataset_name_dict = {"coco60": "COCO Captions", "emnlp60": "EMNLP2017 WMT News"}


class RealExporter:
    possible_base_list = ["bleu3", "bleu4", "bleu5", "last_iter"]
    metric_names = ["NLL", "FBD"] + ["MSJ-%d" % i for i in range(2, 6)] + \
                   ["BLEU-%d" % i for i in range(2, 6)] + ["SBLEU-%d" % i for i in range(2, 6)]

    def make_caption(self, dataset, base):
        good_dataset_name = dataset_name_dict[dataset]
        if base == "last_iter":
            suffix = "maximum number of iterations"
        else:
            suffix = "best %s" % change_metric_name(base)
        caption = "Performance of models (using different measures) on %s when training termination criterion is based on the %s." % (
            good_dataset_name, suffix)
        # caption = "Models tarined on \"%s\" data. training stop criterion based on %s." % (good_dataset_name, suffix)
        return caption

    def export_tables(self, dataset_name, inp_possible_base_list):
        if inp_possible_base_list is None:
            inp_possible_base_list = self.possible_base_list
        res = {dataset_name: {}}
        for k in range(3):
            res[dataset_name][k] = {}
            for base_type in inp_possible_base_list:
                res[dataset_name][k][base_type] = {}
                for model_name in all_models:
                    good_model_name = change_model_name(model_name)
                    res[dataset_name][k][base_type][good_model_name] = {}
                    dumper = Dumper(create_model(model_name, None), k, dataset_name)
                    for metric_name, elem in \
                            dumper.load_final_results(base_type if model_name not in out_side_trained_models else
                                                      'last_iter').items():
                        good_metric_name = change_metric_name(metric_name)
                        # elem = data[base_type + " restore type"][model_name][metric_name]
                        res[dataset_name][k][base_type][good_model_name][good_metric_name] = elem

                    # good_model_name = change_model_name(model_name)
                    # elem = fbd[dataset_name][str(k)][model_name][base_type]
                    # res[dataset_name][k][base_type][good_model_name]["FBD"] = elem

        for base_type in self.possible_base_list:
            best_mask = {x: None for x in self.metric_names}
            best_value = {x: None for x in self.metric_names}

            for model_name in all_models:
                for metric_name in self.metric_names:
                    tmp = [res[dataset_name][k][base_type][model_name][metric_name] for k in range(3)]
                    mu, sigma = np.mean(tmp), np.std(tmp)
                    # nll is ll!
                    if metric_name == "FBD" or metric_name.startswith("SBLEU"):
                        mu *= -1

                    vl = best_value[metric_name]
                    if vl is None or mu > vl:
                        best_value[metric_name] = mu
                        best_mask[metric_name] = model_name

            caption = self.make_caption(dataset_name, base_type)
            tbl = TblGenerator(["Model"] + self.metric_names, caption, "scriptsize",
                               "table:%s:%s" % (dataset_name, "last" if base_type == "last_iter" else base_type))
            for model_name in all_models:
                metric_results = []
                for metric_name in self.metric_names:
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


class OracleExporter:
    possible_base_list = ["-nll_oracle", "last_iter"]
    metric_names = ["NLL", "OracleNLL", "Bhattacharyya", "Jeffreys"]

    def make_caption(self, base):
        if base == "last_iter":
            suffix = "maximum number of iterations"
        elif base == "-nll_oracle":
            suffix = "best NLLOracle"
        caption = "Performance of models (using different measures) on synthetic oracle " \
                  "when training termination criterion is based on the %s." % (suffix)
        # caption = "Models tarined on \"%s\" data. training stop criterion based on %s." % (good_dataset_name, suffix)
        return caption

    def export_tables(self, dataset_name, inp_possible_base_list):
        if inp_possible_base_list is None:
            inp_possible_base_list = self.possible_base_list
        res = {}
        for k in range(3):
            res[k] = {}
            for base_type in inp_possible_base_list:
                res[k][base_type] = {}
                for model_name in all_models:
                    good_model_name = change_model_name(model_name)
                    res[k][base_type][good_model_name] = {}
                    dumper = Dumper(create_model(model_name, None), k, dataset_name)
                    for metric_name, elem in dumper.load_final_results(base_type).items():
                        good_metric_name = change_metric_name(metric_name)
                        res[k][base_type][good_model_name][good_metric_name] = elem

        for base_type in self.possible_base_list:
            best_mask = {x: None for x in self.metric_names}
            best_value = {x: None for x in self.metric_names}

            for model_name in all_models:
                for metric_name in self.metric_names:
                    tmp = [res[dataset_name][k][base_type][model_name][metric_name] for k in range(3)]
                    mu, sigma = np.mean(tmp), np.std(tmp)
                    # nll is ll!
                    if metric_name == "Jeffreys" or metric_name == "Bhattacharyya":
                        mu *= -1

                    vl = best_value[metric_name]
                    if vl is None or mu > vl:
                        best_value[metric_name] = mu
                        best_mask[metric_name] = model_name

            caption = self.make_caption(base_type)
            tbl_label_suffix = "last" if base_type == "last_iter" else "nlloracle"
            tbl = TblGenerator(["Model"] + self.metric_names, caption, "small", "table:synthetic:%s" % tbl_label_suffix)
            for model_name in all_models:
                metric_results = []
                for metric_name in self.metric_names:
                    tmp = [res[dataset_name][k][base_type][model_name][metric_name] for k in range(3)]
                    # metric_results.append("$%.2f$ \\newline $\\pm %.2f$" % (np.mean(tmp), np.std(tmp)))
                    mu, sigma = np.mean(tmp), np.std(tmp)

                    if metric_name == "NLL" or metric_name == "OracleNLL":
                        mu *= -1. / 20.
                    s = ""
                    if best_mask[metric_name] == model_name:
                        assert abs(best_value[metric_name]) == abs(mu), (best_value[metric_name], mu)
                        s = " \\begin{tabular}{@{}c@{}} $\mathbf{%.3f}$ \\\\ $\mathbf{\pm %.2f}$\\end{tabular} "
                    else:
                        s = " \\begin{tabular}{@{}c@{}} $%.3f$ \\\\ $\pm %.2f$\\end{tabular} "

                    metric_results.append(s % (mu, sigma))

                tbl.add_row([model_name] + metric_results)
            print(str(tbl))


def export(training_mode, dataset_name, base_types):
    exporter = RealExporter if training_mode == 'real' else OracleExporter
    exporter = exporter()
    exporter.export_tables(dataset_name, base_types)

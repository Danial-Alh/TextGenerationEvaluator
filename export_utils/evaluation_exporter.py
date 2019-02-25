import numpy as np

from evaluator import Dumper
from export_utils.evaluation_utils import make_caption, TblGenerator
from models import create_model
from utils.file_handler import write_text
from utils.path_configs import TABLE_EXPORT_PATH

new_model_name = {"dgsan": "DGSAN",
                  "mle": "MLE",
                  "newmle": "NewMLE",
                  "real": "Train Data",
                  "textgan": "TextGAN",
                  "leakgan": "LeakGAN",
                  "maligan": "MaliGAN",
                  "rankgan": "RankGAN",
                  "seqgan": "SeqGAN"}
model_name_orders = ['real', 'mle', 'newmle', 'seqgan', 'maligan', 'rankgan', 'textgan', 'leakgan', 'dgsan']


class RealExporter:
    metrics = {
        'jaccard2': 'MSJ-2',
        'jaccard3': 'MSJ-3',
        'jaccard4': 'MSJ-4',
        'jaccard5': 'MSJ-5',

        'bleu2': 'BL-2',
        'bleu3': 'BL-3',
        'bleu4': 'BL-4',
        'bleu5': 'BL-5',

        'self_bleu2': 'SBL-2',
        'self_bleu3': 'SBL-3',
        'self_bleu4': 'SBL-4',
        'self_bleu5': 'SBL-5',

        'fbd': 'FBD',
        'embd': 'W2BD',
        '-nll': 'NLL',
    }
    datasets = {
        "coco60": "COCO Captions",
        "emnlp60": "EMNLP2017 WMT News",
        "wiki72": "Wikitext 103",
        "threecorpus75": "Three Corpus (TextGan)",
        "imdb30": "IMDB Movie Reviews",
        "chpoem5": "Chinese Poem",
    }
    metric_names = ["NLL", "FBD", "W2BD"]
    metric_names += ["MSJ-%d" % i for i in range(2, 6)]
    metric_names += ["BL-%d" % i for i in range(2, 6)]
    metric_names += ["SBL-%d" % i for i in range(2, 6)]


class OracleExporter:
    metrics = {
        "jeffreys": "Jeffreys",
        "bhattacharyya": "Bhattacharyya",
        "lnp_fromq": "OracleNLL",
        "lnq_fromp": "NLL",
    }


def read_data(exporter, dataset_name, model_restore_zip):
    from evaluator import k_fold
    res = {}
    for model_name, restore_type in model_restore_zip.items():
        res[model_name] = []
        for k in range(k_fold if k_fold == 3 else 3):
            dumper = Dumper(create_model(model_name, None), k, dataset_name)
            res[model_name].append(dumper.load_final_results(restore_type))
    new_res = {}
    for model_name in model_restore_zip:
        new_res[model_name] = {}
        for metric in exporter.metrics:
            values = np.array([res[model_name][k][metric] for k in range(len(res[model_name]))])
            if metric == '-nll':
                values *= -1
            if len(res[model_name]) > 1:
                new_res[model_name][exporter.metrics[metric]] = {'mean': np.mean(values), 'std': np.std(values)}
            else:
                new_res[model_name][exporter.metrics[metric]] = {'mean': np.mean(values)}
    res = new_res
    return res


def export_tables(training_mode, dataset_name, model_restore_zip):
    exporter = RealExporter if training_mode == 'real' else OracleExporter
    res = read_data(exporter, dataset_name, model_restore_zip)

    best_model = {x: None for x in exporter.metric_names}

    import re
    for metric_name in exporter.metric_names:
        model_names = [m for m in model_restore_zip.keys() if m != 'real']
        mu = np.array([res[model_name][metric_name]['mean'] for model_name in model_names])
        # nll is ll!
        if re.match('^(FBD|W2BD|NLL|SBL.*)$', metric_name):
            mu *= -1.
        best_model[metric_name] = model_names[int(np.argmax(mu))]
    old_dataset_name = dataset_name
    dataset_name = exporter.datasets[dataset_name]
    caption = make_caption(dataset_name)
    tbl = TblGenerator(["Method"] + list(map(lambda x: "%s" % x, exporter.metric_names)), caption, "scriptsize",
                       "table:%s" % (dataset_name,),
                       # [1, 4, 4, 4])
                       [1, 2, 4, 4, 4])
    for model_name in model_name_orders:
        if model_name not in model_restore_zip:
            continue
        metric_results = []
        for metric_name in exporter.metric_names:
            if 'std' in res[model_name][metric_name]:
                mu, sigma = res[model_name][metric_name]['mean'], res[model_name][metric_name]['std']
                if best_model[metric_name] == model_name or model_name == 'real':
                    s = " \\begin{tabular}{@{}c@{}} $\mathbf{%.3f}$ \\\\ $\mathbf{\pm %.2f}$\\end{tabular}"
                else:
                    s = " \\begin{tabular}{@{}c@{}} $%.3f$ \\\\ $\pm %.2f$\\end{tabular} "
                metric_results.append(s % (mu, sigma))
            else:
                mu = res[model_name][metric_name]['mean']
                if best_model[metric_name] == model_name or model_name == 'real':
                    s = "$\mathbf{%.3f}$"
                else:
                    s = "$%.3f$"
                metric_results.append(s % (mu,))

        tbl.add_row(["%s" % new_model_name[model_name]] + metric_results)
    write_text([str(tbl)], '{}_table_latex'.format(old_dataset_name), TABLE_EXPORT_PATH)

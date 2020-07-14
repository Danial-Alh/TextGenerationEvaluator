import numpy as np

from evaluator import Dumper
from export_utils.evaluation_utils import make_caption, TblGenerator
from models import create_model
from utils.file_handler import write_text
from utils.path_configs import TABLE_EXPORT_PATH

new_model_name = {"dgsan": "DGSAN",
                  "mle": "MLE",
                  "newmle": "NewMLE",
                  "real": "Real Data",
                  "textgan": "TextGAN",
                  "leakgan": "LeakGAN",
                  "maligan": "MaliGAN",
                  "rankgan": "RankGAN",
                  "seqgan": "SeqGAN"}
model_name_orders = ['real', 'mle', 'newmle', 'seqgan', 'maligan', 'rankgan', 'textgan', 'leakgan', 'dgsan']


class RealExporter:
    metrics = {
        'jaccard2': 'MSJ2',
        'jaccard3': 'MSJ3',
        'jaccard4': 'MSJ4',
        'jaccard5': 'MSJ5',

        'bleu2': 'BL2',
        'bleu3': 'BL3',
        'bleu4': 'BL4',
        'bleu5': 'BL5',

        'self_bleu2': 'SBL2',
        'self_bleu3': 'SBL3',
        'self_bleu4': 'SBL4',
        'self_bleu5': 'SBL5',

        'fbd': 'FBD',
        'nll': 'NLL',
    }
    datasets = {
        "coco60": "COCO Captions",
        "emnlp60": "EMNLP2017 WMT News",
        "wiki72": "Wikitext 103",
        "threecorpus75": "Three Corpus (TextGan)",
        "imdb30": "IMDB Movie Reviews",
        "chpoem5": "Chinese Poem",
    }
    metric_names = ["NLL", "FBD"]  # , "W2BD"]
    metric_names += ["MSJ%d" % i for i in range(2, 6)]
    metric_names += ["BL%d" % i for i in range(2, 6)]
    metric_names += ["SBL%d" % i for i in range(2, 6)]
    column_pattern = [1, 1, 4, 4, 4]


class OracleExporter:
    metrics = {
        "jeffreys": "Jeff",
        "bhattacharyya": "Bhattacharyya",
        "nllpfromq": "Oracle-NLL",
        "nllqfromp": "NLL",

        'jaccard2': 'MSJ2',
        'jaccard3': 'MSJ3',
        'jaccard4': 'MSJ4',
        'jaccard5': 'MSJ5',

        'bleu2': 'BL2',
        'bleu3': 'BL3',
        'bleu4': 'BL4',
        'bleu5': 'BL5',

        'self_bleu2': 'SBL2',
        'self_bleu3': 'SBL3',
        'self_bleu4': 'SBL4',
        'self_bleu5': 'SBL5',
    }
    datasets = {
        "oracle75": "Oracle",
    }
    metric_names = ["NLL", "Oraclenll", "Bhattacharyya"]
    # metric_names += ["MSJ%d" % i for i in range(2, 6)]
    # metric_names += ["BL%d" % i for i in range(2, 6)]
    # metric_names += ["SBL%d" % i for i in range(2, 6)]
    column_pattern = [2, 1]


def read_data(exporter, dataset_name, model_run_restore_zip, temperature):
    res = {}
    for model_name, restore_type in model_run_restore_zip.items():
        temp_temperature = temperature
        res[model_name] = []
        # from evaluator import k_fold
        # for run in range(k_fold if k_fold == 3 else 3):
        if dataset_name.startswith('oracle'):
            run = 0
        elif dataset_name.startswith('imdb'):
            run = 2
        elif dataset_name.startswith('emnlp'):
            run = 1
        elif dataset_name.startswith('coco'):
            run = 1
        else:
            raise BaseException('Invalid dataset!! :)')
        if model_name == 'real':
            temp_temperature = {'value': None}
        print("{} {} run{} t {}".format(dataset_name, model_name, run, temp_temperature))
        dumper = Dumper(create_model(model_name, None), run, dataset_name)
        res[model_name].append(dumper.load_final_results(restore_type, temp_temperature))
    new_res = {}
    for model_name in model_run_restore_zip:
        new_res[model_name] = {}
        for metric in exporter.metrics:
            values = np.array([res[model_name][run][metric] for run in range(len(res[model_name]))])
            if metric.startswith('nll'):
                values *= -1
            elif metric.startswith('ln'):
                values *= -1
            if len(res[model_name]) > 1:
                new_res[model_name][exporter.metrics[metric]] = {'mean': np.mean(values), 'std': np.std(values)}
            else:
                new_res[model_name][exporter.metrics[metric]] = {'mean': np.mean(values)}
    res = new_res
    return res


def export_tables(training_mode, dataset_name, model_run_restore_zip, temperature):
    exporter = RealExporter if training_mode == 'real' else OracleExporter
    res = read_data(exporter, dataset_name, model_run_restore_zip, temperature)

    best_model = {x: None for x in exporter.metric_names}

    import re
    for metric_name in exporter.metric_names:
        model_names = [m for m in model_run_restore_zip.keys() if m != 'real']
        mu = np.array([res[model_name][metric_name]['mean'] for model_name in model_names])
        # nll is ll!
        if re.match('^(BL.*|MSJ.*)$', metric_name):
            mu *= -1.
        best_model[metric_name] = model_names[int(np.argmin(mu))]
    old_dataset_name = dataset_name
    dataset_name = exporter.datasets[dataset_name]
    caption = make_caption(dataset_name)
    tbl = TblGenerator(["Method"] + list(map(lambda x: "%s" % x, exporter.metric_names)), caption, "small",
                       "table:%s" % (dataset_name,),
                       column_pattern=exporter.column_pattern)
    # [1, 2, 4, 4, 4])
    for model_name in model_name_orders:
        if model_name not in model_run_restore_zip:
            continue
        metric_results = []
        for metric_name in exporter.metric_names:
            if 'NLL' in metric_name and model_name == 'real':
                s = "-"
                metric_results.append(s)
            elif 'std' in res[model_name][metric_name]:
                mu, sigma = res[model_name][metric_name]['mean'], res[model_name][metric_name]['std']
                if best_model[metric_name] == model_name:  # or model_name == 'real':
                    s = " \\begin{tabular}{@{}c@{}} $\mathbf{%.3f}$ \\\\ $\pm %.3f$\\end{tabular}"
                else:
                    s = " \\begin{tabular}{@{}c@{}} $%.3f$ \\\\ $\pm %.3f$\\end{tabular} "
                metric_results.append(s % (mu, sigma))
            else:
                mu = res[model_name][metric_name]['mean']
                if best_model[metric_name] == model_name:  # or model_name == 'real':
                    s = "$\mathbf{%.3f}$"
                else:
                    s = "$%.3f$"
                metric_results.append(s % (mu,))

        tbl.add_row(["%s" % new_model_name[model_name]] + metric_results)
    write_text([str(tbl)], '{}_table_latex'.format(old_dataset_name), TABLE_EXPORT_PATH)

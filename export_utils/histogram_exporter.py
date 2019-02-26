import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from evaluator import Dumper
from models import create_model
from utils.path_configs import FIG_EXPORT_PATH

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
        'bleu2': 'BL-2',
        'bleu3': 'BL-3',
        'bleu4': 'BL-4',
        'bleu5': 'BL-5',

        'self_bleu2': 'SBL-2',
        'self_bleu3': 'SBL-3',
        'self_bleu4': 'SBL-4',
        'self_bleu5': 'SBL-5',

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
    metric_names = ["NLL"]
    metric_names += ["BL-%d" % i for i in range(3, 5)]
    # metric_names += ["SBL%d" % i for i in range(2, 6)]


class OracleExporter:
    metrics = {
        "jeffreys": "Jeffreys",
        "bhattacharyya": "Bhattacharyya",
        "lnp_fromq": "OracleNLL",
        "lnq_fromp": "NLL",
    }


def export_histogram(training_mode, dataset_name, model_restore_zip, k):
    return
    exporter = RealExporter if training_mode == 'real' else OracleExporter
    for i, metric_name in enumerate(exporter.metrics):
        if exporter.metrics[metric_name] not in exporter.metric_names:
            continue
        plt.figure()
        plt.title('{} - {}'.format(exporter.datasets[dataset_name], exporter.metrics[metric_name]))
        for model_name, restore_type in model_restore_zip.items():
            if model_name == 'real' and metric_name == '-nll':
                continue
            x = Dumper(create_model(model_name, None), k, dataset_name).load_final_results_details(restore_type)
            x = x[metric_name]
            if metric_name.lower().startswith('bleu'):
                sns.kdeplot(x, bw=.2, label=new_model_name[model_name], cut=0)
            else:
                sns.kdeplot(x, bw=.2, label=new_model_name[model_name])
        plt.savefig(FIG_EXPORT_PATH + '{}_{}_hist.png'.format(dataset_name, metric_name))
    for i, metric_name in enumerate(exporter.metrics):
        if exporter.metrics[metric_name] not in exporter.metric_names:
            continue
        if not metric_name.lower().startswith('bleu'):
            continue
        x_label = 'sub_' + metric_name
        y_label = 'self_bleu' + metric_name[-1]
        # plt.figure()
        # plt.title('{} - {}-{}'.format(exporter.datasets[dataset_name],
        #                                    exporter.metrics[metric_name], exporter.metrics[y_label]))
        # sns.set(style="white", color_codes=True)
        for model_name, restore_type in model_restore_zip.items():
            plt.figure()
            plt.title('{} - {} - {}-{}'.format(exporter.datasets[dataset_name], new_model_name[model_name],
                                               exporter.metrics[metric_name], exporter.metrics[y_label]))
            sns.set(style="white", color_codes=True)
            x = Dumper(create_model(model_name, None), k, dataset_name).load_final_results_details(restore_type)
            x, y = np.array(x[x_label]), np.array(x[y_label])
            assert x.shape == y.shape
            x, y = x[:2000], y[:2000]
            x = np.append(x, np.array([0, 0, 1, 1]), axis=0)
            y = np.append(y, np.array([0, 1, 0, 1]), axis=0)
            y = 1 - y
            # sns.scatterplot(x, y, s=7.5, label=new_model_name[model_name])
            # sns.kdeplot(x, y, bw=.1, label=new_model_name[model_name], clip=(0, 1))
            # sns.heatmap(list(zip(x, y)), cmap='magma_r', vmin=0.0, vmax=1.0)
            from pandas import DataFrame
            sns.jointplot('BL', 'SBL', data=DataFrame({'BL': x, 'SBL': y}), kind='kde'
                          , ylim=(0., 1.), xlim=(0., 1.), color='g')
            plt.savefig(FIG_EXPORT_PATH + '{}_{}_{}_{}.png'.format(dataset_name, model_name, metric_name, y_label))
            plt.close()
        # plt.savefig(FIG_EXPORT_PATH + '{}_{}_{}.png'.format(dataset_name, metric_name, y_label))
        # plt.close()

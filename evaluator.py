import os

import numpy as np

from data_management.data_loaders import SentenceDataloader
from data_management.data_manager import SentenceDataManager, OracleDataManager
from file_handler import read_text, zip_folder, create_folder_if_not_exists, unzip_file, dump_json, write_text, \
    load_json
from metrics.bleu import Bleu
from metrics.ms_jaccard import MSJaccard
from metrics.oracle.oracle_lstm import Oracle_LSTM
from metrics.self_bleu import SelfBleu
from models import BaseModel, TexyGen, LeakGan, TextGan
from path_configs import MODEL_PATH, EXPORT_PATH
from utils import tokenize


class Evaluator:
    def __init__(self, data_manager: SentenceDataManager, k=0, name='', init_data=True):
        self.k = k
        self.name = name
        self.data_manager = data_manager
        self.init_dataset()
        if init_data:
            self.init_metrics()

    def init_dataset(self):
        train_data = self.data_manager.get_training_data(self.k, unpack=True)
        valid_data = self.data_manager.get_validation_data(self.k, unpack=True)
        self.train_loc = self.data_manager.dump_unpacked_data_on_file(train_data, self.name + '-train-k' + str(self.k))
        self.valid_loc = self.data_manager.dump_unpacked_data_on_file(valid_data, self.name + '-valid-k' + str(self.k))

    def init_metrics(self):
        id_format_valid_tokenized = tokenize(read_text(self.valid_loc, True))
        valid_texts = self.data_manager.get_parser().id_format2line(id_format_valid_tokenized, True)
        # valid_texts = list(np.random.choice(valid_texts, 1000, replace=False))
        valid_tokens = tokenize(valid_texts)
        # self.bleu5 = Bleu(valid_tokens, weights=np.ones(5) / 5.)
        # self.bleu4 = Bleu(valid_tokens, weights=np.ones(4) / 4., cached_fields=self.bleu5.get_cached_fields())
        # self.bleu3 = Bleu(valid_tokens, weights=np.ones(3) / 3., cached_fields=self.bleu5.get_cached_fields())
        # self.bleu2 = Bleu(valid_tokens, weights=np.ones(2) / 2., cached_fields=self.bleu5.get_cached_fields())
        # self.jaccard5 = MSJaccard(valid_tokens, 5)
        # self.jaccard4 = MSJaccard(valid_tokens, 4, cached_fields=self.jaccard5.get_cached_fields())
        # self.jaccard3 = MSJaccard(valid_tokens, 3, cached_fields=self.jaccard5.get_cached_fields())
        # self.jaccard2 = MSJaccard(valid_tokens, 2, cached_fields=self.jaccard5.get_cached_fields())
        self.oracle = Oracle_LSTM()


class ModelDumper:
    def __init__(self, model: BaseModel, evaluator: Evaluator, k=0, name='', is_dummy=False):
        self.k = k
        self.model = model
        self.evaluator = evaluator
        self.name = name
        self.is_dummy = is_dummy
        self.best_history = {
            # 'bleu3': [{"value": 0.0, "epoch": -1}],
            # 'bleu4': [{"value": 0.0, "epoch": -1}],
            # 'bleu5': [{"value": 0.0, "epoch": -1}],
            '-nll_oracle': [{"value": -np.inf, "epoch": -1}],
            '-nll': [{"value": -np.inf, "epoch": -1}]
        }
        self.n_sampling = 5000

        self.init_paths()
        if not self.is_dummy:
            self.model.set_dumper(self)
            self.model.delete_saved_model()
            self.model.set_train_val_loc(self.evaluator.train_loc, self.evaluator.valid_loc)

    def init_paths(self):
        self.saving_path = MODEL_PATH + self.name + '/' + self.model.get_name() + ('_k%d/' % self.k)
        create_folder_if_not_exists(self.saving_path)

    def store_better_model(self, key):
        zip_folder(self.model.get_saving_path(), key, self.saving_path)

    def restore_model(self, key):
        self.model.delete_saved_model()
        unzip_file(key, self.saving_path, self.model.get_saving_path())
        print('K%d - "%s" based "%s" saved model restored' % (self.k, key, self.model.get_name()))
        self.model.load()

    def update_scores(self, epoch=0, last_iter=False):
        if last_iter:
            self.store_better_model('last_iter')
            print('K%d - model "%s", epoch -, last iter model saved!' % (self.k, self.model.get_name()))
            return
        new_samples = tokenize(self.model.generate_samples(self.n_sampling))
        new_scores = {
            # 'bleu3': self.evaluator.bleu3.get_score(new_samples),
            # 'bleu4': self.evaluator.bleu4.get_score(new_samples),
            # 'bleu5': self.evaluator.bleu5.get_score(new_samples),
            '-nll_oracle': self.evaluator.oracle.log_probability(new_samples),
            '-nll': -float(self.model.get_nll())
        }
        print(new_scores)
        print(new_samples[0])

        for key, new_v in new_scores.items():
            if self.best_history[key][-1]['value'] < new_v:
                print('K%d - model "%s", epoch %d, found better score for "%s": %.4f' %
                      (self.k, self.model.get_name(), epoch, key, new_v))
                self.store_better_model(key)
                self.best_history[key].append({"value": new_v, "epoch": epoch})
                dump_json(self.best_history, 'best_history', self.saving_path)

    def store_generated_samples(self, samples, load_key):
        if not isinstance(samples[0], str):
            samples = [" ".join(s) for s in samples]
        write_text(samples, os.path.join(self.saving_path, load_key + '_based_samples.txt'), is_complete_path=True)

    def load_generated_samples(self, load_key):
        return read_text(os.path.join(self.saving_path, load_key + '_based_samples.txt'), is_complete_path=True)


def final_evaluate(ds_name, dm_name, k=0, n_sampling=5000):
    file_name = 'evaluations_{}k-fold_k{}_{}_nsamp{}'.format(k_fold, k, ds_name, n_sampling)
    t_path = os.path.join(EXPORT_PATH, file_name + '.json')
    if os.path.exists(t_path):
        all_scores = load_json(file_name, EXPORT_PATH)
        print('saved score {} found!'.format(file_name))
        all_scores[k] = all_scores[str(k)]
        del all_scores[str(k)]
    else:
        all_scores = {k: {}}
        print('saved score not found!')

    dm = SentenceDataManager([SentenceDataloader(ds_name)], dm_name + '-words', k_fold=k_fold)
    ev = Evaluator(dm, k, dm_name)

    for restore_type in ['bleu3', 'bleu4', 'bleu5', 'last_iter']:
        restore_type_key = restore_type + ' restore type'
        print(restore_type_key)
        if restore_type_key not in all_scores[k]:
            all_scores[k][restore_type + ' restore type'] = {}
        for model_name in ['seqgan', 'rankgan', 'maligan', 'mle', 'leakgan']:
            print('k: {}, restore_type: {}, model_name: {}'.format(k, restore_type, model_name))
            if model_name in all_scores[k][restore_type_key]:
                print("{} model already evaluated!".format(model_name))
                continue
            if model_name == 'leakgan':
                m = LeakGan(dm.get_parser())
            elif model_name == 'textgan':
                m = TextGan(dm.get_parser())
            else:
                m = TexyGen(model_name, dm.get_parser())
            dumper = ModelDumper(m, ev, k, dm_name, is_dummy=True)

            samples = tokenize(dumper.load_generated_samples(restore_type))
            subsamples_mask = np.random.choice(range(len(samples)), 5000, replace=False)
            subsamples = np.array(samples)[subsamples_mask].tolist()
            # samples = subsamples
            print('subsampled to 5000!')

            # nll = float(m.get_nll())
            nll = -np.inf
            # m.delete()
            del m

            self_bleu5 = SelfBleu(subsamples, weights=np.ones(5) / 5.)
            jaccard5_score, jaccard_cache = ev.jaccard5.get_score(samples, return_cache=True)
            scores = {
                'bleu2': ev.bleu2.get_score(samples),
                'bleu3': ev.bleu3.get_score(samples),
                'bleu4': ev.bleu4.get_score(samples),
                'bleu5': ev.bleu5.get_score(samples),
                'jaccard5': jaccard5_score,
                'jaccard4': ev.jaccard4.get_score(samples, cache=jaccard_cache),
                'jaccard3': ev.jaccard3.get_score(samples, cache=jaccard_cache),
                'jaccard2': ev.jaccard2.get_score(samples, cache=jaccard_cache),
                'self_bleu5': self_bleu5.get_score(),
                'self_bleu4': SelfBleu(subsamples, weights=np.ones(4) / 4.,
                                       cached_fields=self_bleu5.get_cached_fields()).get_score(),
                'self_bleu3': SelfBleu(subsamples, weights=np.ones(3) / 3.,
                                       cached_fields=self_bleu5.get_cached_fields()).get_score(),
                'self_bleu2': SelfBleu(subsamples, weights=np.ones(2) / 2.,
                                       cached_fields=self_bleu5.get_cached_fields()).get_score(),
                '-nll': -nll
            }
            print(scores)
            all_scores[k][restore_type + ' restore type'][model_name] = scores
            dump_json(all_scores, file_name, EXPORT_PATH)


def final_sampling(ds_name, dm_name, k=0, n_sampling=5000):
    file_name = 'evaluations_{}k-fold_k{}_{}_nsamp{}'.format(k_fold, k, ds_name, n_sampling)
    t_path = os.path.join(EXPORT_PATH, file_name + '.json')
    if os.path.exists(t_path):
        all_scores = load_json(file_name, EXPORT_PATH)
        print('saved score {} found!'.format(file_name))
        all_scores[k] = all_scores[str(k)]
        del all_scores[str(k)]
    else:
        all_scores = {k: {}}
        print('saved score not found!')

    dm = SentenceDataManager([SentenceDataloader(ds_name)], dm_name + '-words', k_fold=k_fold)
    ev = Evaluator(dm, k, dm_name, init_data=False)

    for restore_type in ['bleu3', 'bleu4', 'bleu5', 'last_iter']:
        restore_type_key = restore_type + ' restore type'
        print(restore_type_key)
        if restore_type_key not in all_scores[k]:
            all_scores[k][restore_type + ' restore type'] = {}
        for model_name in ['seqgan', 'rankgan', 'maligan', 'mle', 'leakgan']:
            print('k: {}, restore_type: {}, model_name: {}'.format(k, restore_type, model_name))
            if model_name in all_scores[k][restore_type_key]:
                print("{} model already sampled!".format(model_name))
                continue
            if model_name == 'leakgan':
                m = LeakGan(dm.get_parser())
            elif model_name == 'textgan2':
                m = TextGan(dm.get_parser())
            else:
                m = TexyGen(model_name, dm.get_parser())
            dumper = ModelDumper(m, ev, k, dm_name)
            dumper.restore_model(restore_type)

            samples = m.generate_samples(n_sampling)
            dumper.store_generated_samples(samples, restore_type)
            m.delete()
            del m


def start_train(ds_name, dm_name, k=0):
    dm = SentenceDataManager([SentenceDataloader(ds_name)], dm_name + '-words', k_fold=k_fold)

    ev = Evaluator(dm, k, dm_name)

    if m_name == 'leakgan':
        m = LeakGan(dm.get_parser())
    elif m_name == 'textgan2':
        m = TextGan(dm.get_parser())
    else:
        m = TexyGen(m_name, dm.get_parser())
    ModelDumper(m, ev, k, dm_name)
    m.train()


def oracle_train(ds_name, dm_name, k=0):
    dm = OracleDataManager([SentenceDataloader(ds_name)], dm_name + '-words', k_fold=k_fold)

    ev = Evaluator(dm, k, dm_name)

    if m_name == 'leakgan':
        m = LeakGan(dm.get_parser())
    elif m_name == 'textgan2':
        m = TextGan(dm.get_parser())
    else:
        m = TexyGen(m_name, dm.get_parser())
    ModelDumper(m, ev, k, dm_name)
    m.train()


if __name__ == '__main__':
    import sys

    k_fold = 3

    m_name = sys.argv[1]
    dataset_name = sys.argv[2]
    input_k = int(sys.argv[3])
    train = sys.argv[4] == 'train'
    sample = sys.argv[4] == 'sample'
    oracle = sys.argv[4] == 'oracle'
    # k = 1
    dataset_prefix_name = dataset_name.split('-')[0]
    if train:
        start_train(dataset_name, dataset_prefix_name, input_k)
    elif oracle:
        oracle_train(dataset_name, dataset_prefix_name, input_k)
    elif sample:
        final_sampling(dataset_name, dataset_prefix_name, input_k, 20000)
    else:
        final_evaluate(dataset_name, dataset_prefix_name, input_k, 20000)

import os

import numpy as np

from data_management.data_loaders import SentenceDataloader
from data_management.data_manager import SentenceDataManager, OracleDataManager
from utils.file_handler import read_text, zip_folder, create_folder_if_not_exists, unzip_file, dump_json, write_text, \
    load_json
from metrics.oracle.oracle_lstm import Oracle_LSTM
from metrics.self_bleu import SelfBleu
from models import BaseModel, TexyGen, LeakGan, TextGan
from utils.path_configs import MODEL_PATH, EXPORT_PATH
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
            '-nll_oracle': np.mean(self.evaluator.oracle.log_probability(new_samples)),
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

    def get_generated_samples_path(self, load_key):
        return os.path.join(self.saving_path, load_key + '_based_samples.txt')


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


def final_nll_evaluate(ds_name, dm_name, k=0, n_sampling=5000):
    file_name = 'evaluations_{}k-fold_k{}_{}_nsamp{}'.format(k_fold, k, ds_name, n_sampling)
    t_path = os.path.join(EXPORT_PATH, file_name + '.json')
    assert os.path.exists(t_path), 'other scores must be calculated first'
    all_scores = load_json(file_name, EXPORT_PATH)
    print('saved score {} found!'.format(file_name))
    all_scores[k] = all_scores[str(k)]
    del all_scores[str(k)]
    print('evaluating nlls!!!!!!')

    dm = SentenceDataManager([SentenceDataloader(ds_name)], dm_name + '-words', k_fold=k_fold)
    ev = Evaluator(dm, k, dm_name)

    for restore_type in ['bleu3', 'bleu4', 'bleu5', 'last_iter']:
        restore_type_key = restore_type + ' restore type'
        print(restore_type_key)
        assert restore_type_key in all_scores[k], 'all restore type scores must be calculated first'
        for model_name in ['leakgan', 'seqgan', 'rankgan', 'maligan', 'mle']:
            print('k: {}, restore_type: {}, model_name: {}'.format(k, restore_type, model_name))
            assert model_name in all_scores[k][restore_type_key], 'all model scores must be calculated first'
            if model_name == 'leakgan':
                m = LeakGan(dm.get_parser())
            elif model_name == 'textgan':
                m = TextGan(dm.get_parser())
            else:
                m = TexyGen(model_name, dm.get_parser())
            dumper = ModelDumper(m, ev, k, dm_name)
            dumper.restore_model(restore_type)

            ll = float(-m.get_nll())
            m.delete()
            del m

            scores = {
                'bleu2': None,
                'bleu3': None,
                'bleu4': None,
                'bleu5': None,
                'jaccard5': None,
                'jaccard4': None,
                'jaccard3': None,
                'jaccard2': None,
                'self_bleu5': None,
                'self_bleu4': None,
                'self_bleu3': None,
                'self_bleu2': None,
                '-nll': ll
            }
            print(scores)
            for key in scores:
                assert key in all_scores[k][restore_type + ' restore type'][model_name], \
                    'all metric scores must be calculated first'
            if all_scores[k][restore_type + ' restore type'][model_name]['-nll'] != float('inf') and \
                    all_scores[k][restore_type + ' restore type'][model_name]['-nll'] != float('-inf'):
                # assert all_scores[k][restore_type + ' restore type'][model_name]['-nll'] == ll, \
                print('previous nll found!!!!!!!!!!!!!!!!!!!!!!!')
                print(
                    'read nll doesn\'t match %.8f new, %.8f old' % \
                    (ll, all_scores[k][restore_type + ' restore type'][model_name]['-nll']))
            else:
                print('previous nll not found!!!!!!!!!!!!!!!!!!!!!!!')
            all_scores[k][restore_type + ' restore type'][model_name]['-nll'] = ll
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


def final_oracle_evaluate(ds_name, dm_name, k=0, n_sampling=12500):
    file_name = 'oracle_evaluations_{}k-fold_k{}_{}_nsamp{}'.format(k_fold, k, ds_name, n_sampling)
    t_path = os.path.join(EXPORT_PATH, file_name + '.json')
    if os.path.exists(t_path):
        all_scores = load_json(file_name, EXPORT_PATH)
        print('saved score {} found!'.format(file_name))
        all_scores[k] = all_scores[str(k)]
        del all_scores[str(k)]
    else:
        all_scores = {k: {}}
        print('saved score not found!')

    dm = OracleDataManager([SentenceDataloader(ds_name)], dm_name + '-words', k_fold=k_fold)
    print(dm.get_parser().vocab.shape)
    ev = Evaluator(dm, k, dm_name)

    for restore_type in ['-nll_oracle', 'last_iter']:
        restore_type_key = restore_type + ' restore type'
        print(restore_type_key)
        if restore_type_key not in all_scores[k]:
            all_scores[k][restore_type + ' restore type'] = {}
        for model_name in ['leakgan', 'seqgan', 'rankgan', 'maligan', 'mle']:
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
            dumper = ModelDumper(m, ev, k, dm_name)
            dumper.restore_model(restore_type)

            valid_loc = ev.valid_loc
            samples_loc = dumper.get_generated_samples_path(restore_type)
            sample_lines = dumper.load_generated_samples(restore_type)
            id_format_lines = np.array(dm.get_parser().line2id_format(sample_lines)[0])
            print(id_format_lines.shape)
            id_format_sample_loc = dm.dump_unpacked_data_on_file((None, id_format_lines, None),
                                                                 dm_name +
                                                                 '-temp_{}-{}-k'.format(model_name, restore_type) + str(
                                                                     k), parse=False)

            sample_tokens = tokenize(sample_lines)
            sample_tokens = [[int(vv) for vv in v] for v in sample_tokens]
            valid_tokens = tokenize(dm.get_parsed_unpacked_data(dm.get_validation_data(k=k, unpack=True)))
            valid_tokens = [[int(vv) for vv in v] for v in valid_tokens][:len(sample_tokens)]

            oracle_sample_prob = np.array(ev.oracle.log_probability(sample_tokens))
            oracle_valid_prob = np.array(ev.oracle.log_probability(valid_tokens))
            model_valid_prob = np.array(m.get_persample_ll(valid_loc))
            model_sample_prob = np.array(m.get_persample_ll(id_format_sample_loc))

            print('******shapes******')
            print(oracle_sample_prob.shape)
            print(oracle_valid_prob.shape)
            print(model_valid_prob.shape)
            print(model_sample_prob.shape)

            from metrics.divergences import Bhattacharyya, Jeffreys
            bhattacharyya = Bhattacharyya(oracle_valid_prob, model_valid_prob, oracle_sample_prob, model_sample_prob)
            jeffreys = Jeffreys(oracle_valid_prob, model_valid_prob, oracle_sample_prob, model_sample_prob)

            m.delete()
            del m

            scores = {
                'lnp_fromp': float(np.mean(oracle_valid_prob)),
                'lnp_fromq': float(np.mean(oracle_sample_prob)),
                'lnq_fromp': float(np.mean(model_valid_prob)),
                'lnq_fromq': float(np.mean(model_sample_prob)),
                'bhattacharyya': float(bhattacharyya),
                'jeffreys': float(jeffreys)
            }
            print(scores)
            all_scores[k][restore_type + ' restore type'][model_name] = scores
            dump_json(all_scores, file_name, EXPORT_PATH)


def final_oracle_sampling(ds_name, dm_name, k=0, n_sampling=12500):
    file_name = 'oracle_evaluations_{}k-fold_k{}_{}_nsamp{}'.format(k_fold, k, ds_name, n_sampling)
    t_path = os.path.join(EXPORT_PATH, file_name + '.json')
    if os.path.exists(t_path):
        all_scores = load_json(file_name, EXPORT_PATH)
        print('saved score {} found!'.format(file_name))
        all_scores[k] = all_scores[str(k)]
        del all_scores[str(k)]
    else:
        all_scores = {k: {}}
        print('saved score not found!')

    dm = OracleDataManager([SentenceDataloader(ds_name)], dm_name + '-words', k_fold=k_fold)
    ev = Evaluator(dm, k, dm_name, init_data=False)

    for restore_type in ['-nll_oracle', 'last_iter']:
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


def store_parsed_validations(ds_name, dm_name):
    dm = SentenceDataManager([SentenceDataloader(ds_name)], dm_name + '-words', k_fold=k_fold)
    for k in range(k_fold):
        train_data = dm.get_training_data(k, unpack=True)
        valid_data = dm.get_validation_data(k, unpack=True)
        train_loc = dm.dump_unpacked_data_on_file(train_data, dm_name + '-train-k' + str(k), parse=True)
        valid_loc = dm.dump_unpacked_data_on_file(valid_data, dm_name + '-valid-k' + str(k), parse=True)


if __name__ == '__main__':
    import sys

    k_fold = 3

    m_name = sys.argv[1]
    dataset_name = sys.argv[2]
    input_k = int(sys.argv[3])

    train = sys.argv[4] == 'train'
    sample = sys.argv[4] == 'sample'
    eval = sys.argv[4] == 'eval'
    eval_nll = sys.argv[4] == 'eval_nll'

    oracle = sys.argv[4] == 'oracle'
    sample_oracle = sys.argv[4] == 'sample_oracle'
    eval_oracle = sys.argv[4] == 'eval_oracle'

    store_valid = sys.argv[4] == 'store_valid'

    dataset_prefix_name = dataset_name.split('-')[0]
    if train:
        start_train(dataset_name, dataset_prefix_name, input_k)
    elif oracle:
        oracle_train(dataset_name, dataset_prefix_name, input_k)
    elif sample_oracle:
        final_oracle_sampling(dataset_name, dataset_prefix_name, input_k)
    elif eval_oracle:
        final_oracle_evaluate(dataset_name, dataset_prefix_name, input_k)
    elif sample:
        final_sampling(dataset_name, dataset_prefix_name, input_k, 20000)
    elif store_valid:
        store_parsed_validations(dataset_name, dataset_prefix_name)
    elif eval_nll:
        final_nll_evaluate(dataset_name, dataset_prefix_name, input_k, 20000)
    elif eval:
        final_evaluate(dataset_name, dataset_prefix_name, input_k, 20000)
    else:
        print('invalid input!!!')

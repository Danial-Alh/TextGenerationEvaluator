import os

import numpy as np
from metrics.cythonics.lib.bleu import Bleu
from metrics.cythonics.lib.self_bleu import SelfBleu

from metrics.fbd_embd import FBD_EMBD
from metrics.ms_jaccard import MSJaccard
from metrics.oracle.oracle_lstm import Oracle_LSTM
from models import BaseModel
from models import all_models, create_model
from utils.file_handler import read_text, zip_folder, create_folder_if_not_exists, unzip_file, dump_json, write_text, \
    load_json
from utils.nltk_utils import word_base_tokenize
from utils.path_configs import MODEL_PATH, EXPORT_PATH, BERT_PATH


class Evaluator:
    def __init__(self, train_data, valid_data, test_data, parser, mode, k=0, dm_name=''):
        # in train and gen mode, data is coded but in eval mode data is raw
        self.parser = parser
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.k = k
        self.dm_name = dm_name
        self.during_training_n_sampling = 5000
        self.test_n_sampling = test_n_sampling
        self.init_metrics(mode)

    def init_metrics(self, mode):
        pass

    def get_initial_scores_during_training(self):
        pass

    def get_during_training_scores(self, model: BaseModel):
        pass

    def get_test_scores(self, refs_with_additional_fields, samples_with_additional_fields):
        pass

    def generate_samples(self, model_restore_zip):
        # if restore_types is None:
        #     restore_types = self.test_restore_types
        # if model_names is None:
        #     model_names = all_models
        assert model_restore_zip is not None

        for model_name, restore_type in model_restore_zip.items():
            print(restore_type)
            # for model_name in model_names:
            print('k: {}, restore_type: {}, model_name: {}'.format(self.k, restore_type, model_name))

            tracker = BestModelTracker(model_name, self)
            dumper = tracker.dumper
            model = tracker.model
            dumper.restore_model(restore_type)

            sample_codes = model.generate_samples(self.test_n_sampling)
            additional_fields = self.get_sample_additional_fields(model, sample_codes,
                                                                  self.test_data, restore_type)

            sample_lines = self.parser.id_format2line(sample_codes)

            key = list(additional_fields['gen'].keys())[0]
            min_len = min(len(sample_lines), len(additional_fields['gen'][key]))
            sample_lines = sample_lines[:min_len]
            for key in additional_fields['gen']:
                additional_fields['gen'][key] = additional_fields['gen'][key][:min_len]

            key = list(additional_fields['test'].keys())[0]
            min_len = min(len(self.test_data), len(additional_fields['test'][key]))
            test_lines = self.parser.id_format2line(self.test_data[:min_len])
            for key in additional_fields['test']:
                additional_fields['test'][key] = additional_fields['test'][key][:min_len]

            dumper.dump_samples_with_additional_fields(sample_lines,
                                                       additional_fields['gen'],
                                                       restore_type, 'gen')
            dumper.dump_samples_with_additional_fields(test_lines,
                                                       additional_fields['test'],
                                                       restore_type, 'test')
            model.delete()

    def get_sample_additional_fields(self, model: BaseModel, sample_codes, test_codes, restore_type):
        pass

    def final_evaluate(self, model_restore_zip):
        # if restore_types is None:
        #     restore_types = self.test_restore_types
        # if model_names is None:
        #     model_names = all_models
        assert model_restore_zip is not None

        # self.eval_pre_check(model_restore_zip)
        for model_name, restore_type in model_restore_zip.items():
            print(restore_type)
            # for model_name in model_names:
            print('k: {}, restore_type: {}, model_name: {}'.format(self.k, restore_type, model_name))
            m = create_model(model_name, None)
            dumper = Dumper(m, self.k, self.dm_name)
            refs_with_additional_fields = dumper.load_samples_with_additional_fields(restore_type, 'test')
            samples_with_additional_fields = \
                dumper.load_samples_with_additional_fields(restore_type, 'gen')

            scores_persample, scores = self.get_test_scores(refs_with_additional_fields,
                                                            samples_with_additional_fields)
            dumper.dump_final_results(scores, restore_type)
            dumper.dump_final_results_details(scores_persample, restore_type)

    def eval_pre_check(self, model_restore_zip):
        # if restore_types is None:
        #     restore_types = self.test_restore_types
        # if model_names is None:
        #     model_names = all_models
        print('ref/test pre checking!')
        assert model_restore_zip is not None

        for model_name, restore_type in model_restore_zip.items():
            print(restore_type)
            # for model_name in model_names:
            print(model_name)
            assert self.test_samples_equals_references(Dumper(create_model(model_name, None), self.k, self.dm_name).
                                                       load_samples_with_additional_fields(restore_type, 'test'))
        print('ref/test pre checking passed!')

    def test_samples_equals_references(self, refs_with_additional_fields):
        print('checking test/ref equivalence!')
        refs = [r['text'] for r in refs_with_additional_fields]
        A = set(refs)
        B = set(self.test_data)
        inter = len(A.intersection(B))
        print('{}% --- {}/({},{}) intersection,{}/{} {}/{} diff'.format(inter*100/float(len(A)), inter, len(A), len(B),
                                                                        len(A) - inter, len(A), len(B) - inter, len(B)))
        return A == B


class RealWorldEvaluator(Evaluator):
    test_restore_types = ['bleu3', 'bleu4', 'bleu5', 'last_iter']

    def __init__(self, train_data, valid_data, test_data, parser, mode, k=0, dm_name=''):
        super().__init__(train_data, valid_data, test_data, parser, mode, k, dm_name)
        self.selfbleu_n_sampling = selfbleu_n_s

    def init_metrics(self, mode):
        if mode == 'train':
            print(len(self.train_data), len(self.valid_data))
            valid_tokens = word_base_tokenize(self.parser.id_format2line(self.valid_data))
            self.bleu5 = Bleu(valid_tokens, weights=np.ones(5) / 5.)
            self.bleu4 = Bleu(valid_tokens, weights=np.ones(4) / 4., other_instance=self.bleu5)
            self.bleu3 = Bleu(valid_tokens, weights=np.ones(3) / 3., other_instance=self.bleu5)
        elif mode == 'eval':
            test_tokens = word_base_tokenize(self.test_data)
            self.bleu5 = Bleu(test_tokens, weights=np.ones(5) / 5.)
            self.bleu4 = Bleu(test_tokens, weights=np.ones(4) / 4., other_instance=self.bleu5)
            self.bleu3 = Bleu(test_tokens, weights=np.ones(3) / 3., other_instance=self.bleu5)
            self.bleu2 = Bleu(test_tokens, weights=np.ones(2) / 2., other_instance=self.bleu5)
            self.jaccard5 = MSJaccard(test_tokens, 5)
            self.jaccard4 = MSJaccard(test_tokens, 4, cached_fields=self.jaccard5.get_cached_fields())
            self.jaccard3 = MSJaccard(test_tokens, 3, cached_fields=self.jaccard5.get_cached_fields())
            self.jaccard2 = MSJaccard(test_tokens, 2, cached_fields=self.jaccard5.get_cached_fields())

            # self.fbd = FBD(self.test_data, 64, BERT_PATH)
            # self.embd = EMBD(self.test_data, 64, BERT_PATH)
            print(BERT_PATH)
            self.fbd_embd = FBD_EMBD(self.test_data, max_length=max_l, bert_model_dir=BERT_PATH)
        elif mode == 'gen':
            pass
        elif mode == 'eval_precheck':
            pass
        else:
            raise BaseException('invalid evaluator mode!')

    def get_initial_scores_during_training(self):
        return {
            'bleu3': [{"value": 0.0, "epoch": -1}],
            'bleu4': [{"value": 0.0, "epoch": -1}],
            'bleu5': [{"value": 0.0, "epoch": -1}],
            '-nll': [{"value": -np.inf, "epoch": -1}]
        }

    def get_during_training_scores(self, model: BaseModel):
        samples = model.generate_samples(self.during_training_n_sampling)
        samples = self.parser.id_format2line(samples, merge=False)
        new_scores = {
            'bleu3': np.mean(self.bleu3.get_score(samples)),
            'bleu4': np.mean(self.bleu4.get_score(samples)),
            'bleu5': np.mean(self.bleu5.get_score(samples)),
            '-nll': -model.get_nll()
        }
        return new_scores

    def get_sample_additional_fields(self, model: BaseModel, sample_codes, test_codes, restore_type):
        lnqfromp = model.get_persample_ll(test_codes)
        lnqfromq = model.get_persample_ll(sample_codes)
        return {'gen': {'lnq': lnqfromq}, 'test': {'lnq': lnqfromp}}

    def get_test_scores(self, refs_with_additional_fields, samples_with_additional_fields):
        sample_lines = [r['text'] for r in samples_with_additional_fields]
        sample_tokens = word_base_tokenize(sample_lines)

        if self.selfbleu_n_sampling == -1:
            subsampled_tokens = sample_tokens
            subsamples_mask = [i for i in range(len(sample_tokens))]
        else:
            subsamples_mask = np.random.choice(range(len(sample_tokens)), self.selfbleu_n_sampling, replace=False)
            subsampled_tokens = np.array(sample_tokens)[subsamples_mask].tolist()

        self_bleu5 = SelfBleu(subsampled_tokens, weights=np.ones(5) / 5.)
        jaccard5_score, jaccard_cache = self.jaccard5.get_score(sample_tokens, return_cache=True)
        fbd_embd_result = self.fbd_embd.get_score(sample_lines)

        scores_persample = {
            'bleu2': self.bleu2.get_score(sample_tokens),
            'bleu3': self.bleu3.get_score(sample_tokens),
            'bleu4': self.bleu4.get_score(sample_tokens),
            'bleu5': self.bleu5.get_score(sample_tokens),
            'self_bleu5': self_bleu5.get_score(),
            'self_bleu4': SelfBleu(subsampled_tokens, weights=np.ones(4) / 4., other_instance=self_bleu5).get_score(),
            'self_bleu3': SelfBleu(subsampled_tokens, weights=np.ones(3) / 3., other_instance=self_bleu5).get_score(),
            'self_bleu2': SelfBleu(subsampled_tokens, weights=np.ones(2) / 2., other_instance=self_bleu5).get_score(),
            '-nll': [r['lnq'] for r in refs_with_additional_fields],
        }
        scores_persample['sub_bleu2'] = list(np.array(scores_persample['bleu2'])[subsamples_mask])
        scores_persample['sub_bleu3'] = list(np.array(scores_persample['bleu3'])[subsamples_mask])
        scores_persample['sub_bleu4'] = list(np.array(scores_persample['bleu4'])[subsamples_mask])
        scores_persample['sub_bleu5'] = list(np.array(scores_persample['bleu5'])[subsamples_mask])
        scores_mean = {
            'jaccard5': jaccard5_score,
            'jaccard4': self.jaccard4.get_score(sample_tokens, cache=jaccard_cache),
            'jaccard3': self.jaccard3.get_score(sample_tokens, cache=jaccard_cache),
            'jaccard2': self.jaccard2.get_score(sample_tokens, cache=jaccard_cache),
            'fbd': fbd_embd_result['FBD'],
            'embd': fbd_embd_result['EMBD']
        }
        for key in scores_persample:
            scores_mean[key] = np.mean(scores_persample[key])
        return scores_persample, scores_mean


class OracleEvaluator(Evaluator):
    test_restore_types = ['-nll_oracle', 'last_iter']

    def __init__(self, train_data, valid_data, test_data, parser, mode, k=0, dm_name=''):
        super().__init__(train_data, valid_data, test_data, parser, mode, k, dm_name)

    def init_metrics(self, mode):
        if mode == 'train':
            self.oracle = Oracle_LSTM()
        elif mode == 'test':
            self.oracle = Oracle_LSTM()
        elif mode == 'gen':
            pass
        elif mode == 'eval_precheck':
            pass
        else:
            raise BaseException('invalid evaluator mode!')

    def get_initial_scores_during_training(self):
        return {
            '-nll_oracle': [{"value": -np.inf, "epoch": -1}],
            '-nll': [{"value": -np.inf, "epoch": -1}]
        }

    def get_during_training_scores(self, model: BaseModel):
        new_samples = model.generate_samples(self.during_training_n_sampling)
        new_scores = {
            '-nll_oracle': np.mean(self.oracle.log_probability(new_samples)),
            '-nll': -model.get_nll()
        }
        return new_scores

    def get_sample_additional_fields(self, model: BaseModel, sample_codes, test_codes, restore_type):
        test_lines = self.parser.id_format2line(test_codes)
        sample_lines = self.parser.id_format2line(sample_codes)
        lnqfromp = model.get_persample_ll(test_codes)
        lnqfromq = model.get_persample_ll(sample_codes)
        lnpfromp = self.oracle.log_probability(test_lines)
        lnpfromq = self.oracle.log_probability(sample_lines)
        return {'gen': {'lnq': lnqfromq, 'lnp': lnpfromq},
                'test': {'lnq': lnqfromp, 'lnp': lnpfromp}}

    def get_test_scores(self, refs_with_additional_fields, samples_with_additional_fields):
        from metrics.divergences import Bhattacharyya, Jeffreys

        lnqfromp = [r['lnq'] for r in refs_with_additional_fields]
        lnqfromq = [r['lnq'] for r in samples_with_additional_fields]
        lnpfromp = [r['lnp'] for r in refs_with_additional_fields]
        lnpfromq = [r['lnp'] for r in samples_with_additional_fields]

        scores_persample = {
            'lnqfromp': lnqfromp,
            'lnqfromq': lnqfromq,
            'lnpfromp': lnpfromp,
            'lnpfromq': lnpfromq
        }
        scores_mean = {
            'bhattacharyya': Bhattacharyya(lnpfromp, lnqfromp, lnpfromq, lnqfromq),
            'jeffreys': Jeffreys(lnpfromp, lnqfromp, lnpfromq, lnqfromq),
        }
        for key in scores_persample:
            scores_mean[key] = np.mean(scores_persample[key])
        return scores_persample, scores_mean


class Dumper:
    def __init__(self, model: BaseModel, k=0, dm_name=''):
        self.k = k
        self.dm_name = dm_name
        self.model = model

        self.init_paths()

    def init_paths(self):
        self.saving_path = MODEL_PATH + self.dm_name + '/' + self.model.get_name() + ('_k%d/' % self.k)
        create_folder_if_not_exists(self.saving_path)
        self.final_result_parent_path = os.path.join(EXPORT_PATH, 'evaluations_{}k-fold_k{}_{}/' \
                                                     .format(k_fold, self.k, self.dm_name))
        create_folder_if_not_exists(self.final_result_parent_path)
        self.final_result_file_name = self.model.get_name()

    def store_better_model(self, key):
        zip_folder(self.model.get_saving_path(), key, self.saving_path)

    def restore_model(self, key):
        self.model.delete_saved_model()
        unzip_file(key, self.saving_path, self.model.get_saving_path())
        print('K%d - "%s" based "%s" saved model restored' % (self.k, key, self.model.get_name()))
        self.model.load()

    def dump_generated_samples(self, samples, load_key):
        if not isinstance(samples[0], str):
            samples = [" ".join(s) for s in samples]
        write_text(samples, os.path.join(self.saving_path, load_key + '_based_samples.txt'), is_complete_path=True)

    def load_generated_samples(self, load_key):
        return read_text(os.path.join(self.saving_path, load_key + '_based_samples.txt'), is_complete_path=True)

    def get_generated_samples_path(self, load_key):
        return os.path.join(self.saving_path, load_key + '_based_samples.txt')

    def dump_best_history(self, best_history):
        dump_json(best_history, 'best_history', self.saving_path)

    def dump_samples_with_additional_fields(self, samples, additional_fields: dict, load_key, sample_label):
        dump_json([
            {**{'text': samples[i]}, **{key: additional_fields[key][i] for key in additional_fields.keys()}}
            for i in range(len(samples))],
            sample_label + '_' + load_key + '_based_samples', self.saving_path)

    def load_samples_with_additional_fields(self, load_key, sample_label):
        result = load_json(sample_label + '_' + load_key + '_based_samples', self.saving_path)
        return result

    def dump_final_results(self, results, restore_type):
        dump_json(results, self.final_result_file_name + '_' + restore_type + 'restore', self.final_result_parent_path)

    def load_final_results(self, restore_type):
        return load_json(self.final_result_file_name + '_' + restore_type + 'restore', self.final_result_parent_path)

    def dump_final_results_details(self, results, restore_type):
        dump_json(results, self.final_result_file_name + '_' + restore_type + 'restore_details',
                  self.final_result_parent_path)

    def load_final_results_details(self, restore_type):
        return load_json(self.final_result_file_name + '_' + restore_type + 'restore_details',
                         self.final_result_parent_path)


class BestModelTracker:
    def __init__(self, model_name, evaluator: Evaluator):
        self.k = evaluator.k
        self.model = create_model(model_name, evaluator.parser)
        self.dumper = Dumper(self.model, self.k, evaluator.dm_name)
        self.evaluator = evaluator
        self.best_history = self.evaluator.get_initial_scores_during_training()
        self.model.set_tracker(self)
        self.model.delete_saved_model()
        self.model.create_model()

    def update_scores(self, epoch=0, last_iter=False):
        if last_iter:
            self.dumper.store_better_model('last_iter')
            print('K%d - model "%s", epoch -, last iter model saved!' % (self.k, self.model.get_name()))
            return
        new_scores = self.evaluator.get_during_training_scores(self.model)
        print(new_scores)

        for key, new_v in new_scores.items():
            if self.best_history[key][-1]['value'] < new_v:
                print('K%d - model "%s", epoch %d, found better score for "%s": %.4f' %
                      (self.k, self.model.get_name(), epoch, key, new_v))
                self.dumper.store_better_model(key)
                self.best_history[key].append({"value": new_v, "epoch": epoch})
                self.dumper.dump_best_history(self.best_history)

    def start(self):
        self.model.set_train_val_data(self.evaluator.train_data, self.evaluator.valid_data)
        self.model.train()


# ######## COCO
# k_fold = 3
# selfbleu_n_s = 5000
# test_n_s = 20000

######## IMDB
# k_fold = 3
# selfbleu_n_s = 5000
# test_n_s = 10000

######## WIKI
# k_fold = 3
# selfbleu_n_s = 5000
# test_n_s = 24000


# ######## CHINESE
# k_fold = 14
# selfbleu_n_s = -1
# test_n_s = 2000
max_l, test_n_sampling, k_fold, selfbleu_n_s = 0, 0, 0, 0


def update_config(dm_name):
    global max_l, test_n_sampling, k_fold, selfbleu_n_s
    if dm_name.startswith('emnlp'):
        k_fold = 3
        max_l = 41
        selfbleu_n_s = 5000
        test_n_sampling = 20000
    elif dm_name.startswith('wiki'):
        k_fold = 3
        max_l = 41
        selfbleu_n_s = 5000
        test_n_sampling = 24000
    elif dm_name.startswith('imdb'):
        k_fold = 3
        max_l = 41
        selfbleu_n_s = 5000
        test_n_sampling = 10000
    elif dm_name.startswith('threecorpus'):
        k_fold = 3
        max_l = 41
        selfbleu_n_s = 5000
        test_n_sampling = 25000
    elif dm_name.startswith('coco'):
        k_fold = 3
        max_l = 26
        selfbleu_n_s = 5000
        test_n_sampling = 20000
    elif dm_name.startswith('chpoem'):
        global BERT_PATH
        from utils.path_configs import CH_BERT_PATH
        k_fold = 14
        max_l = 24
        selfbleu_n_s = -1
        test_n_sampling = 2000
        BERT_PATH = CH_BERT_PATH
    else:
        raise BaseException('dm_name {} is invalid!'.format(dm_name))

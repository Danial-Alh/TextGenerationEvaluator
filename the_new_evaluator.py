import os

import numpy as np

from data_management.data_manager import SentenceDataManager, OracleDataManager
from file_handler import read_text, zip_folder, create_folder_if_not_exists, unzip_file, dump_json, write_text, \
    load_json
from metrics.bleu import Bleu
from metrics.fbd import FBD
from metrics.ms_jaccard import MSJaccard
from metrics.oracle.oracle_lstm import Oracle_LSTM
from metrics.self_bleu import SelfBleu
from models import BaseModel, TexyGen, LeakGan, TextGan
from path_configs import MODEL_PATH, EXPORT_PATH
from utils import tokenize


def create_model(model_name, parser):
    if model_name == 'leakgan':
        m = LeakGan(parser)
    elif model_name == 'textgan':
        m = TextGan(parser)
    else:
        m = TexyGen(model_name, parser)
    return m


class Evaluator:
    modes = ('train', 'test', 'gen')

    def __init__(self, data_manager, mode, k=0, dm_name=''):
        self.k = k
        self.dm_name = dm_name
        self.data_manager = data_manager
        self.during_training_n_sampling = 5000
        self.test_n_sampling = 20000
        self.test_restore_types = None

        self.init_dataset()
        self.init_metrics(mode)

    def init_dataset(self):
        train_data = self.data_manager.get_training_data(self.k, unpack=True)
        valid_data = self.data_manager.get_validation_data(self.k, unpack=True)
        self.train_loc = self.data_manager.dump_unpacked_data_on_file(train_data, self.dm_name + '-train-k'
                                                                      + str(self.k))
        self.valid_loc = self.data_manager.dump_unpacked_data_on_file(valid_data, self.dm_name + '-valid-k'
                                                                      + str(self.k))

    def init_metrics(self, mode):
        pass

    def get_initial_scores_during_training(self):
        pass

    def get_during_training_scores(self, model: BaseModel):
        pass

    def get_test_scores(self, dumper, restore_type):
        pass

    def generate_samples(self):
        for restore_type in self.test_restore_types:
            print(restore_type)
            for model_name in ['seqgan', 'rankgan', 'maligan', 'mle', 'leakgan']:
                print('k: {}, restore_type: {}, model_name: {}'.format(self.k, restore_type, model_name))

                model = create_model(model_name, self.data_manager.get_parser())
                dumper = Dumper(model, self.k, self.dm_name)
                dumper.restore_model(restore_type)

                sample_lines = model.generate_samples(self.test_n_sampling)
                valid_lines = self.data_manager.get_parsed_unpacked_data(
                    self.data_manager.get_validation_data(k=self.k, unpack=True))
                additional_fields = self.get_sample_additional_fields(model, sample_lines, valid_lines, restore_type)
                dumper.dump_samples_with_additional_fields(sample_lines,
                                                           additional_fields['gen'],
                                                           restore_type, 'gen')
                dumper.dump_samples_with_additional_fields(valid_lines,
                                                           additional_fields['valid'],
                                                           restore_type, 'valid')
                model.delete()

    def get_sample_additional_fields(self, model: BaseModel, sample_lines, valid_lines, restore_type):
        pass

    def final_evaluate(self):
        for restore_type in self.test_restore_types:
            print(restore_type)
            for model_name in ['seqgan', 'rankgan', 'maligan', 'mle', 'leakgan']:
                print('k: {}, restore_type: {}, model_name: {}'.format(self.k, restore_type, model_name))
                m = create_model(model_name, self.data_manager.get_parser())
                dumper = Dumper(m, self.k, self.dm_name)

                scores_persample, scores = self.get_test_scores(dumper, restore_type)
                dumper.dump_final_results(scores, restore_type)
                dumper.dump_final_results_details(scores_persample, restore_type)


class RealWorldEvaluator(Evaluator):
    def __init__(self, data_manager: SentenceDataManager, mode, k=0, dm_name=''):
        super().__init__(data_manager, mode, k, dm_name)
        self.test_restore_types = ['bleu3', 'bleu4', 'bleu5', 'last_iter']

    def init_metrics(self, mode):
        if mode == 'train':
            id_format_valid_tokenized = tokenize(read_text(self.valid_loc, True))
            valid_texts = self.data_manager.get_parser().id_format2line(id_format_valid_tokenized, True)
            # valid_texts = list(np.random.choice(valid_texts, 1000, replace=False))
            valid_tokens = tokenize(valid_texts)
            self.bleu5 = Bleu(valid_tokens, weights=np.ones(5) / 5.)
            self.bleu4 = Bleu(valid_tokens, weights=np.ones(4) / 4., cached_fields=self.bleu5.get_cached_fields())
            self.bleu3 = Bleu(valid_tokens, weights=np.ones(3) / 3., cached_fields=self.bleu5.get_cached_fields())
        if mode == 'test':
            id_format_valid_tokenized = tokenize(read_text(self.valid_loc, True))
            valid_texts = self.data_manager.get_parser().id_format2line(id_format_valid_tokenized, True)
            # valid_texts = list(np.random.choice(valid_texts, 1000, replace=False))
            valid_tokens = tokenize(valid_texts)
            self.bleu5 = Bleu(valid_tokens, weights=np.ones(5) / 5.)
            self.bleu4 = Bleu(valid_tokens, weights=np.ones(4) / 4., cached_fields=self.bleu5.get_cached_fields())
            self.bleu3 = Bleu(valid_tokens, weights=np.ones(3) / 3., cached_fields=self.bleu5.get_cached_fields())
            self.bleu2 = Bleu(valid_tokens, weights=np.ones(2) / 2., cached_fields=self.bleu5.get_cached_fields())
            self.jaccard5 = MSJaccard(valid_tokens, 5)
            self.jaccard4 = MSJaccard(valid_tokens, 4, cached_fields=self.jaccard5.get_cached_fields())
            self.jaccard3 = MSJaccard(valid_tokens, 3, cached_fields=self.jaccard5.get_cached_fields())
            self.jaccard2 = MSJaccard(valid_tokens, 2, cached_fields=self.jaccard5.get_cached_fields())
            self.fbd = FBD(valid_texts)
        if mode == 'gen':
            pass

    def get_initial_scores_during_training(self):
        return {
            'bleu3': [{"value": 0.0, "epoch": -1}],
            'bleu4': [{"value": 0.0, "epoch": -1}],
            'bleu5': [{"value": 0.0, "epoch": -1}],
            '-nll': [{"value": -np.inf, "epoch": -1}]
        }

    def get_during_training_scores(self, model: BaseModel):
        new_samples = tokenize(model.generate_samples(self.during_training_n_sampling))
        new_scores = {
            'bleu3': self.bleu3.get_score(new_samples),
            'bleu4': self.bleu4.get_score(new_samples),
            'bleu5': self.bleu5.get_score(new_samples),
            '-nll': -float(model.get_nll())
        }
        return new_scores

    def get_sample_additional_fields(self, model: BaseModel, sample_lines, valid_lines, restore_type):
        id_format_lines = np.array(self.data_manager.get_parser().line2id_format(sample_lines)[0])
        print(id_format_lines.shape)
        id_format_sample_loc = self.data_manager.dump_unpacked_data_on_file(
            (None, id_format_lines, None),
            self.dm_name + '-temp_{}-{}-k'.format(model.get_name(), restore_type) + str(self.k),
            parse=False)

        logqfromp = np.array(model.get_persample_ll(self.valid_loc))
        logqfromq = np.array(model.get_persample_ll(id_format_sample_loc))
        return {'gen': {'logqfromq': logqfromq}, 'valid': {'logqfromp': logqfromp}}

    def get_test_scores(self, dumper, restore_type):
        samples_with_additional_fields = dumper.load_samples_with_additional_fields(restore_type, 'gen')
        refs_with_additional_fields = dumper.load_samples_with_additional_fields(restore_type, 'valid')
        samples = tokenize([r['text'] for r in samples_with_additional_fields])

        subsamples_mask = np.random.choice(range(len(samples)), 5000, replace=False)
        subsamples = np.array(samples)[subsamples_mask].tolist()

        self_bleu5 = SelfBleu(subsamples, weights=np.ones(5) / 5.)
        jaccard5_score, jaccard_cache = self.jaccard5.get_score(samples, return_cache=True)
        scores_persample = {
            'bleu2': self.bleu2.get_score(samples),
            'bleu3': self.bleu3.get_score(samples),
            'bleu4': self.bleu4.get_score(samples),
            'bleu5': self.bleu5.get_score(samples),
            'self_bleu5': self_bleu5.get_score(),
            'self_bleu4': SelfBleu(subsamples, weights=np.ones(4) / 4.,
                                   cached_fields=self_bleu5.get_cached_fields()).get_score(),
            'self_bleu3': SelfBleu(subsamples, weights=np.ones(3) / 3.,
                                   cached_fields=self_bleu5.get_cached_fields()).get_score(),
            'self_bleu2': SelfBleu(subsamples, weights=np.ones(2) / 2.,
                                   cached_fields=self_bleu5.get_cached_fields()).get_score(),
            '-nll': [r['logqfromp'] for r in refs_with_additional_fields]
        }
        scores_mean = {
            'jaccard5': jaccard5_score,
            'jaccard4': self.jaccard4.get_score(samples, cache=jaccard_cache),
            'jaccard3': self.jaccard3.get_score(samples, cache=jaccard_cache),
            'jaccard2': self.jaccard2.get_score(samples, cache=jaccard_cache),
            'fbd': self.fbd.get_score(samples)
        }
        for key in scores_persample:
            scores_mean[key] = float(np.mean(scores_persample[key]))
        return scores_persample, scores_mean


class OracleEvaluator(Evaluator):
    def __init__(self, data_manager: OracleDataManager, mode, k=0, dm_name=''):
        super().__init__(data_manager, mode, k, dm_name)
        self.test_restore_types = ['-nll_oracle', 'last_iter']

    def init_metrics(self, mode):
        if mode == 'train':
            self.oracle = Oracle_LSTM()
        if mode == 'test':
            self.oracle = Oracle_LSTM()
        if mode == 'gen':
            pass

    def get_initial_scores_during_training(self):
        return {
            '-nll_oracle': [{"value": -np.inf, "epoch": -1}],
            '-nll': [{"value": -np.inf, "epoch": -1}]
        }

    def get_during_training_scores(self, model: BaseModel):
        new_samples = tokenize(model.generate_samples(self.during_training_n_sampling))
        new_scores = {
            '-nll_oracle': np.mean(self.oracle.log_probability(new_samples)),
            '-nll': -float(model.get_nll())
        }
        return new_scores

    def get_sample_additional_fields(self, model: BaseModel, sample_lines, valid_lines, restore_type):
        id_format_lines = np.array(self.data_manager.get_parser().line2id_format(sample_lines)[0])
        print(id_format_lines.shape)
        sample_tokens = tokenize(sample_lines)
        sample_tokens = [[int(vv) for vv in v] for v in sample_tokens]
        valid_tokens = tokenize(self.data_manager.get_parsed_unpacked_data(
            self.data_manager.get_validation_data(k=self.k, unpack=True)))
        valid_tokens = [[int(vv) for vv in v] for v in valid_tokens][:len(sample_tokens)]

        id_format_sample_loc = self.data_manager.dump_unpacked_data_on_file(
            (None, id_format_lines, None),
            self.dm_name + '-temp_{}-{}-k'.format(model.get_name(), restore_type) + str(self.k),
            parse=False)

        logqfromp = np.array(model.get_persample_ll(self.valid_loc))
        logqfromq = np.array(model.get_persample_ll(id_format_sample_loc))
        logpfromq = np.array(self.oracle.log_probability(sample_tokens))
        logpfromp = np.array(self.oracle.log_probability(valid_tokens))
        return {'gen': {'logqfromq': logqfromq, 'logpfromq': logpfromq},
                'valid': {'logqfromp': logqfromp, 'logpfromp': logpfromp}}

    def get_test_scores(self, dumper, restore_type):
        from metrics.divergences import Bhattacharyya, Jeffreys

        samples_with_additional_fields = dumper.load_samples_with_additional_fields(restore_type, 'gen')
        refs_with_additional_fields = dumper.load_samples_with_additional_fields(restore_type, 'valid')

        logqfromp = [r['logqfromp'] for r in refs_with_additional_fields]
        logqfromq = [r['logqfromq'] for r in samples_with_additional_fields]
        logpfromp = [r['logpfromp'] for r in refs_with_additional_fields]
        logpfromq = [r['logpfromq'] for r in samples_with_additional_fields]

        scores_persample = {
            'logqfromp': logqfromp,
            'logqfromq': logqfromq,
            'logpfromp': logpfromp,
            'logpfromq': logpfromq
        }
        scores_mean = {
            'bhattacharyya': float(Bhattacharyya(logpfromp, logqfromp, logpfromq, logqfromq)),
            'jeffreys': float(Jeffreys(logpfromp, logqfromp, logpfromq, logqfromq)),
        }
        for key in scores_persample:
            scores_mean[key] = float(np.mean(scores_persample[key]))
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
        self.final_result_file_name = 'evaluations_{}k-fold_k{}_{}' \
            .format(k_fold, self.k, self.dm_name)

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
        if not isinstance(samples[0], str):
            samples = [" ".join(s) for s in samples]
        dump_json([
            {**{'text': samples[i]}, **{key: additional_fields[key][i] for key in additional_fields.keys()}}
            for i in range(len(samples))],
            sample_label + '_' + load_key + '_based_samples.txt', self.saving_path)

    def load_samples_with_additional_fields(self, load_key, sample_label):
        return load_json(sample_label + '_' + load_key + '_based_samples.txt', self.saving_path)

    def dump_final_results(self, results, restore_type):
        dump_json(results, self.final_result_file_name + '_' + restore_type, EXPORT_PATH)

    def load_final_results(self, restore_type):
        return load_json(self.final_result_file_name + '_' + restore_type, EXPORT_PATH)

    def dump_final_results_details(self, results, restore_type):
        dump_json(results, self.final_result_file_name + '_' + restore_type, EXPORT_PATH)

    def load_final_results_details(self, restore_type):
        return load_json(self.final_result_file_name + '_' + restore_type, EXPORT_PATH)


class BestModelTracker:
    def __init__(self, model: BaseModel, evaluator: Evaluator, k=0):
        self.k = k
        self.model = model
        self.dumper = Dumper(model, k, evaluator.dm_name)
        self.evaluator = evaluator
        self.best_history = self.evaluator.get_initial_scores_during_training()
        self.model.set_dumper(self)
        self.model.delete_saved_model()
        self.model.set_train_val_loc(self.evaluator.train_loc, self.evaluator.valid_loc)

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


k_fold = 3

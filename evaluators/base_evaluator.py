from evaluators.best_model_tracker import BestModelTracker
from evaluators.model_dumper import ModelDumper
from evaluators.oracle_evaluator import OracleEvaluator
from evaluators.real_evaluator import RealWorldEvaluator
from previous_works import create_model
from previous_works.model_wrappers.base_model import BaseModel


class Evaluator:
    def __init__(self, train_data, valid_data, test_data, parser, mode, k, temperature, dm_name=''):
        # in train and gen mode, data is coded but in eval mode data is raw
        self.parser = parser
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.k = k
        self.temperature = temperature
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
            if model_name != 'real':
                print(restore_type)
                # for model_name in model_names:
                print('k: {}, restore_type: {}, model_name: {}'.format(self.k, restore_type, model_name))

                tracker = BestModelTracker(model_name, self)
                dumper = tracker.dumper
                model = tracker.model
                dumper.restore_model(restore_type)
                sample_codes = model.generate_samples(self.test_n_sampling, self.temperature)
                additional_fields = self.get_sample_additional_fields(model, sample_codes,
                                                                      self.test_data, restore_type)

                sample_lines = self.parser.reverse(sample_codes)

                min_len = min([len(self.test_data)] + [len(additional_fields['gen'][key]) for key in \
                                                       additional_fields['gen'].keys()])
                sample_lines = sample_lines[:min_len]
                for key in additional_fields['gen']:
                    additional_fields['gen'][key] = additional_fields['gen'][key][:min_len]

                min_len = min([len(self.test_data)] + [len(additional_fields['test'][key]) for key in \
                                                       additional_fields['test'].keys()])
                test_lines = self.parser.id_format2line(self.test_data[:min_len])
                for key in additional_fields['test']:
                    additional_fields['test'][key] = additional_fields['test'][key][:min_len]

                dumper.dump_samples_with_additional_fields(sample_lines,
                                                           additional_fields['gen'],
                                                           restore_type, 'gen', self.temperature)
                dumper.dump_samples_with_additional_fields(test_lines,
                                                           additional_fields['test'],
                                                           restore_type, 'test', self.temperature)
                model.delete()
            else:
                train_data_loader = SentenceDataloader(self.dm_name + '-train')
                tr, _ = dm.get_data(self.k, parse=True)
                ts = dm.get_data(-1, parse=True)
                min_size = min(len(tr), len(ts))
                tr, ts = tr[:min_size], ts[:min_size]
                model = create_model('real', None)
                dumper = ModelDumper(model, self.k, self.dm_name)
                additional_fields = self.get_sample_additional_fields(model, None,
                                                                      ts, restore_type)

                dumper.dump_samples_with_additional_fields(tr,
                                                           additional_fields['gen'],
                                                           restore_type, 'gen', self.temperature)
                dumper.dump_samples_with_additional_fields(ts,
                                                           additional_fields['test'],
                                                           restore_type, 'test', self.temperature)

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
            dumper = ModelDumper(m, self.k, self.dm_name)
            refs_with_additional_fields = dumper.load_samples_with_additional_fields(restore_type, 'test',
                                                                                     self.temperature)
            samples_with_additional_fields = \
                dumper.load_samples_with_additional_fields(restore_type, 'gen', self.temperature)
            print('samples: {}, refs: {}, raw tests: {}'.format(len(samples_with_additional_fields),
                                                                len(refs_with_additional_fields),
                                                                len(self.test_data)))
            min_size = min(len(samples_with_additional_fields), len(self.test_data), len(refs_with_additional_fields))
            print('sample size to ------>>>>>> {}'.format(min_size))
            samples_with_additional_fields = samples_with_additional_fields[:min_size]
            refs_with_additional_fields = refs_with_additional_fields[:min_size]

            scores_persample, scores = self.get_test_scores(refs_with_additional_fields,
                                                            samples_with_additional_fields)
            dumper.dump_final_results(scores, restore_type, self.temperature)
            dumper.dump_final_results_details(scores_persample, restore_type, self.temperature)

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
            assert self.test_samples_equals_references(
                ModelDumper(create_model(model_name, None), self.k, self.dm_name).
                    load_samples_with_additional_fields(restore_type, 'test'))
        print('ref/test pre checking passed!')

    def test_samples_equals_references(self, refs_with_additional_fields):
        print('checking test/ref equivalence!')
        refs = [r['text'] for r in refs_with_additional_fields]
        A = set(refs)
        B = set(self.test_data)
        inter = len(A.intersection(B))
        print('{}% --- {}/({},{}) intersection,{}/{} {}/{} diff'.format(inter * 100 / float(len(A)), inter, len(A),
                                                                        len(B),
                                                                        len(A) - inter, len(A), len(B) - inter, len(B)))
        return A == B


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


def get_temperature_stringified(temperature):
    if temperature['value'] is None:
        return ''
    elif temperature['type'] == 'biased':
        return '_' + format(temperature['value'], '0.6f')
    elif temperature['type'] == 'unbiased':
        return '_unbiased' + format(temperature['value'], '0.6f')
    else:
        raise BaseException('invalid temperature type!!')


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
    elif dm_name.startswith('oracle'):
        k_fold = 3
        # max_l = 26
        selfbleu_n_s = 5000
        test_n_sampling = 25000
    else:
        raise BaseException('dm_name {} is invalid!'.format(dm_name))

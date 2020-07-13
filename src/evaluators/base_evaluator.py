from functools import reduce

from evaluators.best_model_tracker import BestModelTracker
from evaluators.model_dumper import ModelDumper
from db_management.models import ModelSamples
from previous_works import create_model
from previous_works.model_wrappers.base_model import BaseModel
from utils.path_configs import BERT_PATH


class Evaluator:
    max_l, test_n_sampling, total_runs, selfbleu_n_s, bert_path = 0, 0, 0, 0, ''

    @staticmethod
    def update_config(dm_name):
        if dm_name.startswith('coco'):
            Evaluator.total_runs = 3
            Evaluator.Evaluatorbleu_n_s = 5000
            Evaluator.test_n_sampling = 20000
            Evaluator.bert_path = BERT_PATH
        elif dm_name.startswith('oracle'):
            Evaluator.total_runs = 3
            #Evaluator.max_l = 26
            Evaluator.selfbleu_n_s = 5000
            Evaluator.test_n_sampling = 25000
        else:
            raise BaseException('dm_name {} is invalid!'.format(dm_name))

    def __init__(self, train_ds, valid_ds, test_ds, parser, mode, k, temperature, dm_name=''):
        # in train and gen mode, data is coded but in eval mode data is raw
        self.update_config(dm_name)
        self.parser = parser
        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.test_ds = test_ds
        self.k = k
        self.temperature = temperature
        self.dm_name = dm_name
        self.during_training_n_sampling = 5000
        self.init_metrics(mode)

    def init_metrics(self, mode):
        pass

    def get_initial_scores_during_training(self):
        pass

    def get_during_training_scores(self, model: BaseModel):
        pass

    def get_test_scores(self, samples: ModelSamples):
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
            print('k: {}, restore_type: {}, model_name: {}'.format(
                self.k, restore_type, model_name))

            if model_name != 'real':
                tracker = BestModelTracker(model_name, self)
                model = tracker.model
                dumper = tracker.dumper
                dumper.restore_model(restore_type)

                generated_tokens = model.generate_samples(self.test_n_sampling, self.temperature)
                test_tokens = list(self.test_ds.text)
            else:
                model = create_model('real', None)
                dumper = ModelDumper(model, self.k, self.dm_name)

                generated_tokens = list(self.train_ds.text)
                test_tokens = list(self.test_ds.text)

            min_size = min(len(generated_tokens), len(test_tokens))
            generated_tokens, test_tokens = generated_tokens[:min_size], test_tokens[:min_size]

            dumping_object = {'gen': {}, 'test': {}}
            dumping_object['gen']['tokens'] = generated_tokens
            dumping_object['test']['tokens'] = test_tokens
            dumping_object['gen']['sentence'] = self.parser.detokenize(generated_tokens)
            dumping_object['test']['sentence'] = self.parser.detokenize(test_tokens)

            self.add_persample_metrics(dumping_object, model, restore_type)
            assert self.supplementary_info_exists_for_each_sample(dumping_object)

            dumper.dump_samples_with_persample_metrics(
                dumping_object, restore_type, self.temperature)
            model.delete()

    def add_persample_metrics(self, dumping_object, model, restore_type):
        pass

    def final_evaluate(self, model_restore_zip):
        assert model_restore_zip is not None

        # self.eval_pre_check(model_restore_zip)
        for model_name, restore_type in model_restore_zip.items():
            print(restore_type)
            print('run: {}, restore_type: {}, model_name: {}'.format(self.k, restore_type, model_name))

            m = create_model(model_name, None)
            dumper = ModelDumper(m, self.k, self.dm_name)
            samples = dumper.load_samples_with_persample_metrics(restore_type, self.temperature)

            print('samples: {}, refs: {}, raw tests: {}'
                  .format(len(samples.generate_samples),
                          len(samples.test_samples),
                          len(self.test_ds)))

            persample_scores, scores = self.get_test_scores(samples)
            dumper.dump_final_evaluation_results(scores, restore_type, self.temperature)
            dumper.update_persample_metrics_for_generated_samples(
                persample_scores, restore_type, self.temperature)

    @staticmethod
    def supplementary_info_exists_for_each_sample(dumping_object):
        for group_key, group_value in dumping_object.items():
            result = reduce(
                lambda prev, v: prev and (len(v) == len(group_value.values[0])),
                group_value.values(),
                True
            )
            if result == False:
                print('{} has invalid supplementary info length'.format(group_key))
                return False
        return True


# ######## COCO
# total_runs = 3
# selfbleu_n_s = 5000
# test_n_s = 20000

# IMDB
# total_runs = 3
# selfbleu_n_s = 5000
# test_n_s = 10000

# WIKI
# total_runs = 3
# selfbleu_n_s = 5000
# test_n_s = 24000


# ######## CHINESE
# total_runs = 14
# selfbleu_n_s = -1
# test_n_s = 2000

from types import SimpleNamespace

from db_management.model_db_manager import ModelDBManager
from evaluators.best_model_tracker import ModelDumpManager
from previous_works.model_wrappers import create_model
from previous_works.model_wrappers.base_model import BaseModel
from utils.path_configs import BERT_PATH as B_P
from utils.path_configs import COMPUTER_NAME


class Evaluator:
    TEST_N_S, TOTAL_RUNS, SELFBLEU_N_S, bert_path = 0, 0, 0, ''

    @staticmethod
    def update_config(dm_name):
        if dm_name.startswith('coco'):
            Evaluator.TOTAL_RUNS = 3
            Evaluator.SELFBLEU_N_S = 5000
            Evaluator.BERT_PATH = B_P
        elif dm_name.startswith('news'):
            Evaluator.TOTAL_RUNS = 3
            Evaluator.SELFBLEU_N_S = 5000
            Evaluator.BERT_PATH = B_P
        elif dm_name.startswith('ptb'):
            Evaluator.TOTAL_RUNS = 3
            Evaluator.SELFBLEU_N_S = 5000
            Evaluator.BERT_PATH = B_P
        elif dm_name.startswith('amazon_app_book'):
            Evaluator.TOTAL_RUNS = 3
            Evaluator.SELFBLEU_N_S = 5000
            Evaluator.BERT_PATH = B_P
        elif dm_name.startswith('yelp_restaurant'):
            Evaluator.TOTAL_RUNS = 3
            Evaluator.SELFBLEU_N_S = 5000
            Evaluator.BERT_PATH = B_P
        elif dm_name.startswith('oracle'):
            Evaluator.TOTAL_RUNS = 3
            Evaluator.SELFBLEU_N_S = 5000
        else:
            raise BaseException('dm_name {} is invalid!'.format(dm_name))

    def __init__(self, train_ds, valid_ds, test_ds, parser, mode, dm_name=''):
        # in train and gen mode, data is coded but in eval mode data is raw
        self.update_config(dm_name)
        self.parser = parser
        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.test_ds = test_ds
        self.dm_name = dm_name
        self.during_training_n_sampling = 5000
        self.init_metrics(mode)

    def init_metrics(self, mode):
        pass

    def get_initial_scores_during_training(self):
        pass

    def get_during_training_scores(self, model: BaseModel, train_temperature):
        pass

    def get_test_scores(self, samples):
        pass

    def generate_samples(self, model_identifier: SimpleNamespace):
        model_name = model_identifier.model_name
        run = model_identifier.run
        train_temperature = model_identifier.train_temperature
        test_temperature = model_identifier.test_temperature
        restore_type = model_identifier.restore_type

        if model_name != 'real':
            model = create_model(model_identifier, self.parser)
            dumper = ModelDumpManager(model, self.dm_name)
            db_manager = ModelDBManager(
                computer_name=COMPUTER_NAME,
                dataset_name=self.dm_name,
                model_name=model.get_name(),
                run=run,
                train_temperature=train_temperature,
                restore_type=restore_type,
                test_temperature=test_temperature)

            model.init_model((self.train_ds.text, self.valid_ds.text))
            dumper.restore_model()

            generated_tokens = model.generate_samples(len(self.test_ds), test_temperature)
            test_tokens = list(self.test_ds.text)
        else:
            model = create_model('real', None)
            dumper = ModelDumpManager(model, self.dm_name)

            generated_tokens = list(self.train_ds.text)
            test_tokens = list(self.test_ds.text)

        min_size = min(len(generated_tokens), len(test_tokens))
        generated_tokens, test_tokens = generated_tokens[:min_size], test_tokens[:min_size]

        dumping_object = {'generated': {}, 'test': {}}
        dumping_object['generated']['tokens'] = generated_tokens
        dumping_object['test']['tokens'] = test_tokens
        dumping_object['generated']['sentence'] = self.parser.detokenize(generated_tokens)
        dumping_object['test']['sentence'] = self.parser.detokenize(test_tokens)

        self.add_persample_metrics(dumping_object, model, test_temperature)

        db_manager.dump_samples_with_persample_metrics(dumping_object)
        model.reset_model()

    def add_persample_metrics(self, dumping_object, model, test_temperature):
        pass

    def final_evaluate(self, model_identifier):
        model_name = model_identifier.model_name
        run = model_identifier.run
        train_temperature = model_identifier.train_temperature
        test_temperature = model_identifier.test_temperature
        restore_type = model_identifier.restore_type

        m = create_model(model_identifier, None)
        db_manager = ModelDBManager(
            computer_name=COMPUTER_NAME,
            dataset_name=self.dm_name,
            model_name=m.get_name(),
            run=run,
            train_temperature=train_temperature,
            restore_type=restore_type,
            test_temperature=test_temperature)
        samples = db_manager.load_samples_with_persample_metrics()

        print('samples: {}, refs: {}, raw tests: {}'
              .format(len(samples['generated']),
                      len(samples['test']),
                      len(self.test_ds)))

        persample_scores, scores = self.get_test_scores(samples)
        db_manager.dump_final_evaluation_results(scores)
        db_manager.update_persample_metrics_for_generated_samples(persample_scores)


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

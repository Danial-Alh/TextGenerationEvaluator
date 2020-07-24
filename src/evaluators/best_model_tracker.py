
import os

from previous_works.model_wrappers import create_model
from utils.model_db_manager import ModelDBManager
from utils.path_configs import COMPUTER_NAME, EXPORT_PATH, MODEL_PATH
from utils.file_handler import (create_folder_if_not_exists, unzip_file,
                                zip_folder)


class BestModelTracker:
    def __init__(self, model_name, run, temperature, evaluator):
        self.run = run
        self.temperature = temperature
        self.evaluator = evaluator

        self.model = create_model(model_name, evaluator.parser)
        self.model.set_tracker(self)
        self.model.delete_saved_model()

        self.dump_manager = ModelDumpManager(self.model, self.run, evaluator.dm_name)
        self.db_manager = ModelDBManager(COMPUTER_NAME, self.evaluator.dm_name,
                                         self.model.get_name(), self.run)
        self.best_history = None

    def update_metrics(self, epoch=0, last_iter=False):
        if last_iter:
            self.dump_manager.store_better_model('last_iter')
            print('Run %d - model "%s", epoch -, last iter model saved!' %
                  (self.run, self.model.get_name()))
            return
        new_metrics = {metric: {'value': v, 'epoch': epoch}
                       for metric, v
                       in self.evaluator.get_during_training_scores(self.model, self.temperature).items()}
        print(new_metrics)

        updated_metrics = {}
        for metric, new_v in new_metrics.items():
            if self.best_history[metric][-1]['value'] < new_v['value']:
                print('Run %d - model "%s", epoch %d, found better score for "%s": %.4f' %
                      (self.run, self.model.get_name(), epoch, metric, new_v['value']))
                self.dump_manager.store_better_model(metric)
                self.best_history[metric].append(new_v)
                updated_metrics[metric] = new_v

        self.db_manager.append_to_best_history(updated_metrics)
        self.db_manager.append_to_history(new_metrics)

    def start(self):
        initial_scores = self.evaluator.get_initial_scores_during_training()
        self.db_manager.init_history(initial_scores)
        self.best_history = {metric: [v] for metric, v in initial_scores.items()}
        self.model.init_model((self.evaluator.train_ds.text, self.evaluator.valid_ds.text))
        self.model.train()


class ModelDumpManager:
    def __init__(self, model, run=0, dm_name=''):
        self.run = run
        self.dm_name = dm_name
        self.model = model
        self.init_paths()

    def init_paths(self):
        self.saving_path = MODEL_PATH + self.dm_name + '/' + self.model.get_name() + ('_run%d/' % self.run)
        create_folder_if_not_exists(self.saving_path)

    def store_better_model(self, key):
        zip_folder(self.model.get_saving_path(), key, self.saving_path)

    def restore_model(self, key):
        self.model.delete_saved_model()
        unzip_file(key, self.saving_path, self.model.get_saving_path())
        self.model.load()
        print('Run %d - "%s" based "%s"; saved model restored' %
              (self.run, key, self.model.get_name()))

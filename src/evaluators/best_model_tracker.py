
import os
from types import SimpleNamespace

from previous_works.model_wrappers import create_model
from db_management.model_db_manager import ModelDBManager
from utils.path_configs import COMPUTER_NAME, EXPORT_PATH, MODEL_PATH
from utils.file_handler import (create_folder_if_not_exists, unzip_file,
                                zip_folder)


class BestModelTracker:
    def __init__(self, model_identifier: SimpleNamespace, evaluator):
        self.evaluator = evaluator
        self.model_identifier = model_identifier

        self.model = create_model(model_identifier, evaluator.parser)
        self.model.set_tracker(self)
        self.model.delete_saved_model()

        self.dump_manager = ModelDumpManager(self.model,
                                             evaluator.dm_name)
        self.db_manager = ModelDBManager(
            computer_name=COMPUTER_NAME,
            dataset_name=self.evaluator.dm_name,
            model_name=self.model.get_name(),
            run=model_identifier.run,
            train_temperature=model_identifier.train_temperature)
        self.best_history = None

    def update_metrics(self, epoch=0, last_iter=False):
        if last_iter:
            self.dump_manager.store_better_model('last_iter')
            print('{}, epoch -, last iter model saved!'.format(self.model_identifier))
            return
        new_metrics = {metric: {'value': v, 'epoch': epoch}
                       for metric, v
                       in self.evaluator.get_during_training_scores(self.model, self.model_identifier.train_temperature).items()}
        print(new_metrics)

        updated_metrics = {}
        for metric, new_v in new_metrics.items():
            if self.best_history[metric][-1]['value'] < new_v['value']:
                print('{}, epoch {}, found better score for {}: {:.4f}'
                      .format(self.model_identifier, epoch, metric, new_v['value']))
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
    def __init__(self, model, dm_name=''):
        self.dm_name = dm_name
        self.model = model
        self.init_paths()

    def init_paths(self):
        self.saving_path = MODEL_PATH + self.dm_name + '/' + \
            self.model.get_name() + '/temp{}/run{}/'\
            .format(
                ModelDBManager.get_stringified_temperature(self.model.train_temperature),
                self.model.run)
        create_folder_if_not_exists(self.saving_path)

    def store_better_model(self, key):
        zip_folder(self.model.get_saving_path(), key, self.saving_path)

    def restore_model(self):
        self.model.delete_saved_model()
        unzip_file(self.model.restore_type,
                   self.saving_path, self.model.get_saving_path())
        self.model.load()
        print('{} - {} based {}; saved model restored'
              .format(self.model, self.model.restore_type, self.model.get_name()))

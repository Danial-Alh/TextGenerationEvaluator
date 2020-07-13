from previous_works import create_model

from .base_evaluator import Evaluator
from .model_dumper import ModelDumper


class BestModelTracker:
    def __init__(self, model_name, evaluator: Evaluator):
        self.k = evaluator.k
        self.model = create_model(model_name, evaluator.parser)
        self.dumper = ModelDumper(self.model, self.k, evaluator.dm_name)
        self.evaluator = evaluator
        self.best_history = self.evaluator.get_initial_scores_during_training()
        self.model.set_tracker(self)
        self.model.delete_saved_model()
        self.model.create_model()
        self.dumper.init_history()

    def update_metrics(self, epoch=0, last_iter=False):
        if last_iter:
            self.dumper.store_better_model('last_iter')
            print('Run %d - model "%s", epoch -, last iter model saved!' %
                  (self.k, self.model.get_name()))
            return
        new_metrics = {k: {'value': v, 'epoch': epoch}
                       for k, v in self.evaluator.get_during_training_scores(self.model).items()}
        print(new_metrics)

        updated_metrics = {}
        for metric, new_v in new_metrics.items():
            if self.best_history[metric][-1]['value'] < new_v['value']:
                print('Run %d - model "%s", epoch %d, found better score for "%s": %.4f' %
                      (self.k, self.model.get_name(), epoch, metric, new_v['value']))
                self.dumper.store_better_model(metric)
                self.best_history[metric].append(new_v)
                updated_metrics[metric] = new_v

        self.dumper.append_to_best_history(updated_metrics)
        self.dumper.append_to_history(new_metrics)

    def start(self):
        self.model.set_train_val_data(self.evaluator.train_data, self.evaluator.valid_data)
        self.model.train()

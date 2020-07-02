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

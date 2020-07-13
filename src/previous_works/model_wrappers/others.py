from previous_works.model_wrappers.base_model import BaseModel, empty_sentence_remover_decorator, data2tempfile_decorator
from data_management.parsers import Parser


class DGSAN(BaseModel):
    pass


class Real(BaseModel):
    pass


class LeakGan(BaseModel):
    def __init__(self, parser: Parser):
        super().__init__(parser)
        import previous_works.leakgan2.Main as leakgan2main
        self.model_module = leakgan2main

    def create_model(self):
        self.model = self.model_module.LeakGanMain(self, self.parser)
        self.load()

    def set_train_val_data(self, train_data, valid_data):
        super().set_train_val_data(train_data, valid_data)
        with open(self.model_module.positive_file, 'w') as trg:
            trg.write('\n'.join(train_data))

    def train(self):
        self.model.train()
        self.tracker.update_metrics(last_iter=True)

    @empty_sentence_remover_decorator
    def generate_samples(self, n_samples, temperature):
        codes = self.model.generate_samples(n_samples, self.model_module.dummy_file, 0)
        return codes

    @data2tempfile_decorator
    def get_nll(self, temperature, samples=None, samples_loc=None):
        dl = self.model_module.Gen_Data_loader(self.model_module.BATCH_SIZE, self.parser.max_length)
        dl.create_batches(samples_loc)
        score = self.model.target_loss(dl)
        return float(score)

    @data2tempfile_decorator
    def get_persample_ll(self, temperature, samples=None, samples_loc=None):
        dl = self.model_module.Gen_Data_loader(self.model_module.BATCH_SIZE, self.parser.max_length)
        dl.create_batches(samples_loc)
        score = self.model.per_sample_ll(dl)
        return [float(s) for s in score]

    def get_saving_path(self):
        return self.model_module.model_path

    def get_name(self):
        return 'leakgan2'

    def load(self):
        super().load()
        import tensorflow as tf
        self.model.sess.run(tf.local_variables_initializer())
        self.model.sess.run(tf.global_variables_initializer())
        model_path = tf.train.latest_checkpoint(self.get_saving_path())
        if model_path is not None:
            self.model.saver.restore(self.model.sess, model_path)
            print(model_path + ' found!')
        else:
            print('Model not found. Randomly initialized!')

    def delete(self):
        import tensorflow as tf
        tf.reset_default_graph()


class TextGan(BaseModel):
    def __init__(self, parser: Parser):
        super().__init__(parser)

    def create_model(self):
        pass

    def set_train_val_data(self, train_data, valid_data):
        super().set_train_val_data(train_data, valid_data)

    def train(self):
        from previous_works.textgan2.textGAN import train_model
        train_model(self.train_data, self.valid_data, self.valid_data, None, None, None, self.parser.id2vocab,
                    self.parser.vocab.shape[0], self.parser.END_TOKEN_ID)
        # self.model.train_func(self.train_data, self.valid_data)
        # self.tracker.update_metrics(last_iter=True)

    @empty_sentence_remover_decorator
    def generate_samples(self, n_samples, temperature):
        codes = self.model.generate()
        return codes

    @data2tempfile_decorator
    def get_nll(self, temperature, samples=None, samples_loc=None):
        return 0.0

    @data2tempfile_decorator
    def get_persample_ll(self, temperature, samples=None, samples_loc=None):
        pass

    def get_saving_path(self):
        from previous_works.textgan2.textGAN import SAVE_PATH
        return SAVE_PATH

    def get_name(self):
        return 'textgan_org'

    def delete(self):
        import tensorflow as tf
        tf.reset_default_graph()

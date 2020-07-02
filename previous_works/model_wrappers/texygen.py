from previous_works.model_wrappers.base_model import BaseModel, empty_sentence_remover_decorator, data2tempfile_decorator
from data_management.parsers import Parser


class TexyGen(BaseModel):

    def __init__(self, gan_name, parser: Parser):
        from previous_works.texygen.models.leakgan.Leakgan import Leakgan
        from previous_works.texygen.models.leakgan.LeakganDataLoader import DataLoader as LeakganDL
        from previous_works.texygen.models.maligan_basic.Maligan import Maligan
        from previous_works.texygen.models.maligan_basic.MaliganDataLoader import DataLoader as MaliganDL
        from previous_works.texygen.models.mle.Mle import Mle
        from previous_works.texygen.models.mle.MleDataLoader import DataLoader as MLEDL
        from previous_works.texygen.models.rankgan.Rankgan import Rankgan
        from previous_works.texygen.models.rankgan.RankganDataLoader import DataLoader as RankganDL
        from previous_works.texygen.models.seqgan.Seqgan import Seqgan
        from previous_works.texygen.models.seqgan.SeqganDataLoader import DataLoader as SeqganDL
        from previous_works.texygen.models.textGan_MMD.Textgan import TextganMmd
        from previous_works.texygen.models.textGan_MMD.TextganDataLoader import DataLoader as TextganDL
        gans, dls = dict(), dict()
        gans['seqgan'] = Seqgan
        gans['textgan'] = TextganMmd
        gans['leakgan'] = Leakgan
        gans['rankgan'] = Rankgan
        gans['maligan'] = Maligan
        gans['mle'] = Mle
        dls['seqgan'] = SeqganDL
        dls['textgan'] = TextganDL
        dls['leakgan'] = LeakganDL
        dls['rankgan'] = RankganDL
        dls['maligan'] = MaliganDL
        dls['mle'] = MLEDL
        super().__init__(parser)
        self.train_loc = None
        self.valid_loc = None
        self.model_class = gans[gan_name.lower()]
        self.dataloader_class = dls[gan_name.lower()]

    def create_model(self):
        self.model = self.model_class()
        self.model.init_real_trainng(self.parser)
        self.load()

    def set_train_val_data(self, train_data, valid_data):
        super().set_train_val_data(train_data, valid_data)

    def train(self):
        self.model.train_real(self.train_loc, self)
        self.tracker.update_scores(last_iter=True)

    @empty_sentence_remover_decorator
    def generate_samples(self, n_samples, temperature):
        if self.model.__class__.__name__ == 'Leakgan':
            from previous_works.texygen.models.leakgan import Leakgan
            codes = Leakgan.generate_samples_gen(self.model.sess, self.model.generator, self.model.batch_size,
                                                 n_samples, self.model.test_file)
        else:
            from previous_works.texygen.utils.utils import generate_samples
            codes = generate_samples(self.model.sess, self.model.generator,
                                     self.model.batch_size, n_samples, self.model.test_file,
                                     temperature=temperature)
        return codes

    def init_nll(self, data_loc, temperature):
        from previous_works.texygen.utils.metrics.Nll import Nll
        valid_dataloader = self.dataloader_class(batch_size=self.model.batch_size,
                                                 seq_length=self.model.sequence_length)
        valid_dataloader.create_batches(data_loc)

        inll = Nll(data_loader=valid_dataloader, rnn=self.model.generator, sess=self.model.sess,
                   temperature=temperature)
        inll.set_name('nll-test-' + data_loc)
        return inll

    def init_persample_ll(self, data_loc, temperature):
        from previous_works.texygen.utils.metrics.ItemFetcher import ItemFetcher
        dataloader = self.dataloader_class(batch_size=self.model.batch_size,
                                           seq_length=self.model.sequence_length)
        dataloader.create_batches(data_loc)
        if temperature['value'] is None:
            item_to_be_fetched = self.model.generator.selfdefined_persample_ll
        elif temperature['type'] == 'biased':
            item_to_be_fetched = self.model.generator.selfdefined_temp_persample_ll
        elif temperature['type'] == 'unbiased':
            item_to_be_fetched = self.model.generator.unbiased_temperature_persample_ll
        else:
            raise BaseException('invalid temperature type!')
        inll = ItemFetcher(data_loader=dataloader, rnn=self.model.generator,
                           item_to_be_fetched=item_to_be_fetched,
                           sess=self.model.sess, temperature=temperature)
        inll.set_name('persample_ll' + data_loc)
        return inll

    @data2tempfile_decorator
    def get_nll(self, temperature, samples=None, samples_loc=None):
        inll = self.init_nll(samples_loc, temperature)
        score = inll.get_score()
        return float(score)

    @data2tempfile_decorator
    def get_persample_ll(self, temperature, samples=None, samples_loc=None):
        ll = self.init_persample_ll(samples_loc, temperature)
        score = ll.get_score()
        return [float(s) for s in score]

    def get_saving_path(self):
        return self.model_class.saving_path

    def get_name(self):
        return self.model_class.__name__

    def load(self):
        super().load()
        import tensorflow as tf
        self.model.sess.run(tf.global_variables_initializer())
        self.model.sess.run(tf.local_variables_initializer())
        self.model.load_generator_discriminator()

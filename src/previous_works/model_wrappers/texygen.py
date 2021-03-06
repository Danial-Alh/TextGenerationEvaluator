from torchtext.data import ReversibleField
from types import SimpleNamespace

from previous_works.model_wrappers.base_model import (BaseModel,
                                                      data2file_decorator,
                                                      empty_sentence_remover_decorator,
                                                      remove_extra_rows)


class TexyGen(BaseModel):

    def __init__(self, model_identifier: SimpleNamespace, parser: ReversibleField):
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
        super().__init__(model_identifier, parser)
        gan_name = model_identifier.model_name.lower()
        self.train_loc = None
        self.valid_loc = None
        self.model_class = gans[gan_name]
        self.dataloader_class = dls[gan_name]

    @data2file_decorator(delete_tempfile=False)
    def init_model(self, train_samples, valid_samples, train_samples_loc, valid_samples_loc):
        super().init_model(train_samples, valid_samples, train_samples_loc, valid_samples_loc)
        self.model = self.model_class()
        self.model.init_real_trainng(self.parser)

    def train(self):
        self.model.train_real(self.train_samples_loc, self)
        self.tracker.update_metrics(last_iter=True)

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

    def __init_nll(self, data_loc, temperature):
        from previous_works.texygen.utils.metrics.Nll import Nll
        valid_dataloader = self.dataloader_class(batch_size=self.model.batch_size,
                                                 seq_length=self.parser.max_length,
                                                 pad_token=self.parser.vocab.stoi[self.parser.pad_token])
        valid_dataloader.create_batches(data_loc)

        inll = Nll(data_loader=valid_dataloader, rnn=self.model.generator, sess=self.model.sess,
                   temperature=temperature)
        inll.set_name('nll-test-' + data_loc)
        return inll

    def __init_persample_nll(self, data_loc, temperature):
        from previous_works.texygen.utils.metrics.ItemFetcher import ItemFetcher
        dataloader = self.dataloader_class(batch_size=self.model.batch_size,
                                           seq_length=self.parser.max_length,
                                           pad_token=self.parser.vocab.stoi[self.parser.pad_token])
        dataloader.create_batches(data_loc)
        if temperature['value'] is None:
            item_to_be_fetched = self.model.generator.selfdefined_persample_nll
        elif temperature['type'] == 'biased':
            item_to_be_fetched = self.model.generator.selfdefined_temp_persample_nll
        elif temperature['type'] == 'unbiased':
            item_to_be_fetched = self.model.generator.unbiased_temperature_persample_nll
        else:
            raise BaseException('invalid temperature type!')
        inll = ItemFetcher(data_loader=dataloader, rnn=self.model.generator,
                           item_to_be_fetched=item_to_be_fetched,
                           sess=self.model.sess, temperature=temperature)
        inll.set_name('persample_ll' + data_loc)
        return inll

    @data2file_decorator(delete_tempfile=True)
    def get_nll(self, samples, samples_loc, temperature):
        nll = self.__init_nll(samples_loc, temperature)
        score = nll.get_score()
        return float(score)

    @remove_extra_rows
    @data2file_decorator(delete_tempfile=True)
    def get_persample_nll(self, samples, samples_loc, temperature):
        persample_nll = self.__init_persample_nll(samples_loc, temperature)
        score = persample_nll.get_score()
        return [float(s) for s in score]

    def get_training_epoch_threshold_for_evaluation(self):
        if self.get_name() != "mle":
            # return {"neg_nll": 80, "neg_fbd": 80}
            return 80
        return super().get_training_epoch_threshold_for_evaluation()

    def get_saving_path(self):
        return self.model_class.saving_path

    def get_name(self):
        return self.model_class.__name__.lower()

    def load(self):
        super().load()
        import tensorflow as tf
        self.model.sess.run(tf.global_variables_initializer())
        self.model.sess.run(tf.local_variables_initializer())
        self.model.load_generator_discriminator()

    def reset_model(self):
        import tensorflow as tf
        tf.reset_default_graph()


if __name__ == "__main__":
    from data_management.data_manager import load_real_dataset

    train_ds, valid_ds, test_ds, parser = load_real_dataset('coco')
    print(train_ds[0].text)

    m = TexyGen('seqgan', parser)
    m.init_model()

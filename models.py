import tensorflow as tf

from data_management.data_loaders import SentenceDataloader
from data_management.data_manager import SentenceDataManager
from data_management.parsers import Parser


class BaseModel:
    def __init__(self, evaluator):
        self.evaluator = evaluator

    def train(self):
        pass

    def generate(self, n_samples, with_beam_search=False):
        pass

    def get_nll(self, data_loc):
        pass

    def update_results(self):
        if self.evaluator is None:
            print('evaluator called!')
            return
        self.evaluator.evaluate()

    def reset(self):
        pass


class TexyGen(BaseModel):
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

    def __init__(self, gan_name, train_loc, valid_loc, parser: Parser, evaluator=None):
        super().__init__(evaluator)
        self.train_loc = train_loc
        self.valid_loc = valid_loc
        self.parser = parser
        self.model = TexyGen.gans[gan_name.lower()]()
        self.dataloader_class = TexyGen.dls[gan_name.lower()]
        self.model.init_real_trainng(train_loc, self.parser)

        self.valid_nll = self.init_nll(valid_loc)
        self.model.sess.run(tf.global_variables_initializer())
        self.model.sess.run(tf.local_variables_initializer())
        self.model.load_generator_discriminator()

    def train(self):
        # move data to data location of model
        self.model.train_real(self.train_loc, self)

    def generate_samples(self, n_samples, with_beam_search=False):
        print('generating {} samples from {}!'.format(n_samples, self.model.__class__.__name__))
        if self.model.__class__.__name__ == 'Leakgan':
            from previous_works.texygen.models.leakgan import Leakgan
            codes = Leakgan.generate_samples_gen(self.model.sess, self.model.generator, self.model.batch_size,
                                                 n_samples, self.model.test_file)
        else:
            from previous_works.texygen.utils.utils import generate_samples
            codes = generate_samples(self.model.sess, self.model.generator,
                                     self.model.batch_size, n_samples, self.model.test_file)
        print('samples generated!')

        from previous_works.texygen.utils.text_process import code_to_text
        text = code_to_text(codes, self.parser.id2vocab, self.model.test_file, eof_code=self.parser.END_TOKEN_ID)
        return text

    def init_nll(self, data_loc):
        from previous_works.texygen.utils.metrics.Nll import Nll
        valid_dataloader = self.dataloader_class(batch_size=self.model.batch_size,
                                                 seq_length=self.model.sequence_length)
        valid_dataloader.create_batches(data_loc)

        inll = Nll(data_loader=valid_dataloader, rnn=self.model.generator, sess=self.model.sess)
        inll.set_name('nll-test-' + data_loc)
        return inll

    def get_nll(self, data_loc=None):
        if data_loc is not None:
            inll = self.init_nll(data_loc, self)
        else:
            inll = self.valid_nll
        return inll.get_score()

    def reset(self):
        tf.reset_default_graph()


dm = SentenceDataManager([SentenceDataloader('coco-train')], 'coco-words')
train_data = dm.get_training_data(0, unpack=True)
valid_data = dm.get_validation_data(0, unpack=True)
tr_loc = dm.dump_unpacked_data_on_file(train_data, 'coco-train-k0')
valid_loc = dm.dump_unpacked_data_on_file(valid_data, 'coco-valid-k0')
m = TexyGen("seqgan", tr_loc, valid_loc, dm.get_parser())
# m.train()
print(m.get_nll(None))
print(m.generate_samples(64))

import tensorflow as tf

from data_management.data_loaders import SentenceDataloader
from data_management.data_manager import SentenceDataManager
from data_management.parsers import Parser
from file_handler import write_text


class BaseModel:
    def __init__(self):
        self.dumper = None
        pass

    def update_scores(self, epoch_num):
        print('evaluator called %d!' % epoch_num)
        assert self.dumper is not None, 'Evaluator is none!'
        self.dumper.update_scores(epoch_num)

    def set_dumper(self, dumper):
        self.dumper = dumper

    def set_train_val_loc(self, train_loc, valid_loc):
        self.train_loc = train_loc
        self.valid_loc = valid_loc

    def delete_saved_model(self):
        import os, shutil
        if os.path.exists(self.get_saving_path()):
            shutil.rmtree(self.get_saving_path())
            os.mkdir(self.get_saving_path())
            print("saved model at %s deleted!" % self.get_saving_path())

    def train(self):
        pass

    def generate_samples(self, n_samples, with_beam_search=False):
        pass

    def get_nll(self, data_loc=None):
        pass

    def delete(self):
        tf.reset_default_graph()

    def get_saving_path(self):
        pass

    def get_name(self):
        pass

    def load(self):
        pass


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
        super().__init__()
        self.train_loc = None
        self.valid_loc = None
        self.parser = parser
        self.model = gans[gan_name.lower()]()
        self.dataloader_class = dls[gan_name.lower()]

    def set_train_val_loc(self, train_loc, valid_loc):
        super().set_train_val_loc(train_loc, valid_loc)
        self.model.init_real_trainng(train_loc, self.parser)
        self.valid_nll = self.init_nll(valid_loc)
        self.load()

    def train(self):
        self.model.train_real(self.train_loc, self)
        self.dumper.update_scores(last_iter=True)

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
        print('%d samples generated!' % len(codes))

        samples = self.parser.id_format2line(codes, True)
        write_text(samples, self.model.test_file, True)
        print('codes converted to text format and written in file %s.' % self.model.test_file)
        return samples

    def init_nll(self, data_loc):
        from previous_works.texygen.utils.metrics.Nll import Nll
        valid_dataloader = self.dataloader_class(batch_size=self.model.batch_size,
                                                 seq_length=self.model.sequence_length)
        valid_dataloader.create_batches(data_loc)

        inll = Nll(data_loader=valid_dataloader, rnn=self.model.generator, sess=self.model.sess)
        inll.set_name('nll-test-' + data_loc)
        return inll

    def get_nll(self, data_loc=None):
        if data_loc is None or data_loc == self.valid_loc:
            inll = self.valid_nll
        else:
            inll = self.init_nll(data_loc, self)
        return inll.get_score()

    def get_saving_path(self):
        return self.model.saving_path

    def get_name(self):
        return self.model.__class__.__name__

    def load(self):
        self.model.sess.run(tf.global_variables_initializer())
        self.model.sess.run(tf.local_variables_initializer())
        self.model.load_generator_discriminator()


class LeakGan(BaseModel):
    def __init__(self, parser: Parser):
        super().__init__()
        import previous_works.leakgan2.Main as leakgan2main
        self.parser = parser
        self.model_module = leakgan2main

    def set_train_val_loc(self, train_loc, valid_loc):
        super().set_train_val_loc(train_loc, valid_loc)
        with open(train_loc) as ref:
            with open(self.model_module.positive_file, 'w') as trg:
                trg.write(''.join(ref.readlines()))
        self.valid_data_loader = self.model_module.Gen_Data_loader(self.model_module.BATCH_SIZE, self.parser.max_length)
        self.valid_data_loader.create_batches(valid_loc)
        self.model = self.model_module.LeakGanMain(self, self.parser)

    def train(self):
        self.model.train()
        self.dumper.update_scores(last_iter=True)

    def generate_samples(self, n_samples, with_beam_search=False):
        codes = self.model.generate_samples(n_samples, self.model_module.dummy_file, 0)

        print('%d samples generated!' % len(codes))

        samples = self.parser.id_format2line(codes, True)
        write_text(samples, self.model_module.dummy_file, True)
        print('codes converted to text format and written in file %s.' % self.model_module.dummy_file)
        return samples

    def get_nll(self, data_loc=None):
        if data_loc is None or data_loc == self.valid_loc:
            dl = self.valid_data_loader
        else:
            dl = self.model_module.Gen_Data_loader(self.model_module.BATCH_SIZE, self.parser.max_length)
            dl.create_batches(data_loc)
        return self.model.target_loss(dl)

    def get_saving_path(self):
        return self.model_module.model_path

    def get_name(self):
        return 'leakgan2'

    def load(self):
        self.model.sess.run(tf.local_variables_initializer())
        self.model.sess.run(tf.global_variables_initializer())
        model_path = tf.train.latest_checkpoint(self.get_saving_path())
        if model_path is not None:
            self.model.saver.restore(self.model.sess, model_path)
            print(model_path + ' found!')
        else:
            print('Model not found. Randomly initialized!')


if __name__ == '__main__':
    dm = SentenceDataManager([SentenceDataloader('coco-train')], 'coco-words')
    train_data = dm.get_training_data(0, unpack=True)
    valid_data = dm.get_validation_data(0, unpack=True)
    tr_loc = dm.dump_unpacked_data_on_file(train_data, 'coco-train-k0')
    valid_loc = dm.dump_unpacked_data_on_file(valid_data, 'coco-valid-k0')
    # m_name = 'seqgan'
    # m_name = 'maligan'
    # m_name = 'leakgan'
    # m_name = 'rankgan'
    m_name = 'textgan'
    m = TexyGen(m_name, dm.get_parser())
    # m.train()
    print(m.get_nll(None))
    print(m.generate_samples(64))

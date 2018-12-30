from data_management.data_loaders import SentenceDataloader
from data_management.data_manager import SentenceDataManager
from data_management.parsers import Parser
from utils.file_handler import write_text, delete_file
from utils.path_configs import TEMP_PATH


class BaseModel:
    def __init__(self):
        self.tracker = None
        pass

    def update_scores(self, epoch_num):
        print('evaluator called %d!' % epoch_num)
        assert self.tracker is not None, 'Evaluator is none!'
        self.tracker.update_scores(epoch_num)

    def set_tracker(self, dumper):
        self.tracker = dumper

    def set_train_val_data(self, train_data, valid_data):
        self.train_data = train_data
        self.valid_data = valid_data

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

    def get_nll(self, samples=None):
        pass

    def get_persample_ll(self, samples=None):
        pass

    def delete(self):
        import tensorflow as tf
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
        self.model_class = gans[gan_name.lower()]
        self.dataloader_class = dls[gan_name.lower()]

    def __del__(self):
        if self.train_loc is not None:
            delete_file(self.train_loc)

    def set_train_val_data(self, train_data, valid_data):
        train_data = [' '.join(l) for l in self.parser.line2id_format(train_data)]
        valid_data = [' '.join(l) for l in self.parser.line2id_format(valid_data)]
        super().set_train_val_data(train_data, valid_data)
        self.train_loc = write_text(train_data, self.get_name(), TEMP_PATH)
        self.model = self.model_class()
        self.model.init_real_trainng(self.train_loc, self.parser)
        self.load()

    def train(self):
        self.model.train_real(self.train_loc, self)
        self.tracker.update_scores(last_iter=True)

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

    def init_persample_ll(self, data_loc):
        from previous_works.texygen.utils.metrics.ItemFetcher import ItemFetcher
        dataloader = self.dataloader_class(batch_size=self.model.batch_size,
                                           seq_length=self.model.sequence_length)
        dataloader.create_batches(data_loc)

        inll = ItemFetcher(data_loader=dataloader, rnn=self.model.generator,
                           item_to_be_fetched=self.model.generator.selfdefined_persample_ll,
                           sess=self.model.sess)
        inll.set_name('persample_ll' + data_loc)
        return inll

    def get_nll(self, samples=None):
        samples = [' '.join(l) for l in self.parser.line2id_format(samples)]
        if samples is None:
            data_loc = write_text(self.valid_data, self.get_name(), TEMP_PATH)
        else:
            data_loc = write_text(samples, self.get_name(), TEMP_PATH)
        inll = self.init_nll(data_loc)
        score = inll.get_score()
        delete_file(data_loc)
        return score

    def get_persample_ll(self, samples=None):
        samples = [' '.join(l) for l in self.parser.line2id_format(samples)]
        if samples is None:
            data_loc = write_text(self.valid_data, self.get_name(), TEMP_PATH)
        else:
            data_loc = write_text(samples, self.get_name(), TEMP_PATH)
        ll = self.init_persample_ll(data_loc)
        score = ll.get_score()
        delete_file(data_loc)
        return score

    def get_saving_path(self):
        return self.model_class.saving_path

    def get_name(self):
        return self.model_class.__name__

    def load(self):
        import tensorflow as tf
        self.model.sess.run(tf.global_variables_initializer())
        self.model.sess.run(tf.local_variables_initializer())
        self.model.load_generator_discriminator()


class LeakGan(BaseModel):
    def __init__(self, parser: Parser):
        super().__init__()
        import previous_works.leakgan2.Main as leakgan2main
        self.parser = parser
        self.model_module = leakgan2main

    def set_train_val_data(self, train_data, valid_data):
        train_data = [' '.join(l) for l in self.parser.line2id_format(train_data)]
        valid_data = [' '.join(l) for l in self.parser.line2id_format(valid_data)]
        super().set_train_val_data(train_data, valid_data)
        with open(self.model_module.positive_file, 'w') as trg:
            trg.write('\n'.join(train_data))
        self.model = self.model_module.LeakGanMain(self, self.parser)

    def train(self):
        self.model.train()
        self.tracker.update_scores(last_iter=True)

    def generate_samples(self, n_samples, with_beam_search=False):
        codes = self.model.generate_samples(n_samples, self.model_module.dummy_file, 0)
        print('%d samples generated!' % len(codes))

        samples = self.parser.id_format2line(codes, True)
        write_text(samples, self.model_module.dummy_file, True)
        print('codes converted to text format and written in file %s.' % self.model_module.dummy_file)
        return samples

    def get_nll(self, samples=None):
        samples = [' '.join(l) for l in self.parser.line2id_format(samples)]
        if samples is None:
            data_loc = write_text(self.valid_data, self.get_name(), TEMP_PATH)
        else:
            data_loc = write_text(samples, self.get_name(), TEMP_PATH)
        dl = self.model_module.Gen_Data_loader(self.model_module.BATCH_SIZE, self.parser.max_length)
        dl.create_batches(data_loc)
        score = self.model.target_loss(dl)
        delete_file(data_loc)
        return score

    def get_persample_ll(self, samples=None):
        samples = [' '.join(l) for l in self.parser.line2id_format(samples)]
        if samples is None:
            data_loc = write_text(self.valid_data, self.get_name(), TEMP_PATH)
        else:
            data_loc = write_text(samples, self.get_name(), TEMP_PATH)
        dl = self.model_module.Gen_Data_loader(self.model_module.BATCH_SIZE, self.parser.max_length)
        dl.create_batches(data_loc)
        score = self.model.per_sample_ll(dl)
        delete_file(data_loc)
        return score

    def get_saving_path(self):
        return self.model_module.model_path

    def get_name(self):
        return 'leakgan2'

    def load(self):
        import tensorflow as tf
        self.model.sess.run(tf.local_variables_initializer())
        self.model.sess.run(tf.global_variables_initializer())
        model_path = tf.train.latest_checkpoint(self.get_saving_path())
        if model_path is not None:
            self.model.saver.restore(self.model.sess, model_path)
            print(model_path + ' found!')
        else:
            print('Model not found. Randomly initialized!')


class TextGan(BaseModel):
    def __init__(self, parser: Parser):
        super().__init__()
        self.parser = parser

    def set_train_val_data(self, train_data, valid_data):
        super().set_train_val_data(train_data, valid_data)
        from previous_works.textgan2.textGAN import TextGANMMD
        self.model = TextGANMMD(self, self.parser, train_data, valid_data)

    def train(self):
        self.model.train_func()
        self.tracker.update_scores(last_iter=True)

    def generate_samples(self, n_samples, with_beam_search=False):
        codes = self.model.generate()
        samples = self.parser.id_format2line(codes, True)
        # write_text(samples, self.model_module.dummy_file, True)
        # print('codes converted to text format and written in file %s.' % self.model_module.dummy_file)
        return samples

    def get_nll(self, data_loc=None):
        return 0.0

    def delete(self):
        super().delete()

    def get_saving_path(self):
        super().get_saving_path()

    def get_name(self):
        return 'textgan2'

    def load(self):
        super().load()


if __name__ == '__main__':
    dm = SentenceDataManager([SentenceDataloader('coco-train')], 'coco-words', k_fold=3)
    # train_data = dm.get_training_data(0, unpack=True)
    # valid_data = dm.get_validation_data(0, unpack=True)
    # tr_loc = dm.dump_unpacked_data_on_file(train_data, 'coco-train-k0')
    # valid_loc = dm.dump_unpacked_data_on_file(valid_data, 'coco-valid-k0')
    # m = TextGANMMD(dm.get_parser(), train_data, valid_data)
    # m.train_func()

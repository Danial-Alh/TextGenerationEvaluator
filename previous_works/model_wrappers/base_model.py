from data_management.data_loaders import SentenceDataloader
from data_management.data_manager import SentenceDataManager
from data_management.parsers import Parser
from utils.file_handler import write_text, delete_file
from utils.path_configs import TEMP_PATH


def empty_sentence_remover_decorator(func):
    def wrapper(self, n_samples, *args, **kwargs):
        print('generating {} samples from {}!'.format(n_samples, self.model.__class__.__name__))
        result = func(self, n_samples, *args, **kwargs)
        result = [result[i] for i, r in enumerate(self.parser.id_format2line(result)) if len(r) > 0]
        print('{} samples generated from {}!'.format(len(result), self.model.__class__.__name__))
        return result

    return wrapper


def data2tempfile_decorator(func):
    def wrapper(self, temperature, samples=None, samples_loc=None):
        assert not (samples is None and samples_loc is not None)
        file_to_be_deleted = None
        if samples is None:
            samples = self.valid_data
        if samples_loc is None:
            stringified_samples = [' '.join(list(map(str, s))) for s in samples]
            samples_loc = write_text(stringified_samples, self.get_name() +
                                     '_{}'.format(func.__name__), TEMP_PATH)
            file_to_be_deleted = samples_loc
        result = func(self, temperature, samples, samples_loc)
        if file_to_be_deleted is not None:
            delete_file(file_to_be_deleted)
        return result

    return wrapper


class BaseModel:
    def __init__(self, parser: Parser):
        self.tracker = None
        self.model = None
        self.parser = parser
        self.train_data = None
        self.valid_data = None
        self.train_loc = None
        self.valid_loc = None
        pass

    def __del__(self):
        # if self.train_loc is not None:
        #     print(delete_file)
        # if self.valid_loc is not None:
        #     delete_file(self.valid_loc)
        pass

    def update_scores(self, epoch_num):
        print('evaluator called %d!' % epoch_num)
        assert self.tracker is not None, 'Evaluator is none!'
        self.tracker.update_scores(epoch_num)

    def set_tracker(self, dumper):
        self.tracker = dumper

    def set_train_val_data(self, train_data, valid_data):
        assert train_data is not None and valid_data is not None
        self.train_data = train_data
        self.valid_data = valid_data
        self.train_loc = write_text([' '.join(list(map(str, s))) for s in train_data],
                                    self.get_name() + '_train', TEMP_PATH)
        self.valid_loc = write_text([' '.join(list(map(str, s))) for s in valid_data],
                                    self.get_name() + '_valid', TEMP_PATH)

    def delete_saved_model(self):
        import os
        import shutil
        if os.path.exists(self.get_saving_path()):
            shutil.rmtree(self.get_saving_path())
            os.mkdir(self.get_saving_path())
            print("saved model at %s deleted!" % self.get_saving_path())

    def create_model(self):
        pass

    def train(self):
        pass

    @empty_sentence_remover_decorator
    def generate_samples(self, n_samples, temperature):
        pass

    @data2tempfile_decorator
    def get_nll(self, temperature, samples=None, samples_loc=None):
        pass

    @data2tempfile_decorator
    def get_persample_ll(self, temperature, samples=None, samples_loc=None):
        pass

    def delete(self):
        import tensorflow as tf
        tf.reset_default_graph()

    def get_saving_path(self):
        pass

    def load(self):
        pass

    def get_name(self):
        return self.__class__.__name__


if __name__ == '__main__':
    dm = SentenceDataManager([SentenceDataloader('coco60-train')], 'coco-words', k_fold=3)
    # train_data = dm.get_training_data(0, unpack=True)
    # valid_data = dm.get_validation_data(0, unpack=True)
    # tr_loc = dm.dump_unpacked_data_on_file(train_data, 'coco-train-k0')
    # valid_loc = dm.dump_unpacked_data_on_file(valid_data, 'coco-valid-k0')
    m = TexyGen('seqgan', dm.get_parser())
    m.set_train_val_data(*dm.get_data(k=0, parse=False))
    # print(dm.get_parser().id_format2line(m.generate_samples(100)))
    # print(m.get_nll(dm.get_data(k=0)['valid']))
    # print(m.get_persample_ll(dm.get_data(k=0, subsample_size=100)['valid']))
    # m.train_func()

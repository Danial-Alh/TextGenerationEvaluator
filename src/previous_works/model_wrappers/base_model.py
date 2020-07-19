import os
import shutil
from torchtext.data import ReversibleField
import torch

from data_management.data_manager import load_real_dataset
from utils.file_handler import delete_file, write_text
from utils.path_configs import TEMP_PATH


def empty_sentence_remover_decorator(func):
    def wrapper(self, n_samples, *args, **kwargs):
        print('generating {} samples from {}!'.format(n_samples, self.get_name()))
        result = func(self, n_samples, *args, **kwargs)
        result = torch.tensor([list(map(int, l)) for l in result])
        result = self.parser.denumericalize(result)
        result = list(filter(lambda x: len(x) > 0, result))
        print('{} samples generated from {}!'.format(len(result), self.get_name()))
        return result

    return wrapper


def data2tempfile_decorator(func):
    def wrapper(self, samples, *args, **kwargs):
        assert samples is not None

        samples = self.parser.pad(samples)
        samples = self.parser.numericalize(samples)

        stringified_samples = [' '.join(map(str, s)) for s in samples.tolist()]
        samples_loc = write_text(stringified_samples, '{}_{}'.format(self.get_name(), func.__name__),
                                 TEMP_PATH)
        result = func(self, samples, samples_loc, *args, **kwargs)
        delete_file(samples_loc)
        return result

    return wrapper


def remove_extra_rows(func):
    def wrapper(self, samples, *args, **kwargs):
        assert samples is not None
        result = func(self, samples, *args, **kwargs)
        assert len(samples) <= len(result)
        return result[:len(samples)]

    return wrapper


class BaseModel:
    def __init__(self, parser: ReversibleField):
        self.tracker = None
        self.model = None
        self.parser = parser
        self.train_data = None
        self.valid_data = None
        self.train_loc = None
        self.valid_loc = None

    def __del__(self):
        # if self.train_loc is not None:
        #     print(delete_file)
        # if self.valid_loc is not None:
        #     delete_file(self.valid_loc)
        pass

    def update_metrics(self, epoch_num):
        print('evaluator called %d!' % epoch_num)
        assert self.tracker is not None, 'Evaluator is none!'
        self.tracker.update_metrics(epoch_num)

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

    def init_model(self):
        pass

    def train(self, samples, samples_loc):
        pass

    @empty_sentence_remover_decorator
    def generate_samples(self, n_samples, temperature):
        pass

    @data2tempfile_decorator
    def get_nll(self, samples, samples_loc, temperature):
        pass

    @remove_extra_rows
    @data2tempfile_decorator
    def get_persample_nll(self, samples, samples_loc, temperature):
        pass

    def reset_model(self):
        pass

    def delete_saved_model(self):
        if os.path.exists(self.get_saving_path()):
            shutil.rmtree(self.get_saving_path())
            os.mkdir(self.get_saving_path())
            print("saved model at %s deleted!" % self.get_saving_path())

    def get_saving_path(self):
        pass

    def load(self):
        pass

    def get_name(self):
        return self.__class__.__name__


if __name__ == '__main__':
    train_ds, valid_ds, test_ds, parser = load_real_dataset('coco')
    print(train_ds[0].text)

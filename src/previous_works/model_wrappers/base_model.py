import os
import shutil
from functools import wraps
from types import SimpleNamespace

import torch
from torchtext.data import ReversibleField

from data_management.data_manager import load_real_dataset
from utils.file_handler import delete_file, write_text
from utils.path_configs import TEMP_PATH


def empty_sentence_remover_decorator(func):
    @wraps(func)
    def wrapper(self, n_samples, *args, **kwargs):
        print('generating {} samples from {}!'.format(n_samples, self.get_name()))
        result = func(self, n_samples, *args, **kwargs)
        result = torch.tensor([list(map(int, l)) for l in result])
        result = self.parser.denumericalize(result)
        result = list(filter(lambda x: len(x) > 0, result))
        print('{} samples generated from {}!'.format(len(result), self.get_name()))
        return result

    return wrapper


def convert_and_write_samples_to_temp_file(samples, file_name, parser):
    assert samples is not None

    samples = parser.pad(samples)
    samples = parser.numericalize(samples)

    stringified_samples = [' '.join(map(str, s)) for s in samples.tolist()]
    samples_loc = write_text(stringified_samples, file_name, TEMP_PATH)
    return samples_loc


def data2file_decorator(delete_tempfile: bool):

    def inner_decorator(func):
        @wraps(func)
        def wrapper(self, samples, *args, **kwargs):
            if not isinstance(samples, tuple):
                samples = (samples,)
            samples = [list(s) for s in samples]
            samples_locs = [
                convert_and_write_samples_to_temp_file
                (
                    s,
                    '{}_{}_{}'.format(self.get_name(), func.__name__, i),
                    self.parser
                )
                for i, s in enumerate(samples)
            ]

            result = func(self, *samples, *samples_locs, *args, **kwargs)

            if delete_tempfile:
                for s in samples_locs:
                    delete_file(s)
            return result
        return wrapper

    return inner_decorator


def remove_extra_rows(func):
    def wrapper(self, samples, *args, **kwargs):
        assert samples is not None
        result = func(self, samples, *args, **kwargs)
        assert len(samples) <= len(result)
        return result[:len(samples)]

    return wrapper


class BaseModel:
    def __init__(self, model_identifier: SimpleNamespace, parser: ReversibleField):
        self.tracker = None
        self.model = None
        self.parser = parser

        self.train_data = None
        self.valid_data = None
        self.train_loc = None
        self.valid_loc = None

        self.run = model_identifier.run
        self.train_temperature = model_identifier.train_temperature
        self.test_temperature = model_identifier.test_temperature
        self.restore_type = model_identifier.restore_type

    def __del__(self):
        # if self.train_loc is not None:
        #     print(delete_file)
        # if self.valid_loc is not None:
        #     delete_file(self.valid_loc)
        pass

    def update_metrics(self, epoch):
        print('evaluator called %d!' % epoch)
        assert self.tracker is not None, 'Evaluator is none!'
        self.tracker.update_metrics(epoch=epoch)

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

    def init_model(self, train_samples, valid_samples, train_samples_loc, valid_samples_loc):
        self.train_samples = train_samples
        self.train_samples_loc = train_samples_loc
        self.valid_samples = valid_samples
        self.valid_samples_loc = valid_samples_loc

    def train(self):
        pass

    @empty_sentence_remover_decorator
    def generate_samples(self, n_samples, temperature):
        pass

    @data2file_decorator(delete_tempfile=True)
    def get_nll(self, samples, samples_loc, temperature):
        pass

    @remove_extra_rows
    @data2file_decorator(delete_tempfile=True)
    def get_persample_nll(self, samples, samples_loc, temperature):
        pass

    def get_training_epoch_threshold_for_evaluation(self):
        return -1

    def reset_model(self):
        pass

    def delete_saved_model(self):
        if os.path.exists(self.get_saving_path()):
            shutil.rmtree(self.get_saving_path())
            print("saved model at %s deleted!" % self.get_saving_path())
        os.mkdir(self.get_saving_path())
        print("saving path created at %s!" % self.get_saving_path())

    def get_saving_path(self):
        pass

    def load(self):
        pass

    def get_name(self):
        return self.__class__.__name__


class DummyModel(BaseModel):
    def __init__(self, model_identifier: SimpleNamespace, parser: ReversibleField):
        super().__init__(model_identifier, parser)
        self.model_name = model_identifier.model_name.lower()

    def get_name(self):
        return self.model_name


if __name__ == '__main__':
    train_ds, valid_ds, test_ds, TEXT = load_real_dataset('coco')
    print(train_ds[0].text)

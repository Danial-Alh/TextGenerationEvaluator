import numpy as np
import torch as t
from torch.optim import Adam
from torchtext.data import ReversibleField

from previous_works.model_wrappers.base_model import (
    BaseModel,
    data2file_decorator,
    empty_sentence_remover_decorator)


class VAE(BaseModel):
    def __init__(self, parser: ReversibleField):
        super().__init__(parser)
        from previous_works.rvae.train import SAVING_PATH
        self.saving_path = SAVING_PATH

    @data2file_decorator(delete_tempfile=False)
    def init_model(self, train_samples, valid_samples, train_samples_loc, valid_samples_loc):
        super().init_model(train_samples, valid_samples, train_samples_loc, valid_samples_loc)
        from previous_works.rvae.train import create_model as crm, create_batchloader as crb, create_parameters as crp
        self.batchloader = crb(train_samples_loc, valid_samples_loc, self.parser)
        self.parameters = crp(self.batchloader)
        self.model = crm(self.parameters)

    def train(self):
        from previous_works.rvae.train import train as tr
        tr(self.model, self.batchloader, self.parameters, self)
        self.tracker.update_metrics(last_iter=True)

    @empty_sentence_remover_decorator
    def generate_samples(self, n_samples, temperature):
        from previous_works.rvae.train import sample as smp
        samples = smp(self.model, self.batchloader, n_samples, self.parser.max_length)
        return samples

    @data2file_decorator(delete_tempfile=True)
    def get_nll(self, samples, samples_loc, temperature):
        return np.inf

    @data2file_decorator(delete_tempfile=True)
    def get_persample_nll(self, samples, samples_loc, temperature):
        return np.ones(len(samples)) * np.inf

    def get_saving_path(self):
        return self.saving_path

    def get_name(self):
        return 'vae'

    def load(self):
        from previous_works.rvae.train import load as lo
        lo(self.model)

    def reset_model(self):
        from previous_works.rvae.train import create_model as crm
        self.model = crm(self.parameters)


if __name__ == "__main__":
    from data_management.data_manager import load_real_dataset

    train_ds, valid_ds, test_ds, parser = load_real_dataset('ptb')
    print(train_ds[0].text)

    m = VAE(parser)
    m.delete_saved_model()
    m.init_model((train_ds.text, valid_ds.text))
    m.train()
    m.load()
    m.delete()

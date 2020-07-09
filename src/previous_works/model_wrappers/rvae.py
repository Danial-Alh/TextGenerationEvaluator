import numpy as np
import torch as t
from torch.optim import Adam

from data_management.parsers import Parser
from previous_works.model_wrappers.base_model import BaseModel, empty_sentence_remover_decorator, \
    data2tempfile_decorator
from previous_works.rvae.model.rvae import RVAE
from previous_works.rvae.utils.batch_loader import BatchLoader
from previous_works.rvae.utils.parameters import Parameters
from utils.path_configs import ROOT_PATH


class VAE(BaseModel):
    def __init__(self, parser: Parser):
        super().__init__(parser)
        self.BATCH_SIZE = 128
        self.DROPOUT = 0.3
        self.ROOT_PATH = ROOT_PATH + 'previous_works/rvae/'

    def create_model(self):
        pass

    def set_train_val_data(self, train_data, valid_data):
        super().set_train_val_data(train_data, valid_data)
        # data/train.txt
        self.parameters = Parameters(self.batch_loader.max_word_len,
                                     self.batch_loader.max_seq_len,
                                     self.batch_loader.words_vocab_size,
                                     self.batch_loader.chars_vocab_size)

        self.rvae = RVAE(self.parameters).cuda()
        self.batch_loader = BatchLoader(path=self.ROOT_PATH, data_files=[self.train_loc, self.valid_loc])

    def train(self):
        optimizer = Adam(self.rvae.learnable_parameters(), 1e-3)
        train_step = self.rvae.trainer(optimizer, self.batch_loader)
        validate = self.rvae.validater(self.batch_loader)

        ce_result = []
        kld_result = []
        for iteration in range(80 * self.batch_loader.num_lines['train'] / self.BATCH_SIZE):
            cross_entropy, kld, coef = train_step(iteration, self.BATCH_SIZE, True, self.DROPOUT)
            if iteration % 5 == 0:
                print('\n')
                print('------------TRAIN-------------')
                print('----------ITERATION-----------')
                print(iteration)
                print('--------CROSS-ENTROPY---------')
                print(cross_entropy.data.cpu().numpy()[0])
                print('-------------KLD--------------')
                print(kld.data.cpu().numpy()[0])
                print('-----------KLD-coef-----------')
                print(coef)
                print('------------------------------')

            if iteration % 10 == 0:
                cross_entropy, kld = validate(self.BATCH_SIZE, True)
                cross_entropy = cross_entropy.data.cpu().numpy()[0]
                kld = kld.data.cpu().numpy()[0]
                print('\n')
                print('------------VALID-------------')
                print('--------CROSS-ENTROPY---------')
                print(cross_entropy)
                print('-------------KLD--------------')
                print(kld)
                print('------------------------------')
                ce_result += [cross_entropy]
                kld_result += [kld]

            if iteration % 20 == 0:
                seed = np.random.normal(size=[1, self.parameters.latent_variable_size])
                sample = self.rvae.sample(self.batch_loader, 50, seed, True)
                print('\n')
                print('------------SAMPLE------------')
                print('------------------------------')
                print(sample)
                print('------------------------------')

        t.save(self.rvae.state_dict(), 'saved/trained_RVAE')

    @empty_sentence_remover_decorator
    def generate_samples(self, n_samples, temperature):
        samples = []
        for iteration in range(n_samples):
            seed = np.random.normal(size=[1, self.parameters.latent_variable_size])
            result = self.rvae.sample(self.batch_loader, 50, seed, True)
            samples.append(result)
            print(result)
            print()
        return samples

    @data2tempfile_decorator
    def get_nll(self, temperature, samples=None, samples_loc=None):
        return 0.0

    @data2tempfile_decorator
    def get_persample_ll(self, temperature, samples=None, samples_loc=None):
        return np.zeros(len(samples))

    def get_saving_path(self):
        return self.ROOT_PATH + 'saved/'

    def get_name(self):
        return 'vae'

    def load(self):
        self.parameters = Parameters(self.batch_loader.max_word_len,
                                     self.batch_loader.max_seq_len,
                                     self.batch_loader.words_vocab_size,
                                     self.batch_loader.chars_vocab_size)

        self.rvae = RVAE(self.parameters).cuda()
        self.batch_loader = BatchLoader('')
        self.rvae.load_state_dict(t.load('saved/trained_RVAE'))

    
    def delete(self):
        raise BaseException('not implemented!')

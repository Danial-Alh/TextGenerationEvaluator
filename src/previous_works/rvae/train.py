import argparse
import inspect
import os
from types import SimpleNamespace

import numpy as np
import torch as t
from torch.optim import Adam

from previous_works.rvae.utils.batch_loader import BatchLoader

from .model.rvae import RVAE
from .utils.batch_loader import BatchLoader
from .utils.parameters import Parameters

CURRENT_ROOT_PATH = os.path.abspath(
    os.path.dirname(
        os.path.abspath(
            inspect.getfile(inspect.currentframe())
        )
    )
) + '/'

SAVING_PATH = CURRENT_ROOT_PATH + '__saved/'


def create_batchloader(train_samples_loc, valid_samples_loc):
    batch_loader = BatchLoader(
        CURRENT_ROOT_PATH,
        data_files=[
            train_samples_loc,
            valid_samples_loc
        ]
    )
    return batch_loader


def create_parameters(batch_loader=None):
    if batch_loader is not None:
        parameters = Parameters(batch_loader.max_word_len,
                                batch_loader.max_seq_len,
                                batch_loader.words_vocab_size,
                                batch_loader.chars_vocab_size)
    else:
        parameters = Parameters(10, 10, 10, 10)

    return parameters


def create_model(parameters):
    rvae = RVAE(parameters)
    return rvae


def load(rvae: t.nn.Module):
    rvae.load_state_dict(t.load(CURRENT_ROOT_PATH + 'saved/trained_RVAE'))


def train(rvae, batch_loader, parameters, wrapper):
    # if not os.path.exists('saved/word_embeddings.npy'):
    #     raise FileNotFoundError("word embeddings file was't found")

    # parser = argparse.ArgumentParser(description='RVAE')
    # parser.add_argument('--num-iterations', type=int, default=120000, metavar='NI',
    #                     help='num iterations (default: 120000)')
    # parser.add_argument('--batch-size', type=int, default=32, metavar='BS',
    #                     help='batch size (default: 32)')
    # parser.add_argument('--use-cuda', type=bool, default=True, metavar='CUDA',
    #                     help='use cuda (default: True)')
    # parser.add_argument('--learning-rate', type=float, default=0.00005, metavar='LR',
    #                     help='learning rate (default: 0.00005)')
    # parser.add_argument('--dropout', type=float, default=0.3, metavar='DR',
    #                     help='dropout (default: 0.3)')
    # parser.add_argument('--use-trained', type=bool, default=False, metavar='UT',
    #                     help='load pretrained model (default: False)')
    # parser.add_argument('--ce-result', default='', metavar='CE',
    #                     help='ce result path (default: '')')
    # parser.add_argument('--kld-result', default='', metavar='KLD',
    #                     help='ce result path (default: '')')

    # args = parser.parse_args()

    args = SimpleNamespace(
        num_iterations=120000,
        batch_size=32,
        use_cuda=False,
        learning_rate=5e-5,
        dropout=0.3,
        use_trained=False,
        ce_result='',
        kld_result=''
    )

    if args.use_trained:
        load(rvae)
    if args.use_cuda:
        rvae = rvae.cuda()

    optimizer = Adam(rvae.learnable_parameters(), args.learning_rate)

    train_step = rvae.trainer(optimizer, batch_loader)
    validate = rvae.validater(batch_loader)

    ce_result = []
    kld_result = []

    for iteration in range(args.num_iterations):

        cross_entropy, kld, coef = train_step(
            iteration, args.batch_size, args.use_cuda, args.dropout)

        if iteration % 5 == 0:
            print('\n')
            print('------------TRAIN-------------')
            print('----------ITERATION-----------')
            print(iteration)
            print('--------CROSS-ENTROPY---------')
            print(cross_entropy.data.cpu().numpy())
            print('-------------KLD--------------')
            print(kld.data.cpu().numpy())
            print('-----------KLD-coef-----------')
            print(coef)
            print('------------------------------')
            wrapper.update_metrics(epoch=iteration+1)

        if iteration % 10 == 0:
            cross_entropy, kld = validate(args.batch_size, args.use_cuda)

            cross_entropy = cross_entropy.data.cpu().numpy()
            kld = kld.data.cpu().numpy()

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
            seed = np.random.normal(size=[1, parameters.latent_variable_size])

            sample = rvae.sample(batch_loader, 50, seed, args.use_cuda)

            print('\n')
            print('------------SAMPLE------------')
            print('------------------------------')
            print(sample)
            print('------------------------------')

    t.save(rvae.state_dict(), CURRENT_ROOT_PATH + 'saved/trained_RVAE')

    # np.save(CURRENT_ROOT_PATH + 'ce_result_{}.npy'.format(args.ce_result), np.array(ce_result))
    # np.save(CURRENT_ROOT_PATH + 'kld_result_npy_{}'.format(args.kld_result), np.array(kld_result))

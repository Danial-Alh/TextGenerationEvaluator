import argparse
import inspect
import os
from types import SimpleNamespace
from tqdm.auto import tqdm, trange

import numpy as np
import torch as t
import torch.nn.functional as F
from torch.autograd import Variable
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


def create_batchloader(train_samples_loc, valid_samples_loc, wrapper):
    batch_loader = BatchLoader(
        CURRENT_ROOT_PATH,
        data_files=[
            train_samples_loc,
            valid_samples_loc
        ],
        wrapper=wrapper
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
    if t.cuda.is_available():
        rvae = rvae.cuda()
    return rvae


def load(rvae: t.nn.Module):
    rvae.load_state_dict(t.load(CURRENT_ROOT_PATH + '__saved/trained_RVAE'))


def train(rvae, batch_loader, parameters, wrapper):
    # if not os.path.exists('__saved/word_embeddings.npy'):
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
        use_cuda=t.cuda.is_available(),
        learning_rate=1e-3,
        dropout=0.0,
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
    N_EPOCHS = 80
    num_batches = batch_loader.num_lines[0] // args.batch_size + 1

    for epoch in trange(N_EPOCHS):
        for _iteration in trange(num_batches):
            iteration = epoch * num_batches + _iteration

            cross_entropy, kld, coef = train_step(
                iteration, args.batch_size, args.use_cuda, args.dropout)

            if iteration % (num_batches//4) == 0:
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

        t.save(rvae.state_dict(), CURRENT_ROOT_PATH + '__saved/trained_RVAE')

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
        samples = sample(rvae, batch_loader, 1,  wrapper.parser.max_length, {'value': None})
        samples = wrapper.parser.reverse(samples)
        print('\n')
        print('------------SAMPLE------------')
        print('------------------------------')
        print(samples[0])
        print('------------------------------')
        wrapper.update_metrics(epoch=epoch+1)

    # np.save(CURRENT_ROOT_PATH + 'ce_result_{}.npy'.format(args.ce_result), np.array(ce_result))
    # np.save(CURRENT_ROOT_PATH + 'kld_result_npy_{}'.format(args.kld_result), np.array(kld_result))


def sample(rvae, batch_loader, n_samples, seq_len, temperatue):
    if temperatue['value'] is None:
        temperatue = 1.0
    with t.no_grad():
        use_cuda = t.cuda.is_available()
        seed = np.random.normal(size=[n_samples, rvae.params.latent_variable_size])

        seed = Variable(t.from_numpy(seed).float())
        if use_cuda:
            seed = seed.cuda()

        decoder_word_input_np, decoder_character_input_np = batch_loader.go_input(n_samples)

        decoder_word_input = Variable(t.from_numpy(decoder_word_input_np).long())
        decoder_character_input = Variable(t.from_numpy(decoder_character_input_np).long())

        if use_cuda:
            decoder_word_input, decoder_character_input = decoder_word_input.cuda(), decoder_character_input.cuda()

        result_ids = []

        initial_state = None

        for i in trange(seq_len):
            logits, initial_state, _ = rvae(0., None, None,
                                            decoder_word_input, decoder_character_input,
                                            seed, initial_state)
            logits = logits[:, -1]
            logits = logits / temperatue
            probs = F.softmax(logits, dim=-1).cpu()
            decoder_word_input = t.multinomial(probs, 1,)

            result_ids.append(decoder_word_input[:, 0])

            if use_cuda:
                decoder_word_input, decoder_character_input = decoder_word_input.cuda(), decoder_character_input.cuda()

    result_ids = t.stack(result_ids, dim=1)
    result_ids = result_ids
    return result_ids

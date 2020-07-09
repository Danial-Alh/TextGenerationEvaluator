import os
from multiprocessing.pool import Pool
from time import time

import numpy as np
import tensorflow as tf
from nltk.translate.bleu_score import ngrams
from tqdm import tqdm

from ..utils.text_process import get_tokenlized


def generate_samples(sess, trainable_model, batch_size, generated_num, output_file=None, get_code=True,
                     temperature=None):
    # Generate Samples
    generated_samples = []
    if temperature['value'] is None:
        gen_func = trainable_model.generate
    elif temperature['type'] == 'biased':
        gen_func =lambda s: trainable_model.temperature_generate(s, temperature['value'])
    elif temperature['type'] == 'unbiased':
        gen_func = lambda s: trainable_model.unbiased_temperature_generate(s, temperature['value'])
    else:
        raise BaseException('invalid temperature type!')
    for _ in tqdm(range(int(generated_num / batch_size) + 1), ncols=80):
        sample = gen_func(sess)
        generated_samples.extend(sample)
    generated_samples = generated_samples[:generated_num]
    codes = list()
    if output_file is not None:
        with open(output_file, 'w') as fout:
            for poem in generated_samples:
                buffer = ' '.join([str(x) for x in poem]) + '\n'
                fout.write(buffer)
                if get_code:
                    codes.append(poem)
        return np.array(codes)
    codes = ""
    for poem in generated_samples:
        buffer = ' '.join([str(x) for x in poem]) + '\n'
        codes += buffer
    return codes


def init_sess():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    return sess


def pre_train_epoch(sess, trainable_model, data_loader):
    # Pre-train the generator using MLE for one epoch
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        batch = data_loader.next_batch()
        _, g_loss = trainable_model.pretrain_step(sess, batch)
        trainable_model.save_model(sess)
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)


def save_obj(obj, folder_path, file_name):
    import json
    file_path = folder_path + file_name + '.json'
    with open(file_path, 'w') as file:
        json.dump(obj, file, indent=True)
        print('{} saved in {}'.format(file_name, file_path))


def load_obj(folder_path, file_name):
    import os, json
    file_path = folder_path + file_name + '.json'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            print('{} loaded from {}'.format(file_name, file_path))
            return json.load(file)
    print('{} not founded in {}'.format(file_name, file_path))
    return None


def evaluate_bleu(reference_file, test_files: list, n_gram=3):
    from ..utils.metrics.CustomizedBLEU import Bleu as CBleu

    start = time()

    print('evaluating bleu {}!'.format(str(n_gram)))
    scores = {}
    weights = np.ones(n_gram) / n_gram
    references = get_tokenlized(reference_file)
    metric = CBleu(references, weights)
    for test_file in test_files:
        model_name = test_file.split('/')[-2]
        print('model: {}'.format(model_name))
        tests = get_tokenlized(test_file)
        score1 = metric.get_score(tests)
        score = np.mean(score1)
        print(score)
        print('metric evaluated!')
        scores[model_name] = score
    end = time()
    print('time spent: {}'.format(end - start))

    # from .. utils.metrics.Bleu import Bleu
    #
    # start = time()
    #
    # scores = {}
    # print('evaluating bleu!')
    # for test_file in test_files:
    #     print(test_file)
    #     metric = Bleu(test_file, reference_file, n_gram)
    #     score2 = metric.get_score(False)
    #     score = np.mean(score2)
    #     print(score)
    #     print('metric evaluated!')
    #     scores[test_file] = score
    #
    # end = time()
    # print('time spent: {}'.format(end - start))
    #
    # for i, s1 in enumerate(score1):
    #     if s1 != score2[i]:
    #         input("fault!")
    # print('they areeeeeeeeeeeeeeeeeeeee the saaaaaaaaaaaaaame!')

    return scores


def evaluate_selfbleu(test_files: list, n_gram=3):
    from ..utils.metrics.CustomizedSelfBleu import SelfBleu as CSelfBleu

    start = time()

    print('evaluating self-bleu {}!'.format(str(n_gram)))
    scores = {}
    weights = np.ones(n_gram) / n_gram
    for test_file in test_files:
        model_name = test_file.split('/')[-2]
        print('model: {}'.format(model_name))
        tests = get_tokenlized(test_file)
        metric = CSelfBleu(tests, weights)
        score1 = metric.get_score()
        score = np.mean(score1)
        print(score)
        print('metric evaluated!')
        scores[model_name] = score

    end = time()
    print('time spent: {}'.format(end - start))

    # from .. utils.metrics.SelfBleu import SelfBleu
    #
    # start = time()
    #
    # scores = {}
    # print('evaluating self-bleu!')
    # for test_file in test_files:
    #     print(test_file)
    #     metric = SelfBleu(test_file, n_gram)
    #     score2 = metric.get_score(False)
    #     score = np.mean(score2)
    #     print(score)
    #     print('metric evaluated!')
    #     scores[test_file] = score
    #
    # end = time()
    # print('time spent: {}'.format(end - start))

    # for i, s1 in enumerate(score1):
    #     if s1 != score2[i]:
    #         input("fault!")
    #         print(i)
    #         print(s1)
    #         print(score2[i])
    # print('they areeeeeeeeeeeeeeeeeeeee the saaaaaaaaaaaaaame!')
    return scores


def evaluate_msJaccard(reference_file, test_files: list, n_gram=3):
    from ..utils.metrics.msjaccard import MSJaccard
    print('evaluating msjaccard {}!'.format(str(n_gram)))
    scores = {}
    references = get_tokenlized(reference_file)
    metric = MSJaccard(references, n_gram)
    for test_file in test_files:
        model_name = test_file.split('/')[-2]
        print('model: {}'.format(model_name))
        tests = get_tokenlized(test_file)
        score = metric.get_score(tests)
        print(score)
        print('metric evaluated!')
        scores[model_name] = score
    return scores

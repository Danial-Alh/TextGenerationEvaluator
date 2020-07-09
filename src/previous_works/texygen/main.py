import getopt
import sys

from colorama import Fore

from .models.gsgan.Gsgan import Gsgan
from .models.leakgan.Leakgan import Leakgan
from .models.maligan_basic.Maligan import Maligan
from .models.mle.Mle import Mle
from .models.rankgan.Rankgan import Rankgan
from .models.seqgan.Seqgan import Seqgan
from .models.textGan_MMD.Textgan import TextganMmd


def set_gan(gan_name):
    gans = dict()
    gans['seqgan'] = Seqgan
    gans['gsgan'] = Gsgan
    gans['textgan'] = TextganMmd
    gans['leakgan'] = Leakgan
    gans['rankgan'] = Rankgan
    gans['maligan'] = Maligan
    gans['mle'] = Mle
    try:
        Gan = gans[gan_name.lower()]
        gan = Gan()
        gan.vocab_size = 5000
        gan.generate_num = 10000
        return gan
    except KeyError:
        print(Fore.RED + 'Unsupported GAN type: ' + gan_name + Fore.RESET)
        sys.exit(-2)


def set_training(gan, training_method):
    try:
        if training_method == 'oracle':
            gan_func = gan.train_oracle
        elif training_method == 'cfg':
            gan_func = gan.train_cfg
        elif training_method == 'real':
            gan_func = gan.train_real
        else:
            print(Fore.RED + 'Unsupported training setting: ' + training_method + Fore.RESET)
            sys.exit(-3)
    except AttributeError:
        print(Fore.RED + 'Unsupported training setting: ' + training_method + Fore.RESET)
        sys.exit(-3)
    return gan_func


def evaluate(data_loc=None, model_names=['seqgan'], max_n=3, n_samples=2000):
    dataset_name = data_loc.split('/')[-1].replace('.txt', '')

    def dump_to_file(obj):
        import json
        with open('evaluations_{}.txt'.format(dataset_name), 'w') as file:
            json.dump(obj, file, indent='\t')

    import tensorflow as tf
    valid_loc = data_loc.replace('-train', '-valid')
    test_files = []
    for gan_name in model_names:
        gan = set_gan(gan_name)
        gan.generate_sample2(data_loc, n_samples)
        gan.sess.close()
        tf.reset_default_graph()
        test_files.append(gan.test_file)
    # test_files = [data_loc]
    from .utils.utils import evaluate_bleu, evaluate_msJaccard, evaluate_selfbleu
    result = {}
    for n in range(2, max_n + 1):
        result['blue' + str(n)] = evaluate_bleu(valid_loc, test_files, n)
        dump_to_file(result)
        result['self-blue' + str(n)] = evaluate_selfbleu(test_files, n)
        dump_to_file(result)
        result['msJaccard' + str(n)] = evaluate_msJaccard(valid_loc, test_files, n)
        dump_to_file(result)
    sys.exit(1)


def parse_cmd(argv):
    try:
        opts, args = getopt.getopt(argv, "hg:t:d:e:n:m:")

        opt_arg = dict(opts)
        if '-h' in opt_arg.keys():
            print('usage: python main.py -g <gan_type>')
            print('       python main.py -g <gan_type> -t <train_type>')
            print('       python main.py -g <gan_type> -t realdata -d <your_data_location>')
            sys.exit(0)
        if '-e' in opt_arg.keys():
            max_ngrams = 3
            n_samples = 2000
            if '-m' in opt_arg.keys():
                max_ngrams = int(opt_arg['-m'])
            if '-n' in opt_arg.keys():
                n_samples = int(opt_arg['-n'])
            evaluate(opt_arg['-d'], opt_arg['-e'].split(','), max_ngrams, n_samples)
        if not '-g' in opt_arg.keys():
            print('unspecified GAN type, use MLE training only...')
            gan = set_gan('mle')
        else:
            gan = set_gan(opt_arg['-g'])
        if not '-t' in opt_arg.keys():
            gan.train_oracle()
        else:
            gan_func = set_training(gan, opt_arg['-t'])
            if opt_arg['-t'] == 'real' and '-d' in opt_arg.keys():
                gan_func(opt_arg['-d'])
            else:
                gan_func()
    except getopt.GetoptError:
        print('invalid arguments!')
        print('`python main.py -h`  for help')
        sys.exit(-1)
    pass


if __name__ == '__main__':
    gan = None
    parse_cmd(sys.argv[1:])

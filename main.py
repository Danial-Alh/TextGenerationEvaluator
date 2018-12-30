import argparse

from data_management.data_loaders import SentenceDataloader
from data_management.data_manager import OracleDataManager, SentenceDataManager
from the_new_evaluator import RealWorldEvaluator, BestModelTracker, OracleEvaluator

parser = argparse.ArgumentParser()

parser.add_argument('mode', type=str, help='real(world)/oracle mode', choices=['real', 'oracle'])
parser.add_argument('-d', '--data', type=str, help='dataset name', required=True)
parser.add_argument('-a', '--action', type=str, help='train/gen(erate)/eval(uate)', choices=['train', 'gen', 'eval'],
                    required=True)
parser.add_argument('-k', type=int, help='which fold to action be done', required=True)
parser.add_argument('-m', '--models', type=str, help='model names', nargs='+', choices=['mle', 'textgan', 'rankgan',
                                                                                        'leakgan', 'maligan', 'seqgan',
                                                                                        'dgsan'])
parser.add_argument('-r', '--restore', type=str, help='restore types', nargs='+')
args = parser.parse_args()

k_fold = 3
dataset_prefix_name = args.data.split('-')[0]

if args.mode == 'real':
    dm = SentenceDataManager([SentenceDataloader(args.data)], dataset_prefix_name + '-words', k_fold=k_fold)
    ev = RealWorldEvaluator(dm, args.action, args.k, dataset_prefix_name)
elif args.mode == 'oracle':
    dm = OracleDataManager([SentenceDataloader(args.data)], dataset_prefix_name + '-words', k_fold=k_fold)
    ev = OracleEvaluator(dm, args.action, args.k, dataset_prefix_name)

print(args.models)

if args.action == 'train':
    if args.models is None:
        raise BaseException('specify the model to be trained!')
    for model_name in args.models:
        tracker = BestModelTracker(model_name, ev)
        tracker.start()
        tracker.model.delete()
elif args.action == 'gen':
    ev.generate_samples(args.models, args.restore)
elif args.action == 'eval':
    ev.final_evaluate(args.models, args.restore)

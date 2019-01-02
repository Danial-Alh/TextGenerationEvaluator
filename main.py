import argparse

from data_management.data_loaders import SentenceDataloader
from data_management.data_manager import OracleDataManager, SentenceDataManager
from data_management.parsers import WordBasedParser, OracleBasedParser
from the_new_evaluator import RealWorldEvaluator, BestModelTracker, OracleEvaluator


def do(ParserClass, DataManagerClass, EvaluatorClass):
    parser = ParserClass(name=dataset_prefix_name + '-words')
    test_data_loader = SentenceDataloader(dataset_prefix_name + '-test')
    print(args.models)

    if args.action == 'train':
        train_data_loader = SentenceDataloader(dataset_prefix_name + '-train')
        dm = DataManagerClass(train_data_loader, test_data_loader, parser=parser, k_fold=k_fold)
        tr, va = dm.get_data(args.k).values()
        ev = EvaluatorClass(tr, va, None, parser, args.action, args.k, dataset_prefix_name)

        if args.models is None:
            raise BaseException('specify the model to be trained!')
        for model_name in args.models:
            tracker = BestModelTracker(model_name, ev)
            tracker.start()
            tracker.model.delete()
    elif args.action == 'gen':
        ts = parser.line2id_format(test_data_loader.get_data())
        ev = EvaluatorClass(None, None, ts, parser, args.action, args.k, dataset_prefix_name)

        ev.generate_samples(args.models, args.restore)
    elif args.action == 'eval':
        ts = test_data_loader.get_data()[:1000]
        ev = EvaluatorClass(None, None, ts, None, args.action, args.k, dataset_prefix_name)

        ev.final_evaluate(args.models, args.restore)


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
    do(WordBasedParser, SentenceDataManager, RealWorldEvaluator)
else:
    do(OracleBasedParser, OracleDataManager, OracleEvaluator)

import argparse

from data_management.data_loaders import SentenceDataloader
from data_management.data_manager import OracleDataManager, SentenceDataManager
from data_management.parsers import WordBasedParser, OracleBasedParser
from evaluator import RealWorldEvaluator, OracleEvaluator, BestModelTracker, Dumper
from models import all_models, create_model

parser = argparse.ArgumentParser()


def convert_legacies():
    ev_name = args.mode
    m = create_model('dgsan', None)
    dmp = Dumper(m, args.k, dataset_prefix_name)
    ts_lines = [r['text'] for r in dmp.load_samples_with_additional_fields('bleu3', 'test')]
    for m_name in all_models:
        if m_name == 'dgsan':
            continue
        for restore in (RealWorldEvaluator.test_restore_types if ev_name == 'real' \
                else OracleEvaluator.test_restore_types):
            print(m_name)
            print(restore)
            for k in args.k:
                print('********************* Legacy convert K{} *********************'.format(k))
                m = create_model(m_name, None)
                dmp = Dumper(m, args.k, dataset_prefix_name)
                sample_lines = dmp.load_generated_samples(restore)
                dmp.dump_samples_with_additional_fields(sample_lines, {'lnq': [1.0 for _ in sample_lines],
                                                                       'lnp': [1.0 for _ in sample_lines]},
                                                        restore, 'gen')
                dmp.dump_samples_with_additional_fields(ts_lines, {'lnq': [1.0 for _ in ts_lines],
                                                                   'lnp': [1.0 for _ in ts_lines]},
                                                        restore, 'test')


parser.add_argument('mode', type=str, help='real(world)/oracle mode', choices=['real', 'oracle'])
parser.add_argument('-d', '--data', type=str, help='dataset name', required=True)
parser.add_argument('-a', '--action', type=str, help='train/gen(erate)/eval(uate)', choices=['train', 'gen', 'eval',
                                                                                             'eval_precheck', 'legacy',
                                                                                             'export', 'dump'],
                    required=True)
parser.add_argument('-k', type=int, help='which fold to action be done on', nargs='+')
parser.add_argument('--temper_mode', type=str, help='biased/unbiased temperature mode', choices=['unbiased', 'biased'],
                    default='biased')
parser.add_argument('-t', '--temperatures', type=float, help='softmax temperatures', nargs='+')
parser.add_argument('-m', '--models', type=str, help='model names', nargs='+', choices=all_models)
parser.add_argument('-r', '--restore', type=str, help='restore types', nargs='+')
args = parser.parse_args()

if args.restore is not None:
    args.restore = [r if not r.startswith('nll') else ('-' + r) for r in args.restore]
    model_restore_zip = dict(zip(args.models, args.restore))
    print(model_restore_zip)
if args.temperatures is None:
    args.temperatures = [None]
args.temperatures = [{'type': args.temper_mode, 'value': v} for v in args.temperatures]
print('temperatures: {}, K: {}'.format(args.temperatures, args.k))

dataset_prefix_name = args.data.split('-')[0]
from evaluator import update_config

update_config(dataset_prefix_name)
from evaluator import k_fold

print("********************* k fold is '%d' *********************" % k_fold)

if args.mode == 'real':
    ParserClass, DataManagerClass, EvaluatorClass = WordBasedParser, SentenceDataManager, RealWorldEvaluator
elif args.mode == 'oracle':
    ParserClass, DataManagerClass, EvaluatorClass = OracleBasedParser, OracleDataManager, OracleEvaluator

print(args.models)

if args.k is not None:
    from evaluator import k_fold

    assert not (True in [k >= k_fold for k in args.k])

if args.action == 'train':
    assert args.k is not None
    assert len(args.temperatures) == 0 and args.temperatures[0]['value'] is None
    parser = ParserClass(name=dataset_prefix_name + '-words')
    test_data_loader = SentenceDataloader(dataset_prefix_name + '-test')
    train_data_loader = SentenceDataloader(dataset_prefix_name + '-train')
    dm = DataManagerClass(train_data_loader, test_data_loader, parser=parser, k_fold=k_fold)
    for k in args.k:
        print('********************* training K{} *********************'.format(k))
        tr, va = dm.get_data(k)
        print(len(tr), len(va))
        ev = EvaluatorClass(tr, va, None, parser=parser, mode=args.action, k=k, temperature=args.temperatures[0],
                            dm_name=dataset_prefix_name)

        if args.models is None:
            raise BaseException('specify the model to be trained!')
        for model_name in args.models:
            tracker = BestModelTracker(model_name, ev)
            tracker.start()
            tracker.model.delete()
elif args.action == 'gen':
    assert args.k is not None
    parser = ParserClass(name=dataset_prefix_name + '-words')
    test_data_loader = SentenceDataloader(dataset_prefix_name + '-test')
    ts = parser.line2id_format(test_data_loader.get_data())
    for temperature in args.temperatures:
        for k in args.k:
            print('********************* sample generation K{}, temperature: {} *********************'.
                  format(k, temperature))
            ev = EvaluatorClass(None, None, ts, parser=parser, mode=args.action, k=k, temperature=temperature,
                                dm_name=dataset_prefix_name)
            # ev.generate_samples(args.models, args.restore)
            ev.generate_samples(model_restore_zip)
elif args.action.startswith('eval'):
    assert args.k is not None
    m_name = args.models[0] if args.models is not None else all_models[0]
    restore_type = args.restore[0] if args.restore is not None else \
        (RealWorldEvaluator.test_restore_types[0] if args.mode == 'real' else OracleEvaluator.test_restore_types[0])
    parser = ParserClass(name=dataset_prefix_name + '-words')
    test_data_loader = SentenceDataloader(dataset_prefix_name + '-test')
    # ts, _ = parser.line2id_format(test_data_loader.get_data())
    ts = test_data_loader.get_data()
    for k in args.k:
        # m = create_model(m_name, None)
        ev = EvaluatorClass(None, None, ts, parser=None, mode=args.action, k=k, temperature=None,
                            dm_name=dataset_prefix_name)
        # dmp = Dumper(m, k, dataset_prefix_name)
        for temperature in args.temperatures:
            print('********************* evaluating K{}, temperature: {} *********************'.
                  format(k, temperature))
            ev.temperature = temperature
            # ts = [r['text'] for r in dmp.load_samples_with_additional_fields(restore_type, 'test')]

            # write_text([r['text'] for r in dmp.load_samples_with_additional_fields(args.restore[0], 'test')], 't')
            # write_text([r['text'] for r in dmp.load_samples_with_additional_fields(args.restore[0], 'gen')], 'g')
            # exit(0)

            # ts = test_data_loader.get_data()[:1000]
            # from utils.file_handler import read_text
            # ts = read_text('{}-valid-k{}_parsed'.format(dataset_prefix_name, k), False)
            # EvaluatorClass(None, None, ts, None, 'eval_precheck', k, dataset_prefix_name).eval_pre_check(model_restore_zip)
            if args.action == 'eval':
                # ev.final_evaluate(args.models, args.restore)
                ev.final_evaluate(model_restore_zip)
elif args.action == 'legacy':
    assert args.k is not None
    convert_legacies()
elif args.action == 'export':
    from export_utils.evaluation_exporter import export_tables
    for temperature in args.temperatures:
        export_tables(args.mode, dataset_prefix_name, model_restore_zip, temperature)
        # for k in args.k:
        # from export_utils.histogram_exporter import export_histogram
        #     export_histogram(args.mode, dataset_prefix_name, model_restore_zip, k)
elif args.action == 'dump':
    assert args.k is not None
    parser = ParserClass(name=dataset_prefix_name + '-words')
    test_data_loader = SentenceDataloader(dataset_prefix_name + '-test')
    train_data_loader = SentenceDataloader(dataset_prefix_name + '-train')
    dm = DataManagerClass(train_data_loader, test_data_loader, parser=parser, k_fold=k_fold)
    for k in args.k:
        print('********************* dumping K{} *********************'.format(k))
        dm.dump_data_on_file(k, True, dataset_prefix_name)

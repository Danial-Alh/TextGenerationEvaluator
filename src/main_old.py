from evaluators.base_evaluator import k_fold
from evaluators.base_evaluator import update_config
import argparse

from evaluators import BestModelTracker, ModelDumpManager
from previous_works import all_models, create_model

parser = argparse.ArgumentParser()


def convert_legacies():
    ev_name = args.mode
    m = create_model('dgsan', None)
    dmp = ModelDumpManager(m, args.run, dataset_prefix_name)
    ts_lines = [r['text'] for r in dmp.load_samples_with_additional_fields('bleu3', 'test')]
    for m_name in all_models:
        if m_name == 'dgsan':
            continue
        for restore in (RealWorldEvaluator.test_restore_types if ev_name == 'real'
                        else OracleEvaluator.test_restore_types):
            print(m_name)
            print(restore)
            for run in args.run:
                print('********************* Legacy convert run{} *********************'.format(run))
                m = create_model(m_name, None)
                dmp = ModelDumpManager(m, args.run, dataset_prefix_name)
                sample_lines = dmp.load_generated_samples(restore)
                dmp.dump_samples_with_additional_fields(sample_lines, {'nllq': [1.0 for _ in sample_lines],
                                                                       'nllp': [1.0 for _ in sample_lines]},
                                                        restore, 'gen')
                dmp.dump_samples_with_additional_fields(ts_lines, {'nllq': [1.0 for _ in ts_lines],
                                                                   'nllp': [1.0 for _ in ts_lines]},
                                                        restore, 'test')


parser.add_argument('mode', type=str, help='real(world)/oracle mode', choices=['real', 'oracle'])
parser.add_argument('-d', '--data', type=str, help='dataset name', required=True)
parser.add_argument('-a', '--action', type=str, help='train/gen(erate)/eval(uate)', choices=['train', 'gen', 'eval',
                                                                                             'eval_precheck', 'legacy',
                                                                                             'export', 'dump'],
                    required=True)
parser.add_argument('-run', type=int, help='which fold to action be done on', nargs='+')
parser.add_argument('--temper_mode', type=str, help='biased/unbiased temperature mode', choices=['unbiased', 'biased'],
                    default='biased')
parser.add_argument('-t', '--temperatures', type=float, help='softmax temperatures', nargs='+')
parser.add_argument('-m', '--models', type=str, help='model names', nargs='+', choices=all_models)
parser.add_argument('-r', '--restore', type=str, help='restore types', nargs='+')
args = parser.parse_args()

if args.restore is not None:
    args.restore = [r if not r.startswith('nll') else ('-' + r) for r in args.restore]
    model_run_restore_zip = dict(zip(args.models, args.restore))
    print(model_run_restore_zip)
if args.temperatures is None:
    args.temperatures = [None]
args.temperatures = [{'type': args.temper_mode, 'value': v} for v in args.temperatures]
print('temperatures: {}, run: {}'.format(args.temperatures, args.run))

dataset_prefix_name = args.data.split('-')[0]

update_config(dataset_prefix_name)

print("********************* run fold is '%d' *********************" % k_fold)

if args.mode == 'real':
    ParserClass, DataManagerClass, EvaluatorClass = WordBasedParser, SentenceDataManager, RealWorldEvaluator
elif args.mode == 'oracle':
    ParserClass, DataManagerClass, EvaluatorClass = OracleBasedParser, OracleDataManager, OracleEvaluator

print(args.models)

if args.run is not None:
    from evaluator import k_fold

    assert not (True in [run >= k_fold for run in args.run])

if args.action == 'train':
    assert args.run is not None
    assert len(args.temperatures) == 0 and args.temperatures[0]['value'] is None
    parser = ParserClass(name=dataset_prefix_name + '-words')
    test_data_loader = SentenceDataloader(dataset_prefix_name + '-test')
    train_data_loader = SentenceDataloader(dataset_prefix_name + '-train')
    dm = DataManagerClass(train_data_loader, test_data_loader, parser=parser, k_fold=k_fold)
    for run in args.run:
        print('********************* training run{} *********************'.format(run))
        tr, va = dm.get_data(run)
        print(len(tr), len(va))
        ev = EvaluatorClass(tr, va, None, parser=parser, mode=args.action, run=run, temperature=args.temperatures[0],
                            dm_name=dataset_prefix_name)

        if args.models is None:
            raise BaseException('specify the model to be trained!')
        for model_name in args.models:
            tracker = BestModelTracker(model_name, ev)
            tracker.start()
            tracker.model.reset_model()
elif args.action == 'gen':
    assert args.run is not None
    parser = ParserClass(name=dataset_prefix_name + '-words')
    test_data_loader = SentenceDataloader(dataset_prefix_name + '-test')
    ts = parser.line2id_format(test_data_loader.get_data())
    for temperature in args.temperatures:
        for run in args.run:
            print('********************* sample generation run{}, temperature: {} *********************'.
                  format(run, temperature))
            ev = EvaluatorClass(None, None, ts, parser=parser, mode=args.action, run=run, temperature=temperature,
                                dm_name=dataset_prefix_name)
            # ev.generate_samples(args.models, args.restore)
            ev.generate_samples(model_run_restore_zip)
elif args.action.startswith('eval'):
    assert args.run is not None
    m_name = args.models[0] if args.models is not None else all_models[0]
    restore_type = args.restore[0] if args.restore is not None else \
        (RealWorldEvaluator.test_restore_types[0] if args.mode ==
         'real' else OracleEvaluator.test_restore_types[0])
    parser = ParserClass(name=dataset_prefix_name + '-words')
    test_data_loader = SentenceDataloader(dataset_prefix_name + '-test')
    # ts, _ = parser.line2id_format(test_data_loader.get_data())
    ts = test_data_loader.get_data()
    for run in args.run:
        # m = create_model(m_name, None)
        ev = EvaluatorClass(None, None, ts, parser=None, mode=args.action, run=run, temperature=None,
                            dm_name=dataset_prefix_name)
        # dmp = ModelDumper(m, run, dataset_prefix_name)
        for temperature in args.temperatures:
            print('********************* evaluating run{}, temperature: {} *********************'.
                  format(run, temperature))
            ev.temperature = temperature
            # ts = [r['text'] for r in dmp.load_samples_with_additional_fields(restore_type, 'test')]

            # write_text([r['text'] for r in dmp.load_samples_with_additional_fields(args.restore[0], 'test')], 't')
            # write_text([r['text'] for r in dmp.load_samples_with_additional_fields(args.restore[0], 'gen')], 'g')
            # exit(0)

            # ts = test_data_loader.get_data()[:1000]
            # from utils.file_handler import read_text
            # ts = read_text('{}-valid-run{}_parsed'.format(dataset_prefix_name, run), False)
            # EvaluatorClass(None, None, ts, None, 'eval_precheck', run, dataset_prefix_name).eval_pre_check(model_run_restore_zip)
            if args.action == 'eval':
                # ev.final_evaluate(args.models, args.restore)
                ev.final_evaluate(model_run_restore_zip)
elif args.action == 'legacy':
    assert args.run is not None
    convert_legacies()
elif args.action == 'export':
    from export_utils.evaluation_exporter import export_tables
    for temperature in args.temperatures:
        export_tables(args.mode, dataset_prefix_name, model_run_restore_zip, temperature)
        # for run in args.run:
        # from export_utils.histogram_exporter import export_histogram
        #     export_histogram(args.mode, dataset_prefix_name, model_run_restore_zip, run)
elif args.action == 'dump':
    assert args.run is not None
    parser = ParserClass(name=dataset_prefix_name + '-words')
    test_data_loader = SentenceDataloader(dataset_prefix_name + '-test')
    train_data_loader = SentenceDataloader(dataset_prefix_name + '-train')
    dm = DataManagerClass(train_data_loader, test_data_loader, parser=parser, k_fold=k_fold)
    for run in args.run:
        print('********************* dumping run{} *********************'.format(run))
        dm.dump_data_on_file(run, True, dataset_prefix_name)

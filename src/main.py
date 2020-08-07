import argparse
from types import SimpleNamespace

from data_management.data_manager import load_oracle_dataset, load_real_dataset
from evaluators import BestModelTracker
from evaluators.base_evaluator import Evaluator
from previous_works.model_wrappers import all_model_names

arg_parser = argparse.ArgumentParser()

arg_parser.add_argument('-m', '--mode', type=str, help='real(world)/oracle mode',
                        choices=['real', 'oracle'])
arg_parser.add_argument('-d', '--data', type=str, help='dataset name', required=True)
arg_parser.add_argument('-a', '--action', type=str, help='train/gen(erate)/eval(uate)',
                        choices=['train', 'gen', 'eval', 'export'], required=True)
arg_parser.add_argument('-R', '--runs', type=int, help='The run number', nargs='+')
arg_parser.add_argument('--temper-mode', type=str, help='biased/unbiased temperature mode',
                        choices=['unbiased', 'biased'], default='biased')
arg_parser.add_argument('--train-temperatures', type=float,
                        help='train softmax temperatures', nargs='+')
arg_parser.add_argument('--test-temperatures', type=float,
                        help='test softmax temperatures', nargs='+')
arg_parser.add_argument('-M', '--model-names', type=str, help='model names',
                        nargs='+', choices=all_model_names)
arg_parser.add_argument('-r', '--restore-types', type=str, help='restore types', nargs='+')
args = arg_parser.parse_args()

assert not (args.runs is None and len(args.runs) == 0)

if args.train_temperatures is None or len(args.train_temperatures) == 0:
    args.train_temperatures = [None]
if args.test_temperatures is None or len(args.test_temperatures) == 0:
    args.test_temperatures = args.train_temperatures
if args.restore_types is None or len(args.restore_types) == 0:
    args.restore_types = ['undefined']

args.train_temperatures = [{'type': args.temper_mode, 'value': v} for v in args.train_temperatures]
args.test_temperatures = [{'type': args.temper_mode, 'value': v} for v in args.test_temperatures]

if len(args.runs) == 1:
    args.runs = args.runs * len(args.model_names)
if len(args.train_temperatures) == 1:
    args.train_temperatures = args.train_temperatures * len(args.model_names)
if len(args.test_temperatures) == 1:
    args.test_temperatures = args.test_temperatures * len(args.model_names)
if len(args.restore_types) == 1:
    args.restore_types = args.restore_types * len(args.model_names)

assert\
    len(args.model_names) ==\
    len(args.runs) ==\
    len(args.train_temperatures) ==\
    len(args.test_temperatures) ==\
    len(args.restore_types)

model_identifier_dicts = [
    SimpleNamespace(
        model_name=args.model_names[i],
        run=args.runs[i],
        train_temperature=args.train_temperatures[i],
        test_temperature=args.test_temperatures[i],
        restore_type=args.restore_types[i],
    )
    for i in range(len(args.model_names))
]


Evaluator.update_config(args.data)

if args.mode == 'real':
    from evaluators.real_evaluator import RealWorldEvaluator
    EvaluatorClass = RealWorldEvaluator
    trn, vld, tst, TEXT = load_real_dataset(args.data)
elif args.mode == 'oracle':
    from evaluators.oracle_evaluator import OracleEvaluator
    EvaluatorClass = OracleEvaluator,
    trn, vld, tst, TEXT = load_oracle_dataset()


if args.runs is not None:
    assert not (True in [run >= Evaluator.TOTAL_RUNS for run in args.runs])


if args.action == 'train':
    del tst
    for model_identifier in model_identifier_dicts:
        print('********************* training {} *********************'.format(model_identifier))
        print(len(trn), len(vld))
        ev = EvaluatorClass(trn, vld, None, parser=TEXT, mode=args.action, dm_name=args.data)

        if args.model_names is None:
            raise BaseException('specify the model to be trained!')
        for model_name in args.model_names:
            tracker = BestModelTracker(model_identifier, ev)
            tracker.start()
            tracker.model.reset_model()


elif args.action == 'gen':
    for model_identifier in model_identifier_dicts:
        print('********************* sample generation {} *********************'.format(model_identifier))
        ev = EvaluatorClass(trn, vld, tst, parser=TEXT, mode=args.action, dm_name=args.data)
        ev.generate_samples(model_identifier)


elif args.action == 'eval':
    del trn, vld
    for model_identifier in model_identifier_dicts:
        print('********************* evaluating {} *********************'.format(model_identifier))
        ev = EvaluatorClass(None, None, tst, parser=TEXT,
                            mode=args.action, dm_name=args.data)
        ev.final_evaluate(model_identifier)

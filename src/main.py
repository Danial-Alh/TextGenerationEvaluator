import argparse

from data_management.data_manager import load_oracle_dataset, load_real_dataset
from evaluators import BestModelTracker
from evaluators.base_evaluator import Evaluator
from previous_works import all_models

arg_parser = argparse.ArgumentParser()

arg_parser.add_argument('-m', '--mode', type=str, help='real(world)/oracle mode',
                    choices=['real', 'oracle'])
arg_parser.add_argument('-d', '--data', type=str, help='dataset name', required=True)
arg_parser.add_argument('-a', '--action', type=str, help='train/gen(erate)/eval(uate)',
                    choices=['train', 'gen', 'eval', 'export'], required=True)
arg_parser.add_argument('-R', '--runs', type=int, help='The run number', nargs='+')
arg_parser.add_argument('--temper_mode', type=str, help='biased/unbiased temperature mode',
                    choices=['unbiased', 'biased'], default='biased')
arg_parser.add_argument('-t', '--temperatures', type=float, help='softmax temperatures', nargs='+')
arg_parser.add_argument('-M', '--models', type=str, help='model names', nargs='+', choices=all_models)
arg_parser.add_argument('-r', '--restores', type=str, help='restore types', nargs='+')
args = arg_parser.parse_args()

assert args.runs is not None

if args.restores is not None:
    model_run_restore_zip = list(zip(args.models, args.runs, args.restores))
if args.temperatures is None:
    args.temperatures = [None]

args.temperatures = [{'type': args.temper_mode, 'value': v} for v in args.temperatures]
print('temperatures: {}, run: {}'.format(args.temperatures, args.runs))


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
    assert args.temperatures[0]['value'] is None
    del tst
    for run in args.runs:
        print('********************* training run {} *********************'.format(run))
        print(len(trn), len(vld))
        ev = EvaluatorClass(trn, vld, None, parser=TEXT, mode=args.action,
                            temperature=args.temperatures[0], dm_name=args.data)

        if args.models is None:
            raise BaseException('specify the model to be trained!')
        for model_name in args.models:
            tracker = BestModelTracker(model_name, run, ev)
            tracker.start()
            tracker.model.reset_model()


elif args.action == 'gen':
    del trn, vld
    for temperature in args.temperatures:
        for model_name, run, restore_type in model_run_restore_zip:
            print('********************* sample generation run: {}, restore_type: {}, temperature: {} *********************'.
                  format(run, restore_type, temperature))
            ev = EvaluatorClass(None, None, tst, parser=TEXT, mode=args.action,
                                temperature=temperature, dm_name=args.data)
            ev.generate_samples(model_name, run, restore_type)


elif args.action == 'eval':
    del trn, vld
    for temperature in args.temperatures:
        for model_name, run, restore_type in model_run_restore_zip:
            ev = EvaluatorClass(None, None, tst, parser=TEXT, mode=args.action,
                                temperature=None, dm_name=args.data)

            print('********************* evaluating run{}, temperature: {} *********************'.
                  format(run, temperature))
            ev.temperature = temperature
            if args.action == 'eval':
                ev.final_evaluate(model_name, run, restore_type)

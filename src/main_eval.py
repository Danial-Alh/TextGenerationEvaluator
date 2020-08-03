from types import SimpleNamespace

from data_management.data_manager import load_oracle_dataset, load_real_dataset

from db_management.model_db_manager import ModelDBManager
from db_management.models import *
from db_management.predefined_queries import LEFT_JOIN_QUERY

from evaluators.real_evaluator import RealWorldEvaluator

result = TrainedModel.objects.aggregate(
    [
        {
            "$match":
            {
                # "dataset_name": {"$in": ["amazon_app_book"]},
                "dataset_name": {"$in": ["amazon_app_book", "coco", "yelp_restaurant"]},
                "model_name": {"$in": ["dgsan", "mle_ehsan"]},
                # "run": {"$in": [0]},
                # "train_temperature": {"$in": [""]},
            }
        },
        *LEFT_JOIN_QUERY,
        {
            "$match":
            {
                # "restore_type": {"$in": ["bleu3"]},
                # "test_temperature": {"$in": [""]},
                # "evaluated": False
            }
        },
        {"$sort": {"dataset_name": +1, "test_temperature": -1}}
    ]
)

result = list(result)

model_identifier_dicts = [SimpleNamespace(**record) for record in result]

print(len(model_identifier_dicts))

input("continue?")

EvaluatorClass = RealWorldEvaluator
tst = None
for model_identifier in model_identifier_dicts:
    if tst is None or ev.dm_name != model_identifier.dataset_name:
        _, _, tst, TEXT = load_real_dataset(model_identifier.dataset_name)
        ev = EvaluatorClass(None, None, tst, parser=TEXT,
                            mode="eval", dm_name=model_identifier.dataset_name)
    print('********************* evaluating {} *********************'.format(model_identifier))
    ev.final_evaluate(model_identifier)

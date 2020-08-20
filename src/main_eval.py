import datetime
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
                # "dataset_name": {"$in": ["amazon_app_book", "coco", "yelp_restaurant"]},
                # "dataset_name": {"$in": []},

                # "model_name": {"$in": ["maligan", "seqgan", "rankgan"]},
                # "model_name": {"$in": []},

                # "run": {"$in": [0]},
                # "train_temperature": {"$in": [""]},
            }
        },
        *LEFT_JOIN_QUERY,
        {
            "$match":
            {
                # "restore_type": {"$in": ["bleu3"]},
                # "test_temperature": {"$in": ["{:.10f}".format(1e-2)]},
                # "test_temperature": {"$in": [""]},
                "evaluated_model_created_at": {"$gte":
                                               datetime.datetime.strptime("2020-08-14 00:00:00", '%Y-%m-%d %H:%M:%S')},
                # "evaluated": False
            }
        },
        {"$sort": {"dataset_name": +1, "test_temperature": -1}}
    ]
)

result = list(result)
for r in result:
    del r["evaluated_model_created_at"]
    del r["evaluated_model_updated_at"]

model_identifier_dicts = [SimpleNamespace(**record) for record in result]

print(len(model_identifier_dicts))

for md in model_identifier_dicts:
    print(md)

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

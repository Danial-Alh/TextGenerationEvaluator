from types import SimpleNamespace

from db_management.model_db_manager import ModelDBManager
from db_management.models import *
from db_management.predefined_queries import LEFT_JOIN_QUERY

result = TrainedModel.objects.aggregate(
    [
        {
            "$match":
            {
                # "dataset_name": {"$in": ["coco"]},
                "model_name": {"$in": ["dgsan"]},
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
                # "evaluated": True
            }
        }
    ]
)

result = list(result)

model_identifier_dicts = [SimpleNamespace(**record) for record in result]

print(model_identifier_dicts)
print(len(model_identifier_dicts))

for m in model_identifier_dicts:
    manager = ModelDBManager("", m.dataset_name, m.model_name, m.run,
                             m.train_temperature, m.restore_type, m.test_temperature, verbose=False)

import db_management.setup
from db_management.models import Model, InTrainingEvaluationHistory


# record = Model.objects(
#     model_name="mle",
#     dataset_name="ptb",
#     run=0,
#     restore_type="last_iter",
#     temperature="",
# ).get()

record = InTrainingEvaluationHistory.objects(
    model_name="vae",
    dataset_name="ptb",
    run=0,
).get()

input('sure??')

record.delete()
print('deleted!')

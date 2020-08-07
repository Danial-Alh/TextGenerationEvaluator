from db_management.setup import *
from db_management.models import *

result = TrainedModel.objects(
    model_name="vae",
).delete()

print(result)

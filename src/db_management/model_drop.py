from db_management.setup import *
from db_management.models import *

result = TrainedModel.objects(
    model_name__in=["mle"],
    # dataset_name__in=["yelp_restaurant", "amazon_app_book"],
    # train_temperature="{:.10f}".format(1.)
)

print(result.count())

if input('continue?(y/N)') == 'y':
    result.delete()
    print('deleted')

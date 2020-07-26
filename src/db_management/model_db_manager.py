from functools import reduce

import mongoengine
import numpy as np
import pandas as pd
from pymongo import UpdateOne

from .models import (TrainedModel, EvaluatedModel, Sample,
                     MetricHistoryRecord, MetricResult)


class ModelDBManager:
    def __init__(self, computer_name, dataset_name, model_name, run, train_temperature=None,
                 restore_type=None, test_temperature=None):
        self.computer_name = computer_name
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.run = run
        self.train_temperature = self.get_stringified_temperature(train_temperature)

        self.restore_type = restore_type
        self.test_temperature = self.get_stringified_temperature(test_temperature)

    def fetch_db_trained_model(self):
        print('DB: fetching TrainedModel record.')
        trained_model = TrainedModel.objects(
            model_name=self.model_name,
            dataset_name=self.dataset_name,
            run=self.run,
            train_temperature=self.train_temperature,
        ).get()

        print('DB: done!')
        return trained_model

    def fetch_db_evaluated_model(self, trained_model=None):
        print('DB: fetching EvaluatedModel record.')

        if trained_model is None:
            trained_model = self.fetch_db_trained_model()

        evaluated_model = EvaluatedModel.objects(
            trained_model=trained_model,
            restore_type=self.restore_type,
            test_temperature=self.test_temperature,
        ).get()

        print('DB: done!')
        return evaluated_model

    def create_empty_db_trained_model(self):
        print('DB: creating TrainedModel record.')
        trained_model = TrainedModel(
            machine_name=self.computer_name, model_name=self.model_name,
            dataset_name=self.dataset_name, run=self.run, train_temperature=self.train_temperature,
            best_history={}, all_history={}
        )

        try:
            trained_model.save()
        except mongoengine.errors.NotUniqueError as e:
            if input('TrainedModel record found. delete it? (y/N)') == 'y':
                self.fetch_db_trained_model().delete()
                print("DB: old record deleted!")
                trained_model.save()
            else:
                print("duplicate keys not allowed!")
                raise e

        print('DB: done!')
        return trained_model

    def init_history(self, initial_scores=None):
        print('*' * 10 + ' DB SECTION ' + '*' * 10)
        self.create_empty_db_trained_model()

        if initial_scores is not None:
            self.append_to_history(initial_scores)
            self.append_to_best_history(initial_scores)
        print('DB: done!')
        print('*' * 10 + ' END OF DB SECTION ' + '*' * 10)

    def append_to_best_history(self, new_metrics):
        print('*' * 10 + ' DB SECTION ' + '*' * 10)
        print('DB: append to best history.')
        trained_model = self.fetch_db_trained_model()

        for metric, value in new_metrics.items():
            if metric not in trained_model.best_history:
                trained_model.best_history[metric] = []
            trained_model.best_history[metric].append(MetricHistoryRecord(**value))

        trained_model.save()
        print('DB: done!')
        print('*' * 10 + ' END OF DB SECTION ' + '*' * 10)

    def append_to_history(self, new_metrics):
        print('*' * 10 + ' DB SECTION ' + '*' * 10)
        print('DB: append to history.')
        trained_model = self.fetch_db_trained_model()

        for metric, value in new_metrics.items():
            if metric not in trained_model.all_history:
                trained_model.all_history[metric] = []
            trained_model.all_history[metric].append(MetricHistoryRecord(**value))

        trained_model.save()
        print('DB: done!')
        print('*' * 10 + ' END OF DB SECTION ' + '*' * 10)

    def dump_samples_with_persample_metrics(self, dumping_object: dict):
        def convert_dmpobj_to_db_sample(dmp_obj, origin):
            return [
                Sample(
                    evaluated_model=evaluated_model,
                    index=i,
                    origin=origin,
                    tokens=dmp_obj[origin]['tokens'][i],
                    sentence=dmp_obj[origin]['sentence'][i],
                    metrics={key: MetricResult(value=value[i])
                             for key, value in dmp_obj[origin].items() if key not in ['sentence', 'tokens']}
                )
                for i in range(len(dmp_obj[origin]['sentence']))
            ]

        print('*' * 10 + ' DB SECTION ' + '*' * 10)

        assert self.persample_metrics_exist_for_each_sample(dumping_object)

        trained_model = None
        try:
            trained_model = self.fetch_db_trained_model()
        except mongoengine.errors.DoesNotExist as e:
            print("DB: TrainedModel record not found! creating new one!")
            trained_model = self.create_empty_db_trained_model()

        print('DB: creating EvaluatedModel record.')

        evaluated_model = EvaluatedModel(
            trained_model=trained_model,
            restore_type=self.restore_type,
            test_temperature=self.test_temperature,
        )

        try:
            evaluated_model.save()
        except mongoengine.errors.NotUniqueError as e:
            if input('EvaluatedModel record found. delete it? (y/N)') == 'y':
                self.fetch_db_evaluated_model().delete()
                print("DB: old record deleted!")
                evaluated_model.save()
            else:
                print("duplicate keys not allowed!")
                raise e

        print('DB: done!')

        print('DB: creating { generated, test } samples record.')
        generated_samples = convert_dmpobj_to_db_sample(dumping_object, origin='generated')
        test_samples = convert_dmpobj_to_db_sample(dumping_object, origin='test')

        Sample.objects.insert(generated_samples)
        Sample.objects.insert(test_samples)

        print('DB: {} generated samples and {} test samples inserted into database! done!'
              .format(len(generated_samples), len(test_samples)))
        print('*' * 10 + ' END OF DB SECTION ' + '*' * 10)

    def update_persample_metrics_for_generated_samples(self, dumping_object: dict):

        print('*' * 10 + ' DB SECTION ' + '*' * 10)
        print('DB: updating generated samples metrics.')

        evaluated_model = self.fetch_db_evaluated_model()

        generated_samples = Sample.objects(
            evaluated_model=evaluated_model, origin='generated').order_by('+index')
        new_metrics = [{} for _ in range(len(generated_samples))]

        for metric in dumping_object:
            if isinstance(dumping_object[metric], dict):
                ids = dumping_object[metric]['ids']
                values = dumping_object[metric]['values']
            else:
                ids = np.arange(len(dumping_object[metric]))
                values = dumping_object[metric]
                assert len(dumping_object[metric]) == len(generated_samples)

            for i, v in zip(ids, values):
                new_metrics[i]['metrics.{}'.format(metric)] = {'value': v}

        update_operations = [
            UpdateOne(
                {'_id': generated_samples[i].id}, {'$set': nm}
            )
            for i, nm in enumerate(new_metrics) if len(nm) != 0
        ]

        Sample._get_collection().bulk_write(update_operations, ordered=False)

        print('DB: done!')
        print('*' * 10 + ' END OF DB SECTION ' + '*' * 10)

    def load_samples_with_persample_metrics(self):
        print('*' * 10 + ' DB SECTION ' + '*' * 10)
        print('DB: fetching { generated, test } samples.')

        trained_model = self.fetch_db_trained_model()
        evaluated_model = self.fetch_db_evaluated_model(trained_model)

        generated_samples = Sample.objects(evaluated_model=evaluated_model,
                                           origin='generated').order_by('+index')
        test_samples = Sample.objects(evaluated_model=evaluated_model,
                                      origin='test').order_by('+index')

        generated_samples = list(generated_samples)
        test_samples = list(test_samples)

        print('{} generated samples and {} test samples fetched! done!'.format(len(generated_samples),
                                                                               len(test_samples)))
        print('*' * 10 + ' END OF DB SECTION ' + '*' * 10)
        return {'trained_model': trained_model, 'evaluated_model': evaluated_model,
                'generated': generated_samples, 'test': test_samples}

    def dump_final_evaluation_results(self, dumping_object: dict):
        print('*' * 10 + ' DB SECTION ' + '*' * 10)
        print('DB: inserting evaluation results.')

        evaluated_model = self.fetch_db_evaluated_model()

        for metric, value in dumping_object.items():
            if isinstance(value, dict):
                evaluated_model.metrics[metric] = MetricResult(**value)
            else:
                evaluated_model.metrics[metric] = MetricResult(value=value)

        evaluated_model.save()

        print('DB: done!')
        print('*' * 10 + ' END OF DB SECTION ' + '*' * 10)

    def load_final_evaluation_results(self):
        print('*' * 10 + ' DB SECTION ' + '*' * 10)
        print('DB: fetcing EvaluatedModel record.')

        trained_model = self.fetch_db_trained_model()
        evaluated_model = self.fetch_db_evaluated_model(trained_model)

        print('DB: done!')
        print('*' * 10 + ' END OF DB SECTION ' + '*' * 10)
        return {'trained_model': trained_model, 'evaluated_model': evaluated_model}

    def add_db_model_evaluation_result_to_dataframe(self, initial_dataframe=pd.DataFrame()):
        final_result = self.load_final_evaluation_results()

        trained_model = final_result['trained_model']
        evaluated_model = final_result['evaluated_model']

        df_model_record = {field_name: getattr(trained_model, field_name)
                           for field_name in TrainedModel._fields.keys()
                           if field_name not in ('all_history', 'best_history')}
        df_model_record['restore_type'] = evaluated_model.restore_type
        df_model_record['test_temperature'] = evaluated_model.test_temperature

        df_evaluation_record = {metric: value['value']
                                for metric, value in evaluated_model.metrics.items()}

        df_record = {**df_model_record, **df_evaluation_record}
        return initial_dataframe.append(df_record, ignore_index=True)

    def add_db_samples_to_dataframe(self, initial_dataframe=pd.DataFrame()):
        result = self.load_samples_with_persample_metrics()

        trained_model = result['trained_model']
        evaluated_model = result['evaluated_model']
        samples = result['generated'] + result['test']

        df_model_record = {field_name: getattr(trained_model, field_name)
                           for field_name in TrainedModel._fields.keys()
                           if field_name not in ('all_history', 'best_history')}
        df_model_record['restore_type'] = evaluated_model.restore_type
        df_model_record['test_temperature'] = evaluated_model.test_temperature

        df_records = [
            {
                **df_model_record,
                **{metric: value['value'] for metric, value in sample.metrics.items()},
                **{field_name: getattr(sample, field_name)
                   for field_name in Sample._fields.keys()
                   if field_name not in ('evaluated_model', 'tokens', 'metrics')},
            }
            for sample in samples
        ]

        return initial_dataframe.append(df_records, ignore_index=True)

    @staticmethod
    def get_stringified_temperature(temperature):
        if temperature is None:
            return ''
        elif isinstance(temperature, str):
            return temperature
        elif temperature['value'] is None:
            return ''
        elif temperature['type'] == 'biased':
            return format(temperature['value'], '0.6f')
        elif temperature['type'] == 'unbiased':
            return 'unbiased-' + format(temperature['value'], '0.6f')
        else:
            raise BaseException('invalid temperature type!!')

    @staticmethod
    def persample_metrics_exist_for_each_sample(dumping_object):
        for group_key, group_value in dumping_object.items():
            lens = [len(v) for v in group_value.values()]
            result = reduce(lambda prev, v: prev and (v == lens[0]), lens, True)
            if result is not True:
                print('{} : {} : {} has invalid persample metrics length'
                      .format(
                          group_key,
                          list(group_value.keys()),
                          lens)
                      )
                return False
        return True


if __name__ == "__main__":
    db_manager = ModelDBManager("", "coco", "vae", 0, restore_type="last_iter")
    # df_model = db_manager.add_db_model_evaluation_result_to_dataframe()
    # print(df_model)
    df_samples = db_manager.add_db_samples_to_dataframe()
    print(df_samples)

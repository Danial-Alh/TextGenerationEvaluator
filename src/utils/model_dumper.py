import os
from functools import reduce

import mongoengine
import numpy as np
import pandas as pd
import pymongo

from db_management.models import (InTrainingEvaluationHistory,
                                  MetricHistoryRecord, MetricResult, Model,
                                  ModelEvaluationResult, Sample)
from utils.file_handler import (create_folder_if_not_exists, unzip_file,
                                zip_folder)
from utils.path_configs import COMPUTER_NAME, EXPORT_PATH, MODEL_PATH

# from .base_evaluator import BaseModel


class ModelDumpManager:
    def __init__(self, model, run=0, dm_name=''):
        self.run = run
        self.dm_name = dm_name
        self.model = model
        self.init_paths()

    def init_paths(self):
        self.saving_path = MODEL_PATH + self.dm_name + '/' + self.model.get_name() + ('_run%d/' % self.run)
        create_folder_if_not_exists(self.saving_path)
        self.final_result_parent_path = os.path.join(
            EXPORT_PATH, 'evaluations_{}/'.format(self.dm_name))
        create_folder_if_not_exists(self.final_result_parent_path)
        self.final_result_file_name = self.model.get_name()

    def store_better_model(self, key):
        zip_folder(self.model.get_saving_path(), key, self.saving_path)

    def restore_model(self, key):
        self.model.delete_saved_model()
        unzip_file(key, self.saving_path, self.model.get_saving_path())
        self.model.load()
        print('Run %d - "%s" based "%s"; saved model restored' %
              (self.run, key, self.model.get_name()))

    def init_history(self, initial_scores=None):
        print('*' * 10 + ' DB SECTION ' + '*' * 10)
        print('DB: creating in_training_evaluation_history record.')
        record = InTrainingEvaluationHistory(
            machine_name=COMPUTER_NAME, model_name=self.model.get_name(),
            dataset_name=self.dm_name, run=self.run,
            best_history={}, all_history={}
        )

        try:
            record.save()
        except mongoengine.errors.NotUniqueError as e:
            print(e)
            if input('InTrainingEvaluationHistory record found. delete it? (y/N)') == 'y':
                InTrainingEvaluationHistory.objects(
                    model_name=self.model.get_name(),
                    dataset_name=self.dm_name, run=self.run,
                ).get().delete()
                print("DB: old record deleted!")
                record.save()
            else:
                print("duplicate keys not allowed!")
                raise e

        if initial_scores is not None:
            self.append_to_history(initial_scores)
            self.append_to_best_history(initial_scores)
        print('done!')
        print('*' * 10 + ' END OF DB SECTION ' + '*' * 10)

    def append_to_best_history(self, new_metrics):
        print('*' * 10 + ' DB SECTION ' + '*' * 10)
        print('DB: append to best history.')
        model_history = InTrainingEvaluationHistory.objects(
            machine_name=COMPUTER_NAME, model_name=self.model.get_name(),
            dataset_name=self.dm_name, run=self.run,
        ).get()

        for metric, value in new_metrics.items():
            if metric not in model_history.best_history:
                model_history.best_history[metric] = []
            model_history.best_history[metric].append(MetricHistoryRecord(**value))

        model_history.save()
        print('done!')
        print('*' * 10 + ' END OF DB SECTION ' + '*' * 10)

    def append_to_history(self, new_metrics):
        print('*' * 10 + ' DB SECTION ' + '*' * 10)
        print('DB: append to history.')
        model_history = InTrainingEvaluationHistory.objects(
            machine_name=COMPUTER_NAME, model_name=self.model.get_name(),
            dataset_name=self.dm_name, run=self.run,
        ).get()

        for metric, value in new_metrics.items():
            if metric not in model_history.all_history:
                model_history.all_history[metric] = []
            model_history.all_history[metric].append(MetricHistoryRecord(**value))

        model_history.save()
        print('done!')
        print('*' * 10 + ' END OF DB SECTION ' + '*' * 10)

    def fetch_db_model(self, restore_type, temperature):
        return Model.objects(
            model_name=self.model.get_name(),
            dataset_name=self.dm_name, run=self.run, restore_type=restore_type,
            temperature=self.get_temperature_stringified(temperature),
        ).get()

    def dump_samples_with_persample_metrics(self, dumping_object: dict, restore_type, temperature):
        def convert_dmpobj_to_db_sample(dmp_obj, origin):
            return [
                Sample(
                    model=model,
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

        assert ModelDumpManager.persample_metrics_exist_for_each_sample(dumping_object)

        print('DB: creating model record.')

        model = Model(
            machine_name=COMPUTER_NAME, model_name=self.model.get_name(),
            dataset_name=self.dm_name, run=self.run, restore_type=restore_type,
            temperature=self.get_temperature_stringified(temperature),
        )

        try:
            model.save()
        except mongoengine.errors.NotUniqueError as e:
            print(e)
            if input('Model record found. delete it? (y/N)') == 'y':
                Model.objects(
                    model_name=self.model.get_name(),
                    dataset_name=self.dm_name, run=self.run, restore_type=restore_type,
                    temperature=self.get_temperature_stringified(temperature),
                ).get().delete()
                print("DB: old record deleted!")
                model.save()
            else:
                print("duplicate keys not allowed!")
                raise e

        print('done!')

        print('DB: creating { generated, test } samples record.')
        generated_samples = convert_dmpobj_to_db_sample(dumping_object, origin='generated')
        test_samples = convert_dmpobj_to_db_sample(dumping_object, origin='test')

        Sample.objects.insert(generated_samples)
        Sample.objects.insert(test_samples)

        print('DB: {} generated samples and {} test samples inserted into database! done!'
              .format(len(generated_samples), len(test_samples)))
        print('*' * 10 + ' END OF DB SECTION ' + '*' * 10)

    def update_persample_metrics_for_generated_samples(self, dumping_object: dict, restore_type, temperature):
        from pymongo import UpdateOne

        print('*' * 10 + ' DB SECTION ' + '*' * 10)
        print('DB: updating generated samples metrics.')

        model = self.fetch_db_model(restore_type, temperature)

        generated_samples = Sample.objects(model=model, origin='generated').order_by('+index')
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
            for i, nm in enumerate(new_metrics) if len(new_metrics) != 0
        ]
        Sample._get_collection().bulk_write(update_operations, ordered=False)
        print('done!')
        print('*' * 10 + ' END OF DB SECTION ' + '*' * 10)

    def load_samples_with_persample_metrics(self, restore_type, temperature):
        print('*' * 10 + ' DB SECTION ' + '*' * 10)
        print('DB: fetching { generated, test } samples.')

        model = self.fetch_db_model(restore_type, temperature)

        generated_samples = Sample.objects(model=model, origin='generated').order_by('+index')
        test_samples = Sample.objects(model=model, origin='test').order_by('+index')

        generated_samples = list(generated_samples)
        test_samples = list(test_samples)

        print('{} generated samples and {} test samples fetched! done!'.format(len(generated_samples),
                                                                               len(test_samples)))
        print('*' * 10 + ' END OF DB SECTION ' + '*' * 10)
        return {'model': model, 'generated': generated_samples, 'test': test_samples}

    def dump_final_evaluation_results(self, dumping_object: dict, restore_type, temperature):
        print('*' * 10 + ' DB SECTION ' + '*' * 10)
        print('DB: inserting model_evaluation_result record.')

        model = self.fetch_db_model(restore_type, temperature)

        evaluation_result = ModelEvaluationResult(model=model)

        for metric, value in dumping_object.items():
            if isinstance(value, dict):
                evaluation_result.metrics[metric] = MetricResult(**value)
            else:
                evaluation_result.metrics[metric] = MetricResult(value=value)

        try:
            evaluation_result.save()
        except mongoengine.errors.NotUniqueError as e:
            print(e)
            if input('ModelEvaluationResult record found. delete it? (y/N)') == 'y':
                ModelEvaluationResult.objects(model=model).get().delete()
                print("DB: old record deleted!")
                evaluation_result.save()
            else:
                print("duplicate keys not allowed!")
                raise e

        print('done!')
        print('*' * 10 + ' END OF DB SECTION ' + '*' * 10)

    def load_final_evaluation_results(self, restore_type, temperature):
        print('*' * 10 + ' DB SECTION ' + '*' * 10)
        print('DB: fetcing model_evaluation_result record.')

        model = self.fetch_db_model(restore_type, temperature)

        result = ModelEvaluationResult.objects(model=model).get()

        print('done!')
        print('*' * 10 + ' END OF DB SECTION ' + '*' * 10)
        return {'model': model, 'result': result}

    def add_db_model_evaluation_result_to_dataframe(self, restore_type, temperature, initial_dataframe=pd.DataFrame()):
        final_result = self.load_final_evaluation_results(restore_type, temperature)

        model = final_result['model']
        model_result = final_result['result']

        df_model_record = {field_name:getattr(model, field_name)
                           for field_name in Model._fields.keys()}
        df_result_record = {metric: value['value']
                            for metric, value in model_result.metrics.items()}

        df_record = {**df_model_record, **df_result_record}
        return initial_dataframe.append(df_record, ignore_index=True)

    def add_db_samples_to_dataframe(self, restore_type, temperature, initial_dataframe=pd.DataFrame()):
        final_result = self.load_samples_with_persample_metrics(restore_type, temperature)

        model = final_result['model']
        samples = final_result['generated'] + final_result['test']

        df_model_record = {field_name:getattr(model, field_name)
                           for field_name in Model._fields.keys()}
        df_records = [
            {
                **df_model_record,
                **{metric: value['value'] for metric, value in sample.metrics.items()}
            }
            for sample in samples
        ]

        return initial_dataframe.append(df_records, ignore_index=True)

    @staticmethod
    def get_temperature_stringified(temperature):
        if temperature['value'] is None:
            return ''
        elif temperature['type'] == 'biased':
            return '_' + format(temperature['value'], '0.6f')
        elif temperature['type'] == 'unbiased':
            return '_unbiased' + format(temperature['value'], '0.6f')
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
    from data_management.data_manager import load_real_dataset
    from previous_works.model_wrappers import create_model
    DATASET_NAME = "coco"
    _, _, _, TEXT = load_real_dataset(DATASET_NAME)
    m = create_model('vae', TEXT)
    dumper = ModelDumpManager(m, run=0, dm_name=DATASET_NAME)
    df_model = dumper.add_db_model_evaluation_result_to_dataframe('last_iter', {'value': None})
    print(df_model)
    # df_samples = dumper.add_db_samples_to_dataframe('last_iter', {'value': None})
    # print(df_samples)

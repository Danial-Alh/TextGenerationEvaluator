from functools import reduce
import os

import numpy as np

from db_management.models import (InTrainingEvaluationHistory,
                                  MetricHistoryRecord, MetricResult,
                                  ModelEvaluationResult, Model,
                                  Sample)
from utils.file_handler import (create_folder_if_not_exists,
                                unzip_file, zip_folder)
from utils.path_configs import COMPUTER_NAME, EXPORT_PATH, MODEL_PATH

# from .base_evaluator import BaseModel


class ModelDumper:
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
        print('Run %d - "%s" based "%s"; saved model restored' %
              (self.run, key, self.model.get_name()))
        self.model.load()

    def init_history(self, initial_scores=None):
        InTrainingEvaluationHistory(
            machine_name=COMPUTER_NAME, model_name=self.model.get_name(),
            dataset_name=self.dm_name, run=self.run,
            best_history={}, all_history={}
        ).save()

        if initial_scores is not None:
            self.append_to_history(initial_scores)
            self.append_to_best_history(initial_scores)

    def append_to_best_history(self, new_metrics):
        model_history = InTrainingEvaluationHistory.objects(
            machine_name=COMPUTER_NAME, model_name=self.model.get_name(),
            dataset_name=self.dm_name, run=self.run,
        ).get()

        for metric, value in new_metrics.items():
            if metric not in model_history.best_history:
                model_history.best_history[metric] = []
            model_history.best_history[metric].append(MetricHistoryRecord(**value))

        model_history.save()

    def append_to_history(self, new_metrics):
        model_history = InTrainingEvaluationHistory.objects(
            machine_name=COMPUTER_NAME, model_name=self.model.get_name(),
            dataset_name=self.dm_name, run=self.run,
        ).get()

        for metric, value in new_metrics.items():
            if metric not in model_history.all_history:
                model_history.all_history[metric] = []
            model_history.all_history[metric].append(MetricHistoryRecord(**value))

        model_history.save()

    def dump_samples_with_persample_metrics(self, dumping_object: dict, restore_type, temperature):
        def convert_dmpobj_to_db_sample(dmp_obj, origin):
            return [
                Sample(
                    index=i,
                    origin=origin,
                    tokens=dmp_obj[origin]['tokens'][i],
                    sentence=dmp_obj[origin]['sentence'][i],
                    metrics={key: MetricResult(value=value[i])
                             for key, value in dmp_obj[origin].items() if key not in ['sentence', 'tokens']}
                )
                for i in range(len(dmp_obj[origin]['sentence']))
            ]

        assert ModelDumper.persample_metrics_exist_for_each_sample(dumping_object)

        model = Model(
            machine_name=COMPUTER_NAME, model_name=self.model.get_name(),
            dataset_name=self.dm_name, run=self.run, restore_type=restore_type,
            temperature=self.get_temperature_stringified(temperature),
        )

        model.save()

        generated_samples = convert_dmpobj_to_db_sample(dumping_object, origin='generated')
        test_samples = convert_dmpobj_to_db_sample(dumping_object, origin='test')

        Sample.objects.insert(generated_samples)
        Sample.objects.insert(test_samples)

    def update_persample_metrics_for_generated_samples(self, dumping_object: dict, restore_type, temperature):
        from pymongo import UpdateOne

        model = Model.objects(
            machine_name=COMPUTER_NAME, model_name=self.model.get_name(),
            dataset_name=self.dm_name, run=self.run, restore_type=restore_type,
            temperature=self.get_temperature_stringified(temperature),
        ).get()

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

    def load_samples_with_persample_metrics(self, restore_type, temperature):
        result = ModelSamples.objects(
            model_name=self.model.get_name(), dataset_name=self.dm_name,
            run=self.run, restore_type=restore_type,
            temperature=self.get_temperature_stringified(temperature)).get()
        return result

    def dump_final_evaluation_results(self, dumping_object: dict, restore_type, temperature):
        evaluation_result = ModelEvaluationResult(
            machine_name=COMPUTER_NAME, model_name=self.model.get_name(),
            dataset_name=self.dm_name, run=self.run, restore_type=restore_type,
            temperature=self.get_temperature_stringified(temperature),
        )

        for metric, value in dumping_object.items():
            if isinstance(value, dict):
                evaluation_result.metrics[metric] = MetricResult(**value)
            else:
                evaluation_result.metrics[metric] = MetricResult(value=value)

        evaluation_result.save()

    def load_final_evaluation_results(self, restore_type, temperature):
        return ModelEvaluationResult.objects(
            machine_name=COMPUTER_NAME, model_name=self.model.get_name(),
            dataset_name=self.dm_name, run=self.run, restore_type=restore_type,
            temperature=self.get_temperature_stringified(temperature),
        ).order_by('-created_at').first()

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

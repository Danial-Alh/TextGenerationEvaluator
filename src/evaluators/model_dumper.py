import os

from utils.file_handler import (create_folder_if_not_exists, dump_json,
                                load_json, read_text, unzip_file, write_text,
                                zip_folder)
from utils.path_configs import EXPORT_PATH, MODEL_PATH

from .base_evaluator import BaseModel


class ModelDumper:
    def __init__(self, model: BaseModel, k=0, dm_name=''):
        self.k = k
        self.dm_name = dm_name
        self.model = model
        self.init_paths()

    def init_paths(self):
        self.saving_path = MODEL_PATH + self.dm_name + '/' + self.model.get_name() + ('_run%d/' % self.k)
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
              (self.k, key, self.model.get_name()))
        self.model.load()

    def dump_generated_samples(self, samples, load_key, temperature):
        if not isinstance(samples[0], str):
            samples = [" ".join(s) for s in samples]
        write_text(samples, os.path.join(self.saving_path, self.get_temperature_stringified(temperature) + '_' +
                                         load_key + '_based_samples.txt'), is_complete_path=True)

    def load_generated_samples(self, load_key, temperature):
        return read_text(os.path.join(self.saving_path, self.get_temperature_stringified(temperature) + '_' +
                                      load_key + '_based_samples.txt'), is_complete_path=True)

    def get_generated_samples_path(self, load_key, temperature):
        return os.path.join(self.saving_path, self.get_temperature_stringified(temperature) + '_' +
                            load_key + '_based_samples.txt')

    def dump_best_history(self, best_history):
        dump_json(best_history, 'best_history', self.saving_path)

    def dump_samples_with_supplementary_info(self, dumping_object: dict, load_key, sample_label, temperature):
        print({"#" + key: len(value) for key, value in dumping_object.items()})
        dump_json(
            [
                {key: value[i] for key, value in dumping_object.items()}
                for i in range(len(dumping_object['sentence']))
            ],
            sample_label + self.get_temperature_stringified(temperature) + '_' +
            load_key + '_based_samples', self.saving_path
        )

    def load_samples_with_supplementary_info(self, load_key, sample_label, temperature):
        result = load_json(
            sample_label +
            self.get_temperature_stringified(temperature) + '_' + load_key + '_based_samples',
            self.saving_path)
        return result

    def dump_final_results(self, results, restore_type, temperature):
        dump_json(results, self.final_result_file_name + self.get_temperature_stringified(temperature) +
                  '_' + restore_type + 'restore', self.final_result_parent_path)

    def load_final_results(self, restore_type, temperature):
        return load_json(self.final_result_file_name + self.get_temperature_stringified(temperature) +
                         '_' + restore_type + 'restore', self.final_result_parent_path)

    def dump_final_results_details(self, results, restore_type, temperature):
        dump_json(results, self.final_result_file_name + self.get_temperature_stringified(temperature)
                  + '_' + restore_type + 'restore_details',
                  self.final_result_parent_path)

    def load_final_results_details(self, restore_type, temperature):
        return load_json(self.final_result_file_name + self.get_temperature_stringified(temperature)
                         + '_' + restore_type + 'restore_details',
                         self.final_result_parent_path)

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

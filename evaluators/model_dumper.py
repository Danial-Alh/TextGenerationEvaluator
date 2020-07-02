class ModelDumper:
    def __init__(self, model: BaseModel, k=0, dm_name=''):
        self.k = k
        self.dm_name = dm_name
        self.model = model

        self.init_paths()

    def init_paths(self):
        self.saving_path = MODEL_PATH + self.dm_name + '/' + self.model.get_name() + ('_k%d/' % self.k)
        create_folder_if_not_exists(self.saving_path)
        self.final_result_parent_path = os.path.join(EXPORT_PATH, 'evaluations_{}k-fold_k{}_{}/' \
                                                     .format(k_fold, self.k, self.dm_name))
        create_folder_if_not_exists(self.final_result_parent_path)
        self.final_result_file_name = self.model.get_name()

    def store_better_model(self, key):
        zip_folder(self.model.get_saving_path(), key, self.saving_path)

    def restore_model(self, key):
        self.model.delete_saved_model()
        unzip_file(key, self.saving_path, self.model.get_saving_path())
        print('K%d - "%s" based "%s" saved model restored' % (self.k, key, self.model.get_name()))
        self.model.load()

    def dump_generated_samples(self, samples, load_key, temperature):
        if not isinstance(samples[0], str):
            samples = [" ".join(s) for s in samples]
        write_text(samples, os.path.join(self.saving_path, get_temperature_stringified(temperature) + '_' +
                                         load_key + '_based_samples.txt'), is_complete_path=True)

    def load_generated_samples(self, load_key, temperature):
        return read_text(os.path.join(self.saving_path, get_temperature_stringified(temperature) + '_' +
                                      load_key + '_based_samples.txt'), is_complete_path=True)

    def get_generated_samples_path(self, load_key, temperature):
        return os.path.join(self.saving_path, get_temperature_stringified(temperature) + '_' +
                            load_key + '_based_samples.txt')

    def dump_best_history(self, best_history):
        dump_json(best_history, 'best_history', self.saving_path)

    def dump_samples_with_additional_fields(self, samples, additional_fields: dict, load_key, sample_label,
                                            temperature):
        print(len(samples))
        print([len(additional_fields[key]) for key in additional_fields.keys()])
        dump_json([
            {**{'text': samples[i]}, **{key: additional_fields[key][i] for key in additional_fields.keys()}}
            for i in range(len(samples))],
            sample_label + get_temperature_stringified(temperature) + '_' + load_key + '_based_samples',
            self.saving_path)

    def load_samples_with_additional_fields(self, load_key, sample_label, temperature):
        result = load_json(
            sample_label + get_temperature_stringified(temperature) + '_' + load_key + '_based_samples',
            self.saving_path)
        return result

    def dump_final_results(self, results, restore_type, temperature):
        dump_json(results, self.final_result_file_name + get_temperature_stringified(temperature) +
                  '_' + restore_type + 'restore', self.final_result_parent_path)

    def load_final_results(self, restore_type, temperature):
        return load_json(self.final_result_file_name + get_temperature_stringified(temperature) +
                         '_' + restore_type + 'restore', self.final_result_parent_path)

    def dump_final_results_details(self, results, restore_type, temperature):
        dump_json(results, self.final_result_file_name + get_temperature_stringified(temperature)
                  + '_' + restore_type + 'restore_details',
                  self.final_result_parent_path)

    def load_final_results_details(self, restore_type, temperature):
        return load_json(self.final_result_file_name + get_temperature_stringified(temperature)
                         + '_' + restore_type + 'restore_details',
                         self.final_result_parent_path)

import os
import pickle

import numpy as np

from path_configs import DATASET_PATH, OBJ_fILES_PATH


def read_text(file_name):
    with open(DATASET_PATH + file_name + '.txt', 'r', encoding='utf8') as file:
        lines = file.readlines()
    lines = [l.replace('\n', '') for l in lines if l.strip() != '']
    return np.array(lines)


def write_text(lines, file_name):
    path = DATASET_PATH + file_name + '.txt'
    with open(path, 'w', encoding='utf8') as file:
        file.write("\n".join(lines))
    print('data written to file %s!' % file_name)
    return path


def exists(file_name, prefix=OBJ_fILES_PATH):
    return os.path.exists(prefix + file_name)


def dump(obj, file_name):
    with open(OBJ_fILES_PATH + file_name, 'wb') as file:
        pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)


def load(file_name):
    with open(OBJ_fILES_PATH + file_name, 'rb') as file:
        return pickle.load(file)


def is_folder_empty(folder_path):
    return not exists(folder_path, '') or (len(os.listdir(folder_path)) == 0)


class PersistentClass:
    SUCCESSFUL_LOAD = 0
    FAILED_LOAD = 1

    def __init__(self, savable_fields=None, concatenated_name=''):
        self.__savable_fields = savable_fields
        self.class_name = ''
        self.__file_name = ''
        self.class_name = self.__class__.__name__.lower() + \
                          ('' if concatenated_name == '' else ('_' + concatenated_name))
        self.__file_name = self.class_name

    def update_names(self, concatenated_name=''):
        self.class_name = self.__class__.__name__.lower() + \
                          ('' if concatenated_name == '' else ('_' + concatenated_name))
        self.__file_name = self.class_name

    def extend_savable_fields(self, new_savable_fields):
        if self.__savable_fields is None:
            self.__savable_fields = []
        self.__savable_fields.extend(new_savable_fields)

    def save(self):
        temp_obj = tuple([self.__getattribute__(att_name) for att_name in self.__savable_fields])
        with open(OBJ_fILES_PATH + self.__file_name, 'wb') as file:
            pickle.dump(temp_obj, file, pickle.HIGHEST_PROTOCOL)

    def load(self):
        if exists(self.__file_name):
            with open(OBJ_fILES_PATH + self.__file_name, 'rb') as file:
                loaded_obj = pickle.load(file)
                for i, att_name in enumerate(self.__savable_fields):
                    self.__setattr__(att_name, loaded_obj[i])
                print('[#] class "%s" loaded!' % self.class_name)
                return PersistentClass.SUCCESSFUL_LOAD
        return PersistentClass.FAILED_LOAD

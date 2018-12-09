import json
import os
import pickle
import zipfile
from zipfile import ZipFile

import numpy as np

from path_configs import DATASET_PATH, OBJ_fILES_PATH


def read_text(file_name, is_complete_path=False):
    if not is_complete_path:
        path = DATASET_PATH + file_name + '.txt'
    else:
        path = file_name
    with open(path, 'r', encoding='utf8') as file:
        lines = file.readlines()
    lines = [l.replace('\n', '') for l in lines if l.strip() != '']
    return np.array(lines)


def write_text(lines, file_name, is_complete_path=False):
    if not is_complete_path:
        path = DATASET_PATH + file_name + '.txt'
    else:
        path = file_name
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


def zip_folder(zipping_path, zipped_file_name, destination):
    zipping_path = os.path.abspath(zipping_path)
    zipped_file_path = destination + ('' if destination.endswith('/') else '/') + zipped_file_name + '.zip'
    with ZipFile(zipped_file_path, 'w', compression=zipfile.ZIP_BZIP2) as compressor:
        for root, dirs, files in os.walk(zipping_path):
            for file in files:
                file_abs_path = os.path.join(root, file)
                compressor.write(file_abs_path, arcname=file_abs_path[len(zipping_path):])
    return zipped_file_path


def unzip_file(zipped_file_name, zipped_file_parent_path, extraction_path):
    zipped_file_path = zipped_file_parent_path + ('' if zipped_file_parent_path.endswith('/') else '/') + \
                       zipped_file_name + '.zip'
    with ZipFile(zipped_file_path, 'r') as extractor:
        extractor.extractall(extraction_path)


def create_folder_if_not_exists(paths):
    def create(p):
        if not os.path.exists(p):
            os.makedirs(p)

    if isinstance(paths, str):
        create(paths)
        return
    for p in paths:
        create(p)


def dump_json(obj, file_name, parent_path):
    path = os.path.join(parent_path, file_name) + '.json'
    with open(path, 'w') as file:
        json.dump(obj, file, indent='\t')


def load_json(file_name, parent_path):
    path = os.path.join(parent_path, file_name) + '.json'
    with open(path, 'r') as file:
        return json.load(file)


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

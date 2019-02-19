from functools import reduce

import numpy as np

from data_management.data_loaders import SentenceDataloader
from data_management.parsers import WordBasedParser, Parser
from utils.file_handler import PersistentClass, write_text
from utils.path_configs import DATASET_PATH, TEMP_PATH


class DataManager(PersistentClass):
    def __init__(self, train_data_loader, test_data_loader, parser, k_fold=3, concatenated_name=''):
        super().__init__(concatenated_name=concatenated_name + '_{}k-fold'.format(k_fold))
        super().extend_savable_fields(
            ['structured_data',
             'train_k_indices',
             'k_fold']
        )
        self.parser = parser
        self.data_loaders = {'train': train_data_loader, 'test': test_data_loader}
        self.k_fold = k_fold

        self.structured_data = {'train': [], 'test': []}
        self.train_k_indices = [[] for _ in range(self.k_fold)]

        if self.load() == PersistentClass.FAILED_LOAD:
            self._parse_raw_data(*self._load_data())
            self._split_to_k_fold()
            self.save()

    def load(self):
        result = super().load()
        if result == PersistentClass.SUCCESSFUL_LOAD:
            assert self.parser.load() == PersistentClass.SUCCESSFUL_LOAD, 'datamanager exists but parser not found!,' \
                                                                          ' delete datamanager obj first!'
        return result

    def _load_data(self):
        train_raw_data = self.data_loaders['train'].get_data()
        test_raw_data = self.data_loaders['test'].get_data()
        train_raw_data = np.array(train_raw_data)
        test_raw_data = np.array(test_raw_data)
        np.random.shuffle(train_raw_data)
        return train_raw_data, test_raw_data

    def _parse_raw_data(self, train_raw_data, test_raw_data):
        self.parser.init_data(self._prepare_data_for_parser(np.concatenate((train_raw_data, test_raw_data))))
        pass

    def _split_to_k_fold(self):
        total_dataset_size = self.structured_data['train'].shape[0]
        remained_indices = np.array(list(range(total_dataset_size)))
        slot_size = int(total_dataset_size / self.k_fold)
        for k in range(self.k_fold - 1):
            self.train_k_indices[k] = np.random.choice(remained_indices, slot_size, replace=False)
            remained_indices = remained_indices[[(index not in self.train_k_indices[k]) for index in remained_indices]]
        self.train_k_indices[-1] = remained_indices

    def __subsample(self, data, subsample_size=-1, unpack=True):
        if subsample_size != -1:
            data = np.random.choice(data, subsample_size)
        if unpack:
            return self.unpack_data(data)
        return data

    def get_data(self, k, subsample_size=-1, parse=False):  # k = -1 to get test data
        if k == -1:
            test_data = self.structured_data['test']
            return self.__post_process(test_data, subsample_size, parse)
        if self.k_fold == 1:
            train_data = self.__get_data_of_k(0)
            return self.__post_process(train_data, subsample_size, parse), None
        train_data = self.__get_data_of_k([kk for kk in range(self.k_fold) if kk != k])
        valid_data = self.__get_data_of_k(k)
        return self.__post_process(train_data, subsample_size, parse), \
               self.__post_process(valid_data, subsample_size, parse)

    def __post_process(self, data, subsample_size, parse):
        data = self.__subsample(data, subsample_size, unpack=True)
        if parse:
            data = self.parser.id_format2line(data)
        return data

    def dump_data_on_file(self, k, parse, file_name, parent_folder=DATASET_PATH, subsample_size=-1):
        if k == -1:
            test_data = self.structured_data['test']
            return self.__dump_post_process(test_data, file_name + '{}_test'.format(self.class_name),
                                                     parent_folder, subsample_size, parse)
        if self.k_fold == 1:
            train_data = self.__get_data_of_k(0)
            return self.__dump_post_process(train_data, file_name + '{}_train_k{}'.format(self.class_name, k),
                                                      parent_folder, subsample_size, parse), None
        train_data = self.__get_data_of_k([kk for kk in range(self.k_fold) if kk != k])
        valid_data = self.__get_data_of_k(k)
        return self.__dump_post_process(train_data, file_name + '{}_train_k{}'.format(self.class_name, k),
                                                  parent_folder, subsample_size, parse), \
                self.__dump_post_process(valid_data, file_name + '{}_valid_k{}'.format(self.class_name, k),
                                                  parent_folder, subsample_size, parse)

    def __dump_post_process(self, data, file_name, parent_path, subsample_size, parse):
        data = self.__subsample(data, subsample_size, unpack=True)
        if parse:
            file_name += '_parsed'
            data = self.parser.id_format2line(data)
        else:
            data = [' '.join(list(map(str, s))) for s in data]
        return write_text(data, file_name, parent_path)

    def __get_data_of_k(self, k):
        if isinstance(k, int):
            return self.structured_data['train'][self.train_k_indices[k]]
        return reduce(lambda x, y: np.concatenate((x, y), axis=0),
                      [self.structured_data['train'][self.train_k_indices[kk]] for kk in k])

    def _prepare_data_for_parser(self, raw_data):
        pass

    def unpack_data(self, packed_data):
        pass

    def get_parser(self) -> Parser:
        return self.parser


class SentenceDataManager(DataManager):
    def __init__(self, train_data_loader, test_data_loader, parser, k_fold=3, concatenated_name=''):
        super().__init__(train_data_loader, test_data_loader, parser, k_fold,
                         concatenated_name='&'.join(([concatenated_name] if concatenated_name != '' else []) +
                                                    [dl.file_name for dl in [train_data_loader, test_data_loader]]))

    def _prepare_data_for_parser(self, raw_data):
        return raw_data

    def _parse_raw_data(self, train_raw_data, test_raw_data):
        super()._parse_raw_data(train_raw_data, test_raw_data)

        def temp_process(d, tag):
            id_format_lines = [[]]
            for i, d in enumerate(d):
                id_format_line = self.parser.line2id_format(d)
                id_format_lines.append(id_format_line)
            self.structured_data[tag] = np.rec.fromarrays((id_format_lines,),
                                                          shape=(len(id_format_lines),),
                                                          names=('codes',))
            self.structured_data[tag] = self.structured_data[tag][1:]

        temp_process(train_raw_data, 'train')
        temp_process(test_raw_data, 'test')

    def unpack_data(self, packed_data):
        codes = packed_data[:]['codes']
        codes = np.array([s for s in codes])
        return codes


class OracleDataManager(SentenceDataManager):
    def __init__(self, train_data_loader, test_data_loader, parser, k_fold=3):
        super().__init__(train_data_loader, test_data_loader, parser=parser, k_fold=k_fold, concatenated_name='')


if __name__ == '__main__':
    parser = WordBasedParser(name='coco60-words')
    tr_data_loader = SentenceDataloader('coco60-train')
    ts_data_loader = SentenceDataloader('coco60-test')
    dm = SentenceDataManager(tr_data_loader, ts_data_loader, parser=parser)
    # dm = OracleDataManager([SentenceDataloader('oracle37.5')], 'oracle-words')
    tr_data, va_data = dm.get_data(0, parse=True).values()
    print(tr_data[:10], va_data[:10])
    tr_data, va_data = dm.get_data(0, parse=False).values()
    print(tr_data[:10], va_data[:10])
    tr_loc, va_loc = dm.dump_data_on_file(file_name='', parent_folder=TEMP_PATH, k=0, parse=True).values()
    tr_parsed_loc, va_parsed_loc = dm.dump_data_on_file(file_name='', parent_folder=TEMP_PATH, k=0,
                                                        parse=False).values()
    # train_data = dm.get_training_data(0, unpack=False)
    # valid_data = dm.get_validation_data(0, unpack=False)
    # batch_iter = dm.get_batches(valid_data, 32, 'valid')
    # for batch in batch_iter:
    #     # x, x_l, y, y_l, y_gt = batch
    #     x, _, _ = batch
    #     print(x.shape)
    #     # break

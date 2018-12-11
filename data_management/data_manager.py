from functools import reduce

import numpy as np

from data_management.batch_managers import BatchManager
from data_management.data_loaders import SentenceDataloader
from data_management.parsers import WordBasedParser, Parser, OracleBasedParser
from file_handler import PersistentClass, write_text


class DataManager(PersistentClass):
    def __init__(self, data_loaders, parser_classes, parser_names, k_fold=3, concatenated_name=''):
        super().__init__(concatenated_name=concatenated_name)
        super().extend_savable_fields(
            ['data',
             'indices',
             'k_fold']
        )
        self.parser_classes = parser_classes
        self.parser_names = parser_names
        self.data_loaders = data_loaders
        self.k_fold = k_fold

        self.data = []
        self.parsers = None
        self.indices = [[] for _ in range(self.k_fold)]

        if self.load() == PersistentClass.FAILED_LOAD:
            self._load_data()
            self._parse_data()
            self._split_to_k_fold()
            self.save()

    def load(self):
        result = super().load()
        if result == PersistentClass.SUCCESSFUL_LOAD:
            self.parsers = [parser_class(lines=None, name=self.parser_names[i])
                            for i, parser_class in enumerate(self.parser_classes)]
        return result

    def _load_data(self):
        self.data = []
        for data_loader in self.data_loaders:
            self.data.extend(data_loader.get_data())
        self.data = np.array(self.data)
        self.shuffle_data()

    def shuffle_data(self):
        np.random.shuffle(self.data)

    def _split_to_k_fold(self):
        total_dataset_size = self.data.shape[0]
        remained_indices = np.array(list(range(total_dataset_size)))
        slot_size = int(total_dataset_size / self.k_fold)
        for k in range(self.k_fold - 1):
            self.indices[k] = np.random.choice(remained_indices, slot_size, replace=False)
            remained_indices = remained_indices[[(index not in self.indices[k]) for index in remained_indices]]
        self.indices[-1] = remained_indices

    def __subsample(self, data, subsample_size=-1, sample=True, unpack=True):
        if subsample_size == -1:
            pass
        elif sample:
            data = np.random.choice(data, subsample_size)
        elif subsample_size != -1:
            data = data[:subsample_size]
        if unpack:
            return self.unpack_data(data)
        return data

    def get_training_data(self, k, subsample_size=-1, sample=True, unpack=True):
        target_data_set = reduce(lambda x, y: np.concatenate((x, y), axis=0),
                                 [self.data[self.indices[kk]] for kk in range(self.k_fold) if kk != k])
        return self.__subsample(target_data_set, subsample_size, sample, unpack)

    def get_validation_data(self, k, subsample_size=-1, sample=True, unpack=True):
        target_data_set = self.data[self.indices[k]]
        return self.__subsample(target_data_set, subsample_size, sample, unpack)

    def get_batches(self, data, batch_size=32, name=''):
        batch_manager = BatchManager(data, self.unpack_data, batch_size, self.class_name + '_' + name)
        return batch_manager

    def _parse_data(self):
        self.parsers = [parser_class(lines=self._prepare_data_for_parser(i),
                                     name=self.parser_names[i])
                        for i, parser_class in enumerate(self.parser_classes)]
        pass

    def _prepare_data_for_parser(self, parser_id):
        pass

    def unpack_data(self, packed_data):
        pass


class SentenceDataManager(DataManager):
    def __init__(self, data_loaders, parser_name, k_fold=3, concatenated_name='', parsers=None):
        data_loaders = data_loaders
        if parsers is None:
            parsers = [WordBasedParser]
        super().__init__(data_loaders, parsers, [parser_name], k_fold,
                         concatenated_name='&'.join(([concatenated_name] if concatenated_name != '' else []) +
                                                    [dl.file_name for dl in data_loaders]))

    def _prepare_data_for_parser(self, parser_id):
        return list(self.data)

    def _parse_data(self):
        super()._parse_data()
        parser = self.parsers[0]
        self.data = list(self.data)
        id_format_lines, lengths = [[]], [0]
        for i, d in enumerate(self.data):
            id_format_line, l = parser.line2id_format(d)
            id_format_lines.append(id_format_line)
            lengths.append(l)
        self.data = np.rec.fromarrays((id_format_lines, lengths),
                                      shape=(len(id_format_lines),),
                                      names=('text', 'len'))
        self.data = self.data[1:]

    def unpack_data(self, packed_data):
        text, lengths = packed_data[:]['text'], packed_data[:]['len']

        text = np.array([s for s in text])

        text_shifted = np.hstack((text[:, 1:], np.reshape([self.parsers[0].END_TOKEN_ID] * packed_data.shape[0],
                                                          (packed_data.shape[0], 1))))
        return text, text_shifted, lengths

    def get_parser(self) -> Parser:
        return self.parsers[0]

    def dump_unpacked_data_on_file(self, unpacked_data, file_name, parse=False):
        if parse:
            file_name += '_parsed'
        text = unpacked_data[1]  # shifted text
        if parse:
            text = self.get_parser().id_format2line(text, trim=True)
        else:
            text = [list(map(lambda x: str(x), l)) for l in text]
            text = [" ".join(l) for l in text]
        return write_text(text, file_name)

    def get_parsed_unpacked_data(self, unpacked_data):
        text = unpacked_data[1]  # shifted text
        text = self.get_parser().id_format2line(text, trim=True)
        return text

class OracleDataManager(SentenceDataManager):
    def __init__(self, data_loaders, parser_name, k_fold=3):
        data_loaders = data_loaders
        parsers = [OracleBasedParser]
        super().__init__(data_loaders, parsers=parsers, parser_name=parser_name, k_fold=k_fold, concatenated_name='')

    def unpack_data(self, packed_data):
        text, lengths = packed_data[:]['text'], packed_data[:]['len']

        text = np.array([s for s in text])

        text_shifted = np.concatenate(
            (np.reshape([self.parsers[0].START_TOKEN_ID] * packed_data.shape[0], (packed_data.shape[0], 1)),
             text[:, :-1]), axis=1)
        return text_shifted, text, lengths


if __name__ == '__main__':
    # dm = Seq2SeqDataManager([WordDataloader('words-shahnameh-golestan-ghazaliat')],
    #                         ['words'], 32)
    dm = OracleDataManager([SentenceDataloader('oracle37.5')], 'oracle-words')
    train_data = dm.get_training_data(0, unpack=True)
    valid_data = dm.get_validation_data(0, unpack=True)
    tr_loc = dm.dump_unpacked_data_on_file(train_data, 'oracle37.5-train-k0')
    valid_loc = dm.dump_unpacked_data_on_file(valid_data, 'oracle37.5-valid-k0')
    # train_data = dm.get_training_data(0, unpack=False)
    # valid_data = dm.get_validation_data(0, unpack=False)
    # batch_iter = dm.get_batches(valid_data, 32, 'valid')
    # for batch in batch_iter:
    #     # x, x_l, y, y_l, y_gt = batch
    #     x, _, _ = batch
    #     print(x.shape)
    #     # break

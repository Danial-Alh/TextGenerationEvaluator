import numpy as np

from utils.file_handler import PersistentClass


class BatchManager(PersistentClass):
    def __init__(self, data, data_unpacker=None, batch_size=32, concatenated_name=''):
        super().__init__(
            ['data',
             'data_unpacker',
             'epoch',
             'batch_size',
             'current_batch_id',
             'total_batches'],
            'b' + str(batch_size) + '_' + concatenated_name
        )
        self.data = data
        self.data_unpacker = data_unpacker
        self.batch_size = batch_size
        self.current_batch_id = 0
        self.epoch = 0
        self.total_batches = -1
        self.load()

    def load(self):
        if super().load() == PersistentClass.SUCCESSFUL_LOAD:
            if self.current_batch_id != 0:
                self.current_batch_id -= 1

    def reset(self):
        self.current_batch_id = 0
        self.epoch = 0
        self.total_batches = -1
        print('batch manager reset!')

    def __iter__(self):
        if self.total_batches == -1:
            self.total_batches = int(np.ceil(len(self.data) / float(self.batch_size)))
        print('epoch: %d, batch starts from %d, total batches: %d' % (self.epoch, self.current_batch_id,
                                                                      self.total_batches))
        return self

    def __next__(self):
        if self.current_batch_id == self.total_batches:
            self.epoch += 1
            self.current_batch_id = 0
            self.save()
            raise StopIteration
        if self.current_batch_id == (self.total_batches - 1):
            current_slice = slice(self.batch_size * self.current_batch_id,
                                  len(self.data))
        else:
            current_slice = slice(self.batch_size * self.current_batch_id,
                                  self.batch_size * (self.current_batch_id + 1))
        batch = self.data[current_slice]
        self.current_batch_id += 1
        self.save()
        if self.data_unpacker is None:
            return batch
        return self.data_unpacker(batch)

    def __len__(self):
        return self.total_batches

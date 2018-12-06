import numpy as np

from data_management.data_loaders import SentenceDataloader
from data_management.data_manager import SentenceDataManager
from file_handler import read_text, zip_folder, create_folder_if_not_exists, unzip_file, dump_json
from metrics.bleu import Bleu
from models import BaseModel, TexyGen
from path_configs import MODEL_PATH
from utils import tokenize


class Evaluator:
    def __init__(self, data_manager: SentenceDataManager, k=0, name=''):
        self.k = k
        self.name = name
        self.data_manager = data_manager
        self.data_manager = data_manager
        self.init_dataset()
        self.init_metrics()

    def init_dataset(self):
        train_data = self.data_manager.get_training_data(self.k, unpack=True)
        valid_data = self.data_manager.get_validation_data(self.k, unpack=True)
        self.train_loc = self.data_manager.dump_unpacked_data_on_file(train_data, self.name + '-train-k' + str(self.k))
        self.valid_loc = self.data_manager.dump_unpacked_data_on_file(valid_data, self.name + '-valid-k' + str(self.k))

    def init_metrics(self):
        id_format_valid_tokenized = tokenize(read_text(self.valid_loc, True))
        valid_texts = self.data_manager.get_parser().id_format2line(id_format_valid_tokenized, True)
        valid_tokens = tokenize(valid_texts)
        self.bleu3 = Bleu(valid_tokens, weights=np.ones(3) / 3.)
        self.bleu4 = Bleu(valid_tokens, weights=np.ones(4) / 4.)
        self.bleu5 = Bleu(valid_tokens, weights=np.ones(5) / 5.)


class ModelDumper:
    def __init__(self, model: BaseModel, evaluator: Evaluator, k=0, name=''):
        self.k = k
        self.model = model
        self.evaluator = evaluator
        self.name = name
        self.best_history = {
            'bleu3': [{"value": 0.0, "epoch": -1}],
            'bleu4': [{"value": 0.0, "epoch": -1}],
            'bleu5': [{"value": 0.0, "epoch": -1}],
            '-nll': [{"value": -np.inf, "epoch": -1}]
        }
        self.n_sampling = 1000

        self.init_paths()
        self.model.set_dumper(self)
        self.model.delete_saved_model()
        self.model.set_train_val_loc(self.evaluator.train_loc, self.evaluator.valid_loc)

    def init_paths(self):
        self.saving_path = MODEL_PATH + self.name + '/' + self.model.get_name() + ('_k%d/' % self.k)
        create_folder_if_not_exists(self.saving_path)

    def store_better_model(self, key):
        zip_folder(self.model.get_saving_path(), key, self.saving_path)

    def restore_model(self, key):
        self.model.delete_saved_model()
        unzip_file(key, self.saving_path, self.model.get_saving_path())
        self.model.load()
        print('K%d - "%s" based "%s" saved model restored' % (self.k, key, self.model.get_name()))

    def update_scores(self, epoch=0, last_iter=False):
        if last_iter:
            self.store_better_model('last_iter')
            print('K%d - model "%s", epoch -, last iter model saved!' % (self.k, self.model.get_name()))
            return
        new_samples = tokenize(self.model.generate_samples(self.n_sampling))
        new_scores = {
            'bleu3': np.mean(self.evaluator.bleu3.get_score(new_samples)),
            'bleu4': np.mean(self.evaluator.bleu4.get_score(new_samples)),
            'bleu5': np.mean(self.evaluator.bleu5.get_score(new_samples)),
            '-nll': -float(self.model.get_nll())
        }

        for key, new_v in new_scores.items():
            if self.best_history[key][-1]['value'] < new_v:
                print('K%d - model "%s", epoch %d, found better score for "%s": %.4f' %
                      (self.k, self.model.get_name(), epoch, key, new_v))
                self.store_better_model(key)
                self.best_history[key].append({"value": new_v, "epoch": epoch})
                dump_json(self.best_history, 'best_history', self.saving_path)


if __name__ == '__main__':
    import sys

    k_fold = 3
    m_name = sys.argv[1]
    dataset_name = sys.argv[2]
    k = int(sys.argv[3])
    # k = 1
    dm_name = dataset_name.split('-')[0]
    dm = SentenceDataManager([SentenceDataloader(dataset_name)], dm_name + '-words', k_fold=k_fold)

    ev = Evaluator(dm, k, dm_name)
    m = TexyGen(m_name, dm.get_parser())
    dumper = ModelDumper(m, ev, k, dm_name)
    m.train()
    # m.delete()

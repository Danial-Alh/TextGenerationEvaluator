import io
import torchtext
from torchtext.data import Field, ReversibleField, TabularDataset
import numpy as np

from utils.path_configs import DATASET_PATH
from utils.file_handler import dump, load


class LanguageModelingDataset(torchtext.data.Dataset):
    """Defines a dataset for language modeling."""

    def __init__(self, path, text_field, newline_eos=True,
                 encoding='utf-8', **kwargs):
        """Create a LanguageModelingDataset given a path and a field.

        Arguments:
            path: Path to the data file.
            text_field: The field that will be used for text data.
            newline_eos: Whether to add an <eos> token for every newline in the
                data file. Default: True.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        fields = [('text', text_field)]
        texts = []
        with io.open(path, encoding=encoding) as f:
            for line in f:
                preproceessed_line = text_field.preprocess(line)
                if newline_eos:
                    preproceessed_line.append(u'<eos>')
                texts.append(preproceessed_line)

        examples = [torchtext.data.Example.fromlist([t], fields) for t in texts]
        super(LanguageModelingDataset, self).__init__(
            examples, fields, **kwargs)


def load_real_dataset(dataset_name):
    train_filename, valid_filename, test_filename = \
        "{}_train.txt".format(dataset_name),\
        "{}_valid.txt".format(dataset_name),\
        "{}_test.txt".format(dataset_name)

    import random
    random.seed(42)
    print(train_filename, valid_filename, test_filename)

    TEXT = load(file_name=dataset_name+"_vocab.pkl", parent_path=DATASET_PATH)

    trn = LanguageModelingDataset(path=DATASET_PATH + train_filename,
                                  newline_eos=False, text_field=TEXT)

    vld = LanguageModelingDataset(path=DATASET_PATH + valid_filename,
                                  newline_eos=False, text_field=TEXT)

    tst = LanguageModelingDataset(path=DATASET_PATH + test_filename,
                                  newline_eos=False, text_field=TEXT)

    import revtok

    def denumericalize(batch):
        batch = [[TEXT.vocab.itos[ind] for ind in ex] for ex in batch.tolist()]

        def trim(s, t):
            sentence = []
            for w in s:
                if w == t:
                    break
                sentence.append(w)
            return sentence

        batch = [trim(ex, TEXT.eos_token) for ex in batch]  # trim past frst eos

        def filter_special(tok):
            return tok not in (TEXT.init_token, TEXT.pad_token)

        batch = [list(filter(filter_special, ex)) for ex in batch]

        return batch

    TEXT.detokenize = lambda B: [revtok.detokenize(l) for l in B]
    TEXT.denumericalize = denumericalize
    TEXT.fix_length = TEXT.max_length + 1

    lens = [len(x) for x in trn.text]

    print('vocab size: {}\ntrain size: {}\n valid size: {}\n test size: {}\n '
          'min length: {}\n max length: {}\n mean train length: {:.2f}, loaded max length: {}'
          .format(len(TEXT.vocab), len(trn), len(vld), len(tst),
                  np.min(lens), np.max(lens), np.mean(lens), TEXT.max_length))
    return trn, vld, tst, TEXT


def load_oracle_dataset():
    from metrics.oracle.oracle_lstm import Oracle_LSTM

    dataset_name = 'oracle'
    train_filename, valid_filename, test_filename = \
        "{}_train".format(dataset_name),\
        "{}_valid".format(dataset_name),\
        "{}_test".format(dataset_name)

    import random
    random.seed(42)
    print(train_filename, valid_filename, test_filename)

    TEXT = load(file_name=dataset_name+"_vocab.pkl", parent_path=DATASET_PATH)

    trn = LanguageModelingDataset(
        path=DATASET_PATH + train_filename + '.txt',
        newline_eos=False,
        text_field=TEXT)

    vld = LanguageModelingDataset(
        path=DATASET_PATH + valid_filename + '.txt',
        newline_eos=False,
        text_field=TEXT)

    tst = LanguageModelingDataset(
        path=DATASET_PATH + test_filename + '.txt',
        newline_eos=False,
        text_field=TEXT)

    print('vocab size: {}\ntrain size: {}\n valid size: {}\n test size: {}\n max length: {}'
          .format(len(TEXT.vocab), len(trn), len(vld), len(tst), TEXT.max_length))
    return trn, vld, tst, TEXT


if __name__ == '__main__':

    def test_real_dataset():
        DATASET_NAME = "amazon_app_book"
        train_ds, valid_ds, test_ds, TEXT = load_real_dataset(DATASET_NAME)

        tt = next(iter(test_ds.text))
        ttt = TEXT.numericalize([tt])
        print(tt)
        print(ttt)
        print(TEXT.reverse(ttt))
        import numpy as np

        tmp = []
        for x in train_ds.text:
            tmp.append(len(x))
        print("mean length: {}".format(np.mean(tmp)))
        print(next(train_ds.text))

    def test_oracle_dataset():
        train_ds, valid_ds, test_ds, TEXT = load_oracle_dataset()

        print(next(iter(test_ds.text)))
        import numpy as np

        tmp = []
        for x in train_ds.text:
            tmp.append(len(x))
        print("mean length: {}".format(np.mean(tmp)))

    # test_oracle_dataset()
    test_real_dataset()

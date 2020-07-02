import io
import torchtext
from torchtext.data import Field, ReversibleField, TabularDataset

from utils.path_configs import DATASET_PATH
from utils.file_handler import dump, load, write_text
from .data_manager import LanguageModelingDataset


def preprocess_real_dataset(dataset_name):
    train_filename, valid_filename, test_filename = "{}_train.txt".format(dataset_name),\
        "{}_valid.txt".format(dataset_name),\
        "{}_test.txt".format(dataset_name)

    import random
    random.seed(42)
    print(train_filename, valid_filename, test_filename)

    TEXT = ReversibleField(
        tokenize="revtok",
        tokenizer_language="en",
        init_token='<sos>',
        eos_token='<eos>',
        pad_token='<pad>',
        use_vocab=True,
        lower=True,
        batch_first=True,
        # fix_length=MAX_LENGTH
    )

    trn = LanguageModelingDataset(
        path=DATASET_PATH + train_filename,
        text_field=TEXT)

    vld = LanguageModelingDataset(
        path=DATASET_PATH + valid_filename,
        text_field=TEXT)

    tst = LanguageModelingDataset(
        path=DATASET_PATH + test_filename,
        text_field=TEXT)

    TEXT.build_vocab(trn, vld, tst)

    TEXT.max_length = max(
        max([len(t.text) for t in trn]),
        max([len(t.text) for t in vld]),
        max([len(t.text) for t in tst])
    ) + 1

    dump(obj=TEXT, file_name=dataset_name+"_vocab.pkl", parent_path=DATASET_PATH)

    print('vocab size: {}\ntrain size: {}\n valid size: {}\n test size: {}\n max length: {}'
          .format(len(TEXT.vocab), len(trn), len(vld), len(tst), TEXT.max_length))
    return trn, vld, tst, TEXT


def preprocess_oracle_dataset():
    from metrics.oracle.oracle_lstm import Oracle_LSTM

    dataset_name = 'oracle'
    train_filename, valid_filename, test_filename = "{}_train".format(dataset_name),\
        "{}_valid".format(dataset_name),\
        "{}_test".format(dataset_name)

    oracle = Oracle_LSTM(num_emb=5000, batch_size=128, emb_dim=3200,
                         hidden_dim=32, sequence_length=20)

    N = 60 * 10 ** 3
    N1 = int(N * 2/3)
    N2 = int(N * 1/6)

    samples = oracle.generate(N)
    samples = map(lambda xx: list(map(str, xx)), samples)
    samples = list(map(lambda x: " ".join(x), samples))

    train_samples = samples[:N1]
    valid_samples = samples[N1:N1 + N2]
    test_samples = samples[-N2:]

    write_text(train_samples, train_filename)
    write_text(valid_samples, valid_filename)
    write_text(test_samples, test_filename)

    import random
    random.seed(42)
    print(train_filename, valid_filename, test_filename)

    TEXT = ReversibleField(
        init_token='<sos>',
        use_vocab=True,
        lower=False,
        batch_first=True,
    )

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

    TEXT.build_vocab(trn, vld, tst)

    TEXT.max_length = max(
        max([len(t.text) for t in trn]),
        max([len(t.text) for t in vld]),
        max([len(t.text) for t in tst])
    )

    dump(obj=TEXT, file_name=dataset_name+"_vocab.pkl", parent_path=DATASET_PATH)

    print('vocab size: {}\ntrain size: {}\n valid size: {}\n test size: {}\n max length: {}'
          .format(len(TEXT.vocab), len(trn), len(vld), len(tst), TEXT.max_length))
    return trn, vld, tst, TEXT


if __name__ == '__main__':

    def test_real_dataset():
        DATASET_NAME = "coco"
        train_ds, valid_ds, test_ds, TEXT = preprocess_real_dataset(DATASET_NAME)

        print(next(iter(test_ds.text)))
        import numpy as np

        tmp = []
        for x in train_ds.text:
            tmp.append(len(x))
        print("mean length: {}".format(np.mean(tmp)))

    def test_oracle_dataset():
        train_ds, valid_ds, test_ds, TEXT = preprocess_oracle_dataset()

        print(next(iter(test_ds.text)))
        import numpy as np

        tmp = []
        for x in train_ds.text:
            tmp.append(len(x))
        print("mean length: {}".format(np.mean(tmp)))

    # test_oracle_dataset()
    test_real_dataset()

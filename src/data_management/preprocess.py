import numpy as np
from pandas import read_csv
from torchtext.data import ReversibleField

from utils.file_handler import dump, read_text, write_text
from utils.path_configs import DATASET_PATH

from data_management.data_manager import LanguageModelingDataset


def convert_csv_to_txt(dataset_name):
    train_csv_filename, test_csv_filename =\
        "{}{}_train.csv".format(DATASET_PATH, dataset_name),\
        "{}{}_test.csv".format(DATASET_PATH, dataset_name)

    train_txt_filename, test_txt_filename =\
        "{}_train_source".format(dataset_name),\
        "{}_test".format(dataset_name)

    train_df = read_csv(train_csv_filename)
    test_df = read_csv(test_csv_filename)

    train_lines = train_df['text']
    test_lines = test_df['text']

    write_text(train_lines, train_txt_filename)
    write_text(test_lines, test_txt_filename)


def split_train_into_train_valid(dataset_name):
    source_filename, train_filename, valid_filename, test_filename =\
        "{}_train_source".format(dataset_name),\
        "{}_train".format(dataset_name),\
        "{}_valid".format(dataset_name),\
        "{}_test".format(dataset_name)

    all_lines = read_text(source_filename)
    test_size = len(read_text(test_filename))

    valid_ids = set(np.random.choice(np.arange(len(all_lines)), test_size, replace=False))
    train_ids = [i for i in np.arange(len(all_lines)) if i not in valid_ids]
    valid_ids = list(valid_ids)

    train_lines = all_lines[train_ids]
    valid_lines = all_lines[valid_ids]

    write_text(train_lines, train_filename)
    write_text(valid_lines, valid_filename)


def preprocess_real_dataset(dataset_name):
    train_filename, valid_filename, test_filename =\
        "{}_train.txt".format(dataset_name),\
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

    trn = LanguageModelingDataset(path=DATASET_PATH + train_filename,
                                  newline_eos=False, text_field=TEXT)

    vld = LanguageModelingDataset(path=DATASET_PATH + valid_filename,
                                  newline_eos=False, text_field=TEXT)

    tst = LanguageModelingDataset(path=DATASET_PATH + test_filename,
                                  newline_eos=False, text_field=TEXT)

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
        # DATASET_NAME = "coco"
        DATASET_NAME = "news"
        # DATASET_NAME = "ptb"
        # DATASET_NAME = "amazon_app_book"
        # DATASET_NAME = "yelp_restaurant"

        train_ds, valid_ds, test_ds, TEXT = preprocess_real_dataset(DATASET_NAME)

        print('<sos>', TEXT.vocab.stoi['<sos>'])
        print('<eos>', TEXT.vocab.stoi['<eos>'])
        print('<pad>', TEXT.vocab.stoi['<pad>'])

        s = next(iter(test_ds.text))
        s_id = TEXT.numericalize([s])
        print(s)
        print(s_id)

        lens = [len(x) for x in train_ds.text]
        print("mean length: {}".format(np.mean(lens)))

    def test_oracle_dataset():
        train_ds, valid_ds, test_ds, TEXT = preprocess_oracle_dataset()

        print(next(iter(test_ds.text)))
        import numpy as np

        tmp = []
        for x in train_ds.text:
            tmp.append(len(x))
        print("mean length: {}".format(np.mean(tmp)))

    def convert_legacy_datasets():
        # DATASET_NAME = "amazon_app_book"
        DATASET_NAME = "yelp_restaurant"
        convert_csv_to_txt(DATASET_NAME)
        split_train_into_train_valid(DATASET_NAME)
        preprocess_real_dataset(DATASET_NAME)

    # test_oracle_dataset()
    test_real_dataset()
    # convert_legacy_datasets()

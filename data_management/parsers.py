from collections import Counter

import numpy as np

from utils.file_handler import PersistentClass
from utils.path_configs import DATASET_PATH


class Parser(PersistentClass):
    def __init__(self, lines, vocab_separator_function, vocab_separator_character, START_TOKEN='^', END_TOKEN='#',
                 name=''):
        super().__init__(concatenated_name=name)
        super().extend_savable_fields(
            ['vocab_separator_function',
             'vocab_separator_character',
             'START_TOKEN',
             'END_TOKEN',
             'vocab',
             'vocab2id',
             'id2vocab',
             'max_length'])
        self.name = name
        self.vocab_separator_character = vocab_separator_character
        self.vocab_separator_function = vocab_separator_function
        self.START_TOKEN = START_TOKEN
        self.END_TOKEN = END_TOKEN
        self.START_TOKEN_ID = None
        self.END_TOKEN_ID = None
        self.vocab = None
        self.vocab2id = None
        self.id2vocab = None
        self.max_length = None

        if self.load() == PersistentClass.FAILED_LOAD:
            assert lines is not None, 'delete corresponding data manager obj first!'
            lines = self._split_lines(lines)
            self._compute_max_length(lines)
            self._create_vocab(lines)
            self.save()

    def load(self):
        result = super().load()
        if result == PersistentClass.SUCCESSFUL_LOAD:
            self.START_TOKEN_ID = self.vocab2id[self.START_TOKEN]
            self.END_TOKEN_ID = self.vocab2id[self.END_TOKEN]
        return result

    def _split_lines(self, lines):
        if isinstance(lines, str):
            return self.vocab_separator_function(lines)
        return [self.vocab_separator_function(line) for line in lines]

    def _compute_max_length(self, lines):
        self.max_length = np.max([len(line) for line in lines]) + 2

    def _create_vocab(self, lines):
        vocab_counter = Counter()
        for line in lines:
            vocab_counter += Counter(line)
        del vocab_counter['']
        self.vocab = np.array(list(vocab_counter.keys()))
        np.random.shuffle(self.vocab)
        self.vocab = np.concatenate((self.vocab, [self.START_TOKEN, self.END_TOKEN]), axis=0)
        self.id2vocab = {str(i): v for i, v in enumerate(self.vocab)}
        # self.vocab2id = dict(zip(self.vocab, list(range(self.vocab.shape[0]))))
        self.vocab2id = {v: i for i, v in enumerate(self.vocab)}
        self.START_TOKEN_ID = self.vocab2id[self.START_TOKEN]
        self.END_TOKEN_ID = self.vocab2id[self.END_TOKEN]

    def line2id_format(self, lines, reverse=False):
        """
        converts string formatted line to id format
        :param lines: either str or list(str)
        :return: returns input in id format
        """

        def convert(line):
            vocabs = self._split_lines(line)[::-1] if reverse else self._split_lines(line)
            return [self.vocab2id[self.START_TOKEN]] + \
                   [self.vocab2id[v] for v in vocabs] + \
                   ([self.vocab2id[self.END_TOKEN]] * (self.max_length - (len(vocabs) + 1))), \
                   (len(vocabs) + 2)  # +1 for start token

        if isinstance(lines, str):
            id_formatted, lengths = convert(lines)
        else:
            id_formatted = []
            lengths = []
            for line in lines:
                formatted, l = convert(line)
                id_formatted.append(formatted)
                lengths.append(l)
        return id_formatted, lengths

    def id_format2line(self, id_format_lines, trim=False, reverse=False):
        """
        converts id format line to string format
        :param id_format_lines: either list(int) or list(list(int))
        :return: returns input in string format
        """

        def convert(id_format_line):
            line_slice = slice(0, len(id_format_line))
            start_token_loc = np.where(np.array(id_format_line) == self.START_TOKEN_ID)[0]
            end_token_loc = np.where(np.array(id_format_line) == self.END_TOKEN_ID)[0]
            if start_token_loc.shape[0] != 0:
                line_slice = slice(start_token_loc[-1] + 1, line_slice.stop)
            if end_token_loc.shape[0] != 0:
                line_slice = slice(line_slice.start, end_token_loc[0])
            line = list(id_format_line)  # making copy :D
            if reverse:
                line = line[:line_slice.start] + \
                       line[line_slice.start:line_slice.stop][::-1] + line[line_slice.stop:]
            if trim:
                line = line[line_slice]
            return self.vocab_separator_character.join([self.id2vocab[str(vocab_id)] for vocab_id in line])

        if not isinstance(id_format_lines[0], (np.int, np.int64, np.int32, int)):
            return [convert(l) for l in id_format_lines]
        return convert(id_format_lines)

    def tsv_export(self):
        mappings = ''
        for i, v in enumerate(p.id2vocab):
            mappings += v + '\n'
        with open(DATASET_PATH + self.name + '-mapping.tsv', 'w', encoding='utf8') \
                as file:
            file.write(mappings)


class Parser2(PersistentClass):
    def __init__(self, lines, vocab_separator_function, vocab_separator_character, max_len=None,
                 START_TOKEN='^', END_TOKEN='#', name=''):
        super().__init__(concatenated_name=name)
        super().extend_savable_fields(
            ['vocab_separator_function',
             'vocab_separator_character',
             'START_TOKEN',
             'END_TOKEN',
             'vocab',
             'vocab2id',
             'id2vocab',
             'max_length'])
        self.name = name
        self.vocab_separator_character = vocab_separator_character
        self.vocab_separator_function = vocab_separator_function
        self.START_TOKEN = START_TOKEN
        self.END_TOKEN = END_TOKEN
        self.START_TOKEN_ID = None
        self.END_TOKEN_ID = None
        self.vocab = None
        self.vocab2id = None
        self.id2vocab = None
        self.max_length = max_len

        if self.load() == PersistentClass.FAILED_LOAD:
            assert lines is not None, 'delete corresponding data manager obj first!'
            lines = self._split_lines(lines)
            self._compute_max_length(lines)
            self._create_vocab(lines)
            self.save()

    def load(self):
        result = super().load()
        if not isinstance(self, OracleBasedParser):
            if result == PersistentClass.SUCCESSFUL_LOAD:
                self.START_TOKEN_ID = self.vocab2id[self.START_TOKEN]
                self.END_TOKEN_ID = self.vocab2id[self.END_TOKEN]
        return result

    def _split_lines(self, lines):
        if isinstance(lines, str):
            return self.vocab_separator_function(lines)
        return [self.vocab_separator_function(line) for line in lines]

    def _compute_max_length(self, lines):
        if self.max_length is None:
            self.max_length = np.max([len(line) for line in lines]) + 1

    def _create_vocab(self, lines):
        vocab_counter = Counter()
        for line in lines:
            vocab_counter += Counter(line)
        del vocab_counter['']
        self.vocab = np.array(list(vocab_counter.keys()))
        np.random.shuffle(self.vocab)
        self.vocab = np.concatenate((self.vocab, [self.START_TOKEN, self.END_TOKEN]), axis=0)
        self.id2vocab = {str(i): v for i, v in enumerate(self.vocab)}
        self.vocab2id = {v: i for i, v in enumerate(self.vocab)}
        self.START_TOKEN_ID = self.vocab2id[self.START_TOKEN]
        self.END_TOKEN_ID = self.vocab2id[self.END_TOKEN]

    def line2id_format(self, lines, reverse=False):
        """
        converts string formatted line to id format
        :param lines: either str or list(str)
        :return: returns input in id format
        """

        def convert(line):
            if isinstance(line, str):
                vocabs = self._split_lines(line)[::-1] if reverse else self._split_lines(line)
            else:
                vocabs = line[::-1] if reverse else line
            crop_point = min(self.max_length, len(vocabs))
            return [self.vocab2id[v] for v in vocabs[:crop_point]] + \
                   ([self.vocab2id[self.END_TOKEN]] * (self.max_length - crop_point)), \
                   (self.max_length if self.max_length == crop_point else (crop_point + 1))  # +1 for end token

        if isinstance(lines, str) or isinstance(lines[0], str):
            id_formatted, lengths = convert(lines)
        else:
            id_formatted = []
            lengths = []
            for line in lines:
                formatted, l = convert(line)
                id_formatted.append(formatted)
                lengths.append(l)
        return id_formatted, lengths

    def id_format2line(self, id_format_lines, trim=True, reverse=False, merge=True):
        """
        converts id format line to string format
        :param id_format_lines: either list(int) or list(list(int))
        :return: returns input in string format
        """

        def convert(id_format_line):
            line_slice = slice(0, len(id_format_line))
            start_token_loc = np.where(np.array(id_format_line) == self.START_TOKEN_ID)[0]
            end_token_loc = np.where(np.array(id_format_line) == self.END_TOKEN_ID)[0]
            if start_token_loc.shape[0] != 0:
                line_slice = slice(start_token_loc[-1] + 1, line_slice.stop)
            if end_token_loc.shape[0] != 0:
                line_slice = slice(line_slice.start, end_token_loc[0])
            line = list(id_format_line)  # making copy :D
            if reverse:
                line = line[:line_slice.start] + \
                       line[line_slice.start:line_slice.stop][::-1] + line[line_slice.stop:]
            if trim:
                line = line[line_slice]
            assert self.START_TOKEN_ID not in line, 'start token id must not be in list!'
            line = [self.id2vocab[str(vocab_id)] for vocab_id in line]
            return self.vocab_separator_character.join(line) if merge else line

        if not isinstance(id_format_lines[0], (np.int, np.int64, np.int32, int)):
            return [convert(l) for l in id_format_lines]
        return convert(id_format_lines)


class CharacterBasedParser(Parser):
    def __init__(self, lines, name):
        super().__init__(lines, lambda x: [i for i in x], '', name=name)


class WordBasedParser(Parser2):
    def __init__(self, lines, name=''):
        import nltk
        super().__init__(lines, nltk.word_tokenize, ' ', name=name)


class OracleBasedParser(Parser2):
    def __init__(self, lines, max_len=None, name=''):
        super().__init__(lines, self.temp_separator, ' ', max_len=max_len, name=name)

    def temp_separator(self, x):
        return x.split(' ')

    def load(self):
        result = super().load()
        if result == PersistentClass.SUCCESSFUL_LOAD:
            self.START_TOKEN_ID = len(self.vocab)
            self.END_TOKEN_ID = len(self.vocab) + 1
        return result

    def _compute_max_length(self, lines):
        example_l = len(lines[0])
        assert not (False in [len(line) == example_l for line in lines]), \
            'all oracle samples must have same length'
        if self.max_length is None:
            self.max_length = example_l
        else:
            assert self.max_length <= example_l

    def _create_vocab(self, lines):
        vocab_counter = Counter()
        for line in lines:
            vocab_counter += Counter(line)
        del vocab_counter['']
        self.vocab = np.array(list(vocab_counter.keys()))
        self.id2vocab = {str(i): v for i, v in enumerate(self.vocab)}
        self.vocab2id = {v: i for i, v in enumerate(self.vocab)}
        self.START_TOKEN_ID = len(self.vocab)
        self.END_TOKEN_ID = len(self.vocab) + 1

    def line2id_format(self, lines, reverse=False):
        """
        converts string formatted line to id format
        :param lines: either str or list(str)
        :return: returns input in id format
        """

        def convert(line):
            if isinstance(line, str):
                vocabs = self._split_lines(line)[::-1] if reverse else self._split_lines(line)
            else:
                vocabs = line[::-1] if reverse else line
            assert self.max_length == len(vocabs), 'line should satisfy max len!'
            return [self.vocab2id[v] for v in vocabs], self.max_length

        if isinstance(lines, str) or isinstance(lines[0], str):
            id_formatted, lengths = convert(lines)
        else:
            id_formatted = []
            lengths = []
            for line in lines:
                formatted, l = convert(line)
                id_formatted.append(formatted)
                lengths.append(l)
        return id_formatted, lengths


if __name__ == '__main__':
    from utils.file_handler import read_text

    lines = read_text('coco-train')
    p = WordBasedParser(lines, 10, 'coco-words')

    print(p.id_format2line(p.line2id_format('he is gone gone gone ')[0], True))

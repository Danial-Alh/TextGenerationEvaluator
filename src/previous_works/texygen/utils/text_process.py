# coding=utf-8
import nltk


def chinese_process(filein, fileout):
    with open(filein, 'r') as infile:
        with open(fileout, 'w') as outfile:
            for line in infile:
                output = list()
                line = nltk.word_tokenize(line)[0]
                for char in line:
                    output.append(char)
                    output.append(' ')
                output.append('\n')
                output = ''.join(output)
                outfile.write(output)


def text_to_code(tokens, dictionary, seq_len):
    code_str = ""
    eof_code = len(dictionary)
    for sentence in tokens:
        index = 0
        for word in sentence:
            code_str += (str(dictionary[word]) + ' ')
            index += 1
        while index < seq_len:
            code_str += (str(eof_code) + ' ')
            index += 1
        code_str += '\n'
    return code_str


def code_to_text(codes, dictionary, out_file=None, eof_code=None):
    paras = []
    if eof_code is None:
        eof_code = len(dictionary)
    for line in codes:
        numbers = map(int, line)
        translated_line = ''
        for number in numbers:
            if number == eof_code:
                continue
            translated_line += (dictionary[number] + ' ')
        if translated_line.strip() is not '':
            paras.append(translated_line)
    paras = '\n'.join(paras)
    if out_file is not None:
        with open(out_file, 'w') as ofile:
            ofile.write(paras)
        print('codes converted to text format and written in file %s.' % out_file)
    return paras


def get_tokenlized(file):
    tokenlized = list()
    with open(file) as raw:
        for text in raw:
            text = nltk.word_tokenize(text.lower())
            tokenlized.append(text)
    return tokenlized


def get_word_list(tokens):
    word_set = list()
    for sentence in tokens:
        for word in sentence:
            word_set.append(word)
    return list(set(word_set))


def get_dict(word_set):
    word_index_dict = dict()
    index_word_dict = dict()
    index = 0
    for word in word_set:
        word_index_dict[word] = str(index)
        index_word_dict[str(index)] = word
        index += 1
    return word_index_dict, index_word_dict


def text_precess(train_text_loc, test_text_loc=None):
    train_tokens = get_tokenlized(train_text_loc)
    if test_text_loc is None:
        test_tokens = list()
    else:
        test_tokens = get_tokenlized(test_text_loc)
    word_set = get_word_list(train_tokens + test_tokens)
    [word_index_dict, index_word_dict] = get_dict(word_set)

    if test_text_loc is None:
        sequence_len = len(max(train_tokens, key=len))
    else:
        sequence_len = max(len(max(train_tokens, key=len)), len(max(test_tokens, key=len)))
    # with open('save/eval_data.txt', 'w') as outfile:
    #     outfile.write(text_to_code(test_tokens, word_index_dict, sequence_len))

    return sequence_len, len(word_index_dict) + 1


def load_or_create_dictionary(tokens, path, name):
    from ..utils.utils import load_obj, save_obj
    loaded_obj = load_obj(path, name)
    if loaded_obj is None:
        loaded_obj = get_dict(get_word_list(tokens))
        save_obj(loaded_obj, path, name)
    return loaded_obj

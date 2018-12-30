import codecs
import os
import pickle

import numpy as np

from utils.file_handler import write_text
from metrics.oracle.target_lstm import TARGET_LSTM


class Oracle_LSTM(TARGET_LSTM):
    def __init__(self, num_emb=5000, batch_size=128, emb_dim=3200, hidden_dim=32, sequence_length=20, start_token=0,
                 params=None,
                 sess_config=None):
        import tensorflow as tf
        if params is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            full_path = os.path.join(current_dir, "target_params_py3.pkl")
            with codecs.open(full_path, "rb") as f:
                params = pickle.load(f)
        self.graph_obj = tf.Graph()
        with self.graph_obj.as_default():
            super().__init__(num_emb, batch_size, emb_dim, hidden_dim, sequence_length, start_token, params)
            if sess_config is None:
                sess_config = tf.ConfigProto()
            sess_config.gpu_options.allow_growth = True
            self.sess = tf.Session(graph=self.graph_obj, config=sess_config)
            self.sess.run(tf.global_variables_initializer())

    def generate(self, number=None):
        if number is None:
            number = self.batch_size
        generated_number = 0
        tmp = []
        while generated_number < number:
            tmp.append(super().generate(self.sess))
            generated_number += tmp[-1].shape[0]
        return np.concatenate(tmp, axis=0)[:number, :]

    def log_probability(self, inp):
        if type(inp) is np.ndarray:
            assert len(inp.shape) == 2 and inp.shape[1] == self.sequence_length
        elif type(inp) is list:
            inp = np.array(inp)
            assert len(inp.shape) == 2 and inp.shape[1] == self.sequence_length
        else:
            raise ValueError

        inp_len = inp.shape[0]
        res = np.ones(inp_len) * np.nan

        new_inp = np.concatenate((inp, np.zeros(((-1 * inp_len) % self.batch_size, self.sequence_length))), axis=0)
        assert new_inp.shape[0] % self.batch_size == 0, new_inp.shape
        for start_inx in range(0, new_inp.shape[0], self.batch_size):
            end_inx = start_inx + self.batch_size
            tmp = self.sess.run(self.out_loss, {self.x: new_inp[start_inx:end_inx]})
            res[start_inx:min(end_inx, inp_len)] = tmp[:min(end_inx, inp_len) - start_inx]
        print(res.shape)
        return -1. * res


if __name__ == "__main__":
    oracle = Oracle_LSTM(batch_size=64)
    x = oracle.generate(37500)
    x = [" ".join([str(xxx) for xxx in xx]) for xx in x]
    write_text(x, 'oracle37.5-train')

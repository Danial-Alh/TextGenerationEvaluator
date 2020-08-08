import numpy as np

from ...utils.metrics.Metrics import Metrics


class Nll(Metrics):
    def __init__(self, data_loader, rnn, sess, temperature={'value': None}):
        super().__init__()
        self.name = 'nll-oracle'
        self.data_loader = data_loader
        self.sess = sess
        self.rnn = rnn
        self.temperature = temperature

    def set_name(self, name):
        self.name = name

    def get_name(self):
        return self.name

    def get_score(self):
        return self.nll_loss()

    def nll_loss(self):
        nll = []
        self.data_loader.reset_pointer()
        for it in range(self.data_loader.num_batch):
            batch = self.data_loader.next_batch()
            # fixme bad taste
            # try:
            #     g_loss = self.rnn.get_nll(self.sess, batch)
            # except Exception as e:
            if self.temperature['value'] is None:
                g_loss = self.sess.run(self.rnn.pretrain_loss, {self.rnn.x: batch})
            elif self.temperature['type'] == 'biased':
                g_loss = self.sess.run(self.rnn.temp_pretrain_loss, {self.rnn.x: batch,
                                                                     self.rnn.temperature: self.temperature['value']})
            elif self.temperature['type'] == 'unbiased':
                g_loss = self.sess.run(self.rnn.unbiased_temperature_persample_ll,
                                       {self.rnn.dynamic_batch_x: batch,
                                        self.rnn.unbiased_temperature: self.temperature['value']})
                # g_loss = np.sum(g_loss) / (self.rnn.sequence_length * self.rnn.batch_size)
                g_loss = np.sum(g_loss) / (self.rnn.batch_size)
            nll.append(g_loss)
        print("**************** ohhhhhhhhhhhhhh here nll shape ------> " + str(np.array(nll).shape))
        return np.mean(nll)

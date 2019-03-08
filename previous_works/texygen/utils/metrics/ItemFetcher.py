import numpy as np

from ...utils.metrics.Metrics import Metrics


class ItemFetcher(Metrics):
    def __init__(self, data_loader, rnn, item_to_be_fetched, sess, temperature):
        super().__init__()
        self.name = 'nll-oracle'
        self.data_loader = data_loader
        self.rnn = rnn
        self.sess = sess
        self.item_to_be_fetched = item_to_be_fetched
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
            if self.temperature is None:
                g_loss = self.sess.run(self.item_to_be_fetched, {self.rnn.x: batch})
            else:
                g_loss = self.sess.run(self.item_to_be_fetched, {self.rnn.x: batch,
                                                                 self.rnn.temperature: self.temperature})
            nll.extend(g_loss)
        return nll

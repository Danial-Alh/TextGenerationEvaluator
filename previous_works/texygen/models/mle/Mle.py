from ...models.Gan import Gan
from ...models.mle.MleDataLoader import DataLoader
from ...models.mle.MleGenerator import Generator
from ...utils.metrics.Nll import Nll
from ...utils.oracle.OracleLstm import OracleLstm
from ...utils.text_process import load_or_create_dictionary
from ...utils.utils import *


class Mle(Gan):
    from ...models.mle import SAVING_PATH
    saving_path = SAVING_PATH
    oracle_file = saving_path + 'oracle.txt'
    generator_file = saving_path + 'generator.txt'
    test_file = saving_path + 'test_file.txt'

    def __init__(self, oracle=None):
        super().__init__()
        # you can change parameters, generator here
        self.vocab_size = 20
        self.emb_dim = 32
        self.hidden_dim = 32
        self.sequence_length = 20
        self.filter_size = [2, 3]
        self.num_filters = [100, 200]
        self.l2_reg_lambda = 0.2
        self.dropout_keep_prob = 0.75
        self.batch_size = 64
        self.generate_num = 128
        self.start_token = 0

    def init_oracle_trainng(self, oracle=None):
        if oracle is None:
            oracle = OracleLstm(num_vocabulary=self.vocab_size, batch_size=self.batch_size, emb_dim=self.emb_dim,
                                hidden_dim=self.hidden_dim, sequence_length=self.sequence_length,
                                start_token=self.start_token)
        self.set_oracle(oracle)

        generator = Generator(num_vocabulary=self.vocab_size, batch_size=self.batch_size, emb_dim=self.emb_dim,
                              hidden_dim=self.hidden_dim, sequence_length=self.sequence_length,
                              start_token=self.start_token)
        self.set_generator(generator)

        gen_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length)
        oracle_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length)
        dis_dataloader = None

        self.set_data_loader(gen_loader=gen_dataloader, dis_loader=dis_dataloader, oracle_loader=oracle_dataloader)

    def init_metric(self):
        nll = Nll(data_loader=self.oracle_data_loader, rnn=self.oracle, sess=self.sess)
        self.add_metric(nll)

        inll = Nll(data_loader=self.gen_data_loader, rnn=self.generator, sess=self.sess)
        inll.set_name('nll-test')
        self.add_metric(inll)

        # from ... utils.metrics.DocEmbSim import DocEmbSim
        # docsim = DocEmbSim(oracle_file=self.oracle_file, generator_file=self.generator_file,
        #                    num_vocabulary=self.vocab_size)
        # self.add_metric(docsim)

    def train_discriminator(self):
        generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
        self.dis_data_loader.load_train_data(self.oracle_file, self.generator_file)
        for _ in range(3):
            self.dis_data_loader.next_batch()
            x_batch, y_batch = self.dis_data_loader.next_batch()
            feed = {
                self.discriminator.input_x: x_batch,
                self.discriminator.input_y: y_batch,
            }
            _ = self.sess.run(self.discriminator.train_op, feed)
            self.save_discriminator()

    def evaluate(self):
        generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
        if self.oracle_data_loader is not None:
            self.oracle_data_loader.create_batches(self.generator_file)
        if self.log is not None:
            if self.epoch == 0 or self.epoch == 1:
                for metric in self.metrics:
                    self.log.write(metric.get_name() + ',')
                self.log.write('\n')
            scores = super().evaluate()
            for score in scores:
                self.log.write(str(score) + ',')
            self.log.write('\n')
            return scores
        return super().evaluate()

    def train_oracle(self):
        self.init_oracle_trainng()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        self.load_generator_discriminator()

        self.pre_epoch_num = 80
        self.log = open('experiment-log-mle.csv', 'w')
        generate_samples(self.sess, self.oracle, self.batch_size, self.generate_num, self.oracle_file)
        generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
        self.gen_data_loader.create_batches(self.oracle_file)
        self.oracle_data_loader.create_batches(self.generator_file)
        self.init_metric()

        print('start pre-train generator:')
        for epoch in range(self.pre_epoch_num):
            start = time()
            loss = pre_train_epoch(self.sess, self.generator, self.gen_data_loader)
            end = time()
            print('epoch:' + str(self.epoch) + '\t time:' + str(end - start))
            self.add_epoch()
            if epoch % 5 == 0:
                self.evaluate()
        generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
        return

    def init_real_trainng(self, data_loc=None, parser=None):
        from ...utils.text_process import text_precess, text_to_code
        from ...utils.text_process import get_tokenlized
        tokens = get_tokenlized(data_loc)
        if parser is not None:
            self.sequence_length, self.vocab_size = parser.max_length, parser.vocab.shape[0]
            word_index_dict, index_word_dict = parser.vocab2id, parser.id2vocab,
            self.start_token, self.end_token = parser.START_TOKEN_ID, parser.END_TOKEN_ID
            import shutil
            shutil.copy(data_loc, self.oracle_file)
            print('parser set from outside!')
        else:
            self.start_token, self.end_token = 0, 0
            self.sequence_length, self.vocab_size = text_precess(data_loc)
            dictionaries_name = data_loc.split('/')[-1].split('.')[0] + '_dictionaries'
            [word_index_dict, index_word_dict] = load_or_create_dictionary(tokens, self.saving_path, dictionaries_name)
            with open(self.oracle_file, 'w') as outfile:
                outfile.write(text_to_code(tokens, word_index_dict, self.sequence_length))
        generator = Generator(num_vocabulary=self.vocab_size, batch_size=self.batch_size, emb_dim=self.emb_dim,
                              hidden_dim=self.hidden_dim, sequence_length=self.sequence_length,
                              start_token=self.start_token)
        self.set_generator(generator)

        gen_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length,
                                    end_token=self.end_token)
        oracle_dataloader = None
        dis_dataloader = None

        self.set_data_loader(gen_loader=gen_dataloader, dis_loader=dis_dataloader, oracle_loader=oracle_dataloader)
        return word_index_dict, index_word_dict

    def train_real(self, data_loc=None, wrapper_ref=None):
        from ...utils.text_process import code_to_text
        from ...utils.text_process import get_tokenlized
        if wrapper_ref is None:
            wi_dict, iw_dict = self.init_real_trainng(data_loc)
        else:
            self.wrapper = wrapper_ref
            wi_dict, iw_dict = self.wrapper.parser.vocab2id, self.wrapper.parser.id2vocab
        self.init_real_metric()

        def get_real_test_file(dict=iw_dict):
            with open(self.generator_file, 'r') as file:
                codes = get_tokenlized(self.generator_file)
            with open(self.test_file, 'w') as outfile:
                outfile.write(code_to_text(codes=codes, dictionary=dict, eof_code=self.wrapper.parser.END_TOKEN_ID))

        self.pre_epoch_num = 80
        self.adversarial_epoch_num = 100
        self.log = open('experiment-log-mle-real.csv', 'w')
        generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
        self.gen_data_loader.create_batches(self.oracle_file)

        print('start pre-train generator:')
        for epoch in range(self.pre_epoch_num):
            start = time()
            loss = pre_train_epoch(self.sess, self.generator, self.gen_data_loader)
            end = time()
            print('epoch:' + str(self.epoch) + '\t time:' + str(end - start))
            self.add_epoch()
            if epoch % 5 == 0:
                generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
                get_real_test_file()
                self.evaluate()
        generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)

import json

from ...models.Gan import Gan
from ...models.seqgan.SeqganDataLoader import DataLoader, DisDataloader
from ...models.seqgan.SeqganDiscriminator import Discriminator
from ...models.seqgan.SeqganGenerator import Generator
from ...models.seqgan.SeqganReward import Reward
from ...utils.metrics.Cfg import Cfg
from ...utils.metrics.Nll import Nll
from ...utils.oracle.OracleCfg import OracleCfg
from ...utils.oracle.OracleLstm import OracleLstm
from ...utils.text_process import *
from ...utils.utils import *


class Seqgan(Gan):
    from ...models.seqgan import SAVING_PATH
    saving_path = SAVING_PATH
    oracle_file = saving_path + 'oracle.txt'
    generator_file = saving_path + 'generator.txt'
    test_file = saving_path + 'test_file.txt'

    def __init__(self, oracle=None):
        super().__init__()
        # you can change parameters, generator here
        self.vocab_size = 20
        self.emb_dim = 128
        self.hidden_dim = 64
        self.sequence_length = 20
        self.filter_size = [2, 3]
        self.num_filters = [100, 200]
        self.l2_reg_lambda = 0.2
        self.dropout_keep_prob = 0.75
        self.batch_size = 64
        self.generate_num = 128
        self.start_token = 0

    def init_metric(self):
        nll = Nll(data_loader=self.oracle_data_loader, rnn=self.oracle, sess=self.sess)
        self.add_metric(nll)

        inll = Nll(data_loader=self.gen_data_loader, rnn=self.generator, sess=self.sess)
        inll.set_name('nll-test')
        self.add_metric(inll)

        from ...utils.metrics.DocEmbSim import DocEmbSim
        docsim = DocEmbSim(oracle_file=self.oracle_file, generator_file=self.generator_file,
                           num_vocabulary=self.vocab_size)
        self.add_metric(docsim)

    def train_discriminator(self):
        generate_samples(self.sess, self.generator, self.batch_size, len(self.wrapper.train_samples), self.generator_file)
        self.dis_data_loader.load_train_data(self.oracle_file, self.generator_file)
        for i in range(self.dis_data_loader.num_batch):
            self.dis_data_loader.next_batch()
            x_batch, y_batch = self.dis_data_loader.next_batch()
            feed = {
                self.discriminator.input_x: x_batch,
                self.discriminator.input_y: y_batch,
            }
            loss, _ = self.sess.run([self.discriminator.d_loss, self.discriminator.train_op], feed)
            self.save_discriminator()
            if i % (self.dis_data_loader.num_batch // 5) == 0:
                print(loss)

    def evaluate(self):
        generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
        if self.oracle_data_loader is not None:
            self.oracle_data_loader.create_batches(self.generator_file)
        if self.log is not None:
            if self.epoch == 0 or self.epoch == 1:
                self.log.write('epochs, ')
                for metric in self.metrics:
                    self.log.write(metric.get_name() + ',')
                self.log.write('\n')
            scores = super().evaluate()
            for score in scores:
                self.log.write(str(score) + ',')
            self.log.write('\n')
            return scores
        return super().evaluate()

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

        discriminator = Discriminator(sequence_length=self.sequence_length, num_classes=2, vocab_size=self.vocab_size,
                                      emd_dim=self.emb_dim, filter_sizes=self.filter_size, num_filters=self.num_filters,
                                      l2_reg_lambda=self.l2_reg_lambda)
        self.set_discriminator(discriminator)

        gen_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length)
        oracle_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length)
        dis_dataloader = DisDataloader(batch_size=self.batch_size, seq_length=self.sequence_length)

        self.set_data_loader(gen_loader=gen_dataloader, dis_loader=dis_dataloader, oracle_loader=oracle_dataloader)

    def train_oracle(self):
        self.init_oracle_trainng()
        self.init_metric()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        self.load_generator_discriminator()

        self.pre_epoch_num = 80
        self.adversarial_epoch_num = 100
        self.log = open('experiment-log-seqgan.csv', 'w')
        # disable to prevent new oracle generation
        generate_samples(self.sess, self.oracle, self.batch_size, self.generate_num, self.oracle_file)
        generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
        self.gen_data_loader.create_batches(self.oracle_file)
        self.oracle_data_loader.create_batches(self.generator_file)

        print('start pre-train generator:')
        for epoch in range(self.pre_epoch_num):
            start = time()
            loss = pre_train_epoch(self.sess, self.generator, self.gen_data_loader)
            end = time()
            print('epoch:' + str(self.epoch) + '\t time:' + str(end - start))
            self.add_epoch()
            if epoch % 5 == 0:
                generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
                self.evaluate()

        print('start pre-train discriminator:')
        self.reset_epoch()
        for epoch in range(self.pre_epoch_num):
            print('epoch:' + str(epoch))
            self.train_discriminator()

        self.reset_epoch()
        print('adversarial training:')
        self.reward = Reward(self.generator, .8)
        for epoch in range(self.adversarial_epoch_num):
            # print('epoch:' + str(epoch))
            start = time()
            for index in range(1):
                samples = self.generator.generate(self.sess)
                rewards = self.reward.get_reward(self.sess, samples, 16, self.discriminator)
                feed = {
                    self.generator.x: samples,
                    self.generator.rewards: rewards
                }
                _ = self.sess.run(self.generator.g_updates, feed_dict=feed)
                self.save_generator()
            end = time()
            self.add_epoch()
            print('epoch:' + str(self.epoch) + '\t time:' + str(end - start))
            if epoch % 5 == 0 or epoch == self.adversarial_epoch_num - 1:
                generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
                self.evaluate()

            self.reward.update_params()
            for _ in range(15):
                self.train_discriminator()

    def init_cfg_training(self, grammar=None):
        oracle = OracleCfg(sequence_length=self.sequence_length, cfg_grammar=grammar)
        self.set_oracle(oracle)
        self.oracle.generate_oracle()
        self.vocab_size = self.oracle.vocab_size + 1

        generator = Generator(num_vocabulary=self.vocab_size, batch_size=self.batch_size, emb_dim=self.emb_dim,
                              hidden_dim=self.hidden_dim, sequence_length=self.sequence_length,
                              start_token=self.start_token)
        self.set_generator(generator)

        discriminator = Discriminator(sequence_length=self.sequence_length, num_classes=2, vocab_size=self.vocab_size,
                                      emd_dim=self.emb_dim, filter_sizes=self.filter_size, num_filters=self.num_filters,
                                      l2_reg_lambda=self.l2_reg_lambda)
        self.set_discriminator(discriminator)

        gen_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length)
        oracle_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length)
        dis_dataloader = DisDataloader(batch_size=self.batch_size, seq_length=self.sequence_length)
        self.set_data_loader(gen_loader=gen_dataloader, dis_loader=dis_dataloader, oracle_loader=oracle_dataloader)
        return oracle.wi_dict, oracle.iw_dict

    def init_cfg_metric(self, grammar=None):
        cfg = Cfg(test_file=self.test_file, cfg_grammar=grammar)
        self.add_metric(cfg)

    def train_cfg(self):
        cfg_grammar = """
          S -> S PLUS x | S SUB x |  S PROD x | S DIV x | x | '(' S ')'
          PLUS -> '+'
          SUB -> '-'
          PROD -> '*'
          DIV -> '/'
          x -> 'x' | 'y'
        """

        wi_dict_loc, iw_dict_loc = self.init_cfg_training(cfg_grammar)
        with open(iw_dict_loc, 'r') as file:
            iw_dict = json.load(file)

        def get_cfg_test_file(dict=iw_dict):
            with open(self.generator_file, 'r') as file:
                codes = get_tokenlized(self.generator_file)
            with open(self.test_file, 'w') as outfile:
                outfile.write(code_to_text(codes=codes, dictionary=dict))

        self.init_cfg_metric(grammar=cfg_grammar)
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        self.load_generator_discriminator()

        self.pre_epoch_num = 80
        self.adversarial_epoch_num = 100
        self.log = open('experiment-log-seqgan-cfg.csv', 'w')
        generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
        self.gen_data_loader.create_batches(self.oracle_file)
        self.oracle_data_loader.create_batches(self.generator_file)
        print('start pre-train generator:')
        for epoch in range(self.pre_epoch_num):
            start = time()
            loss = pre_train_epoch(self.sess, self.generator, self.gen_data_loader)
            end = time()
            print('epoch:' + str(self.epoch) + '\t time:' + str(end - start))
            self.add_epoch()
            if epoch % 5 == 0:
                generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
                get_cfg_test_file()
                self.evaluate()

        print('start pre-train discriminator:')
        self.reset_epoch()
        for epoch in range(self.pre_epoch_num * 3):
            print('epoch:' + str(epoch))
            self.train_discriminator()

        self.reset_epoch()
        print('adversarial training:')
        self.reward = Reward(self.generator, .8)
        for epoch in range(self.adversarial_epoch_num):
            # print('epoch:' + str(epoch))
            start = time()
            for index in range(1):
                samples = self.generator.generate(self.sess)
                rewards = self.reward.get_reward(self.sess, samples, 16, self.discriminator)
                feed = {
                    self.generator.x: samples,
                    self.generator.rewards: rewards
                }
                loss, _ = self.sess.run([self.generator.g_loss, self.generator.g_updates], feed_dict=feed)
                self.save_generator()
                print(loss)
            end = time()
            self.add_epoch()
            print('epoch:' + str(self.epoch) + '\t time:' + str(end - start))
            if epoch % 5 == 0 or epoch == self.adversarial_epoch_num - 1:
                generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
                get_cfg_test_file()
                self.evaluate()

            self.reward.update_params()
            for _ in range(15):
                self.train_discriminator()
        return

    def init_real_trainng(self, parser=None):
        assert parser is not None
        self.sequence_length, self.vocab_size = parser.max_length, len(parser.vocab)
        word_index_dict, index_word_dict = parser.vocab.stoi, parser.vocab.itos,
        self.start_token, self.end_token, self.pad_token = parser.vocab.stoi[parser.init_token],\
            parser.vocab.stoi[parser.eos_token], parser.vocab.stoi[parser.pad_token]
        print('parser set from outside!')

        generator = Generator(num_vocabulary=self.vocab_size, batch_size=self.batch_size, emb_dim=self.emb_dim,
                              hidden_dim=self.hidden_dim, sequence_length=self.sequence_length,
                              start_token=self.start_token, pad_token=self.pad_token)
        self.set_generator(generator)

        discriminator = Discriminator(sequence_length=self.sequence_length, num_classes=2, vocab_size=self.vocab_size,
                                      emd_dim=self.emb_dim, filter_sizes=self.filter_size, num_filters=self.num_filters,
                                      l2_reg_lambda=self.l2_reg_lambda)
        self.set_discriminator(discriminator)

        gen_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length,
                                    end_token=self.end_token, pad_token=self.pad_token)
        oracle_dataloader = None
        dis_dataloader = DisDataloader(batch_size=self.batch_size, seq_length=self.sequence_length)

        self.set_data_loader(gen_loader=gen_dataloader, dis_loader=dis_dataloader, oracle_loader=oracle_dataloader)
        return word_index_dict, index_word_dict

    def train_real(self, data_loc=None, wrapper_ref=None):
        from ...utils.text_process import code_to_text
        from ...utils.text_process import get_tokenlized
        assert wrapper_ref is not None
        self.wrapper = wrapper_ref
        import shutil
        shutil.copy(data_loc, self.oracle_file)
        wi_dict, iw_dict = self.wrapper.parser.vocab.stoi, self.wrapper.parser.vocab.itos
        self.init_real_metric()

        def get_real_test_file(dict=iw_dict):
            with open(self.generator_file, 'r') as file:
                codes = get_tokenlized(self.generator_file)
            with open(self.test_file, 'w') as outfile:
                outfile.write(code_to_text(codes=codes, dictionary=dict, eof_code=self.end_token))

        self.pre_epoch_num = 80
        self.adversarial_epoch_num = 200
        self.log = open('experiment-log-seqgan-real.csv', 'w')
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

        print('start pre-train discriminator:')
        self.reset_epoch()
        for epoch in range(self.pre_epoch_num):
            print('epoch:' + str(epoch))
            self.train_discriminator()

        self.reset_epoch()
        print('adversarial training:')
        self.reward = Reward(self.generator, .8)
        for epoch in range(self.adversarial_epoch_num):
            # print('epoch:' + str(epoch))
            start = time()
            for index in range(1):
                samples = self.generator.generate(self.sess)
                rewards = self.reward.get_reward(self.sess, samples, 16, self.discriminator)
                feed = {
                    self.generator.x: samples,
                    self.generator.rewards: rewards
                }
                loss, _ = self.sess.run([self.generator.g_loss, self.generator.g_updates], feed_dict=feed)
                print(loss)
                self.save_generator()
            end = time()
            self.add_epoch()
            print('epoch:' + str(self.epoch) + '\t time:' + str(end - start))
            if epoch % 5 == 0 or epoch == self.adversarial_epoch_num - 1:
                generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
                get_real_test_file()
                self.evaluate()

            self.reward.update_params()
            for _ in range(1):
                self.train_discriminator()

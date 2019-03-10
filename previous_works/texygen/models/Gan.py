from abc import abstractmethod

import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops, control_flow_ops

from ..utils.utils import init_sess, generate_samples


class Gan:
    def __init__(self):
        self.oracle = None
        self.generator = None
        self.discriminator = None
        self.gen_data_loader = None
        self.dis_data_loader = None
        self.oracle_data_loader = None
        self.sess = init_sess()
        self.metrics = list()
        self.epoch = 0
        self.pre_epoch_num = 80
        self.adversarial_epoch_num = 100
        self.log = None
        self.reward = None

    def set_oracle(self, oracle):
        self.oracle = oracle

    def set_generator(self, generator):
        self.generator = generator

    def set_discriminator(self, discriminator):
        self.discriminator = discriminator

    def set_data_loader(self, gen_loader, dis_loader, oracle_loader):
        self.gen_data_loader = gen_loader
        self.dis_data_loader = dis_loader
        self.oracle_data_loader = oracle_loader

    def set_sess(self, sess):
        self.sess = sess

    def add_metric(self, metric):
        self.metrics.append(metric)

    def add_epoch(self):
        self.epoch += 1

    def reset_epoch(self):
        # current not in use
        return
        self.epoch = 0

    def evaluate(self):
        from time import time
        log = "epoch:" + str(self.epoch) + '\t'
        scores = list()
        scores.append(self.epoch)
        for metric in self.metrics:
            tic = time()
            score = metric.get_score()
            log += metric.get_name() + ":" + str(score) + '\t'
            toc = time()
            print('time elapsed of ' + metric.get_name() + ': ' + str(toc - tic))
            scores.append(score)
        print(log)
        self.wrapper.update_scores(self.epoch)
        return scores

    def init_real_metric(self):
        # from .. utils.metrics.DocEmbSim import DocEmbSim
        # docsim = DocEmbSim(oracle_file=self.oracle_file, generator_file=self.generator_file,
        #                    num_vocabulary=self.vocab_size)
        # self.add_metric(docsim)
        from ..utils.metrics.Nll import Nll

        inll = Nll(data_loader=self.gen_data_loader, rnn=self.generator, sess=self.sess)
        inll.set_name('nll-test')
        self.add_metric(inll)

    def load_generator_discriminator(self, name=''):
        if self.generator is not None:
            self.generator.load_model(self.sess, name)
        if self.discriminator is not None:
            self.discriminator.load_model(self.sess, name)

    def save_generator(self, name=''):
        self.generator.save_model(self.sess, name)

    def save_discriminator(self, name=''):
        self.discriminator.save_model(self.sess, name)

    def check_valid(self):
        # TODO
        pass

    @abstractmethod
    def train_oracle(self):
        pass

    def train_cfg(self):
        pass

    def train_real(self):
        pass

    def generate_sample2(self, data_loc=None, n_samples=2000):
        [word_index_dict, index_word_dict] = self.init_real_trainng(data_loc)
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        self.load_generator_discriminator()

        print('generating {} samples from {}!'.format(n_samples, self.__class__.__name__))
        if self.__class__.__name__ == 'Leakgan':
            from ..models.leakgan import Leakgan
            codes = Leakgan.generate_samples_gen(self.sess, self.generator, 64, n_samples, self.test_file)
        else:
            codes = generate_samples(self.sess, self.generator, 64, n_samples, self.test_file)
        print('samples generated!')
        from ..utils.text_process import code_to_text
        code_to_text(codes, index_word_dict, self.test_file)


class GeneralGenerator(object):
    def __init__(self):
        self.fork_degree = 64
        self.naive_temperature_init()
        self.unbiased_temperature_init()

    def naive_temperature_init(self):
        t_gen_o = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length,
                                               dynamic_size=False, infer_shape=True)
        t_gen_x = tensor_array_ops.TensorArray(dtype=tf.int32, size=self.sequence_length,
                                               dynamic_size=False, infer_shape=True)
        self.temperature = tf.placeholder(tf.float32)

        def _temperature_g_recurrence(i, x_t, h_tm1, gen_o, gen_x):
            h_t = self.g_recurrent_unit(x_t, h_tm1)  # hidden_memory_tuple
            o_t = self.g_output_unit(h_t)  # batch x vocab , logits not prob
            o_t *= self.temperature
            next_token = tf.cast(tf.reshape(tf.multinomial(o_t, 1), [self.batch_size]), tf.int32)
            x_tp1 = tf.nn.embedding_lookup(self.g_embeddings, next_token)  # batch x emb_dim
            gen_o = gen_o.write(i, tf.reduce_sum(tf.multiply(tf.one_hot(next_token, self.num_vocabulary, 1.0, 0.0),
                                                             tf.nn.softmax(o_t)), 1))  # [batch_size] , prob
            gen_x = gen_x.write(i, next_token)  # indices, batch_size
            return i + 1, x_tp1, h_t, gen_o, gen_x

        _, _, _, self.temp_gen_o, self.temp_gen_x = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3, _4: i < self.sequence_length,
            body=_temperature_g_recurrence,
            loop_vars=(tf.constant(0, dtype=tf.int32),
                       tf.nn.embedding_lookup(self.g_embeddings, self.start_token), self.h0, t_gen_o, t_gen_x))

        self.temp_gen_x = self.temp_gen_x.stack()  # seq_length x batch_size
        self.temp_gen_x = tf.transpose(self.temp_gen_x, perm=[1, 0])  # batch_size x seq_length

        ####################################################################

        g_predictions = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.sequence_length,
            dynamic_size=False, infer_shape=True)

        ta_emb_x = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.sequence_length)
        ta_emb_x = ta_emb_x.unstack(self.processed_x)

        def _pretrain_recurrence(i, x_t, h_tm1, g_predictions):
            h_t = self.g_recurrent_unit(x_t, h_tm1)
            o_t = self.g_output_unit(h_t)
            o_t *= self.temperature
            g_predictions = g_predictions.write(i, tf.nn.softmax(o_t))  # batch x vocab_size
            x_tp1 = ta_emb_x.read(i)
            return i + 1, x_tp1, h_t, g_predictions

        _, _, _, self.temp_g_predictions = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3: i < self.sequence_length,
            body=_pretrain_recurrence,
            loop_vars=(tf.constant(0, dtype=tf.int32),
                       tf.nn.embedding_lookup(self.g_embeddings, self.start_token),
                       self.h0, g_predictions))

        self.temp_g_predictions = tf.transpose(self.temp_g_predictions.stack(),
                                               perm=[1, 0, 2])  # batch_size x seq_length x vocab_size

        self.temp_pretrain_loss = -tf.reduce_sum(
            tf.one_hot(tf.to_int32(tf.reshape(self.x, [-1])), self.num_vocabulary, 1.0, 0.0) * tf.log(
                tf.clip_by_value(tf.reshape(self.temp_g_predictions, [-1, self.num_vocabulary]), 1e-20, 1.0)
            )
        ) / (self.sequence_length * self.batch_size)

        self.selfdefined_temp_persample_len_ll = \
            tf.reshape(
                tf.reduce_sum(
                    tf.one_hot(tf.to_int32(tf.reshape(self.x, [-1])), self.num_vocabulary, 1.0, 0.0) * \
                    tf.log(tf.clip_by_value(tf.reshape(self.temp_g_predictions, [-1, self.num_vocabulary]),
                                            1e-20, 1.0)),
                    axis=-1), self.x.shape)
        self.selfdefined_temp_persample_ll = tf.reduce_sum(self.selfdefined_temp_persample_len_ll, axis=-1)

    def unbiased_temperature_init(self):
        self.unbiased_temperature = tf.placeholder(tf.float32)
        ln_p = tf.zeros((self.fork_degree,), name='ln_p_temps')

        # When current index i < given_num, use the provided tokens as the input at each time step
        def get_expected_foraward_prbability(prefix_len, x_start, h_start):
            def _g_recurrence_1(i, x_t, h_tm1, ln_prob):
                h_t = self.g_recurrent_unit(x_t, h_tm1)  # hidden_memory_tuple
                o_t = self.g_output_unit(h_t)
                next_token = tf.cast(tf.reshape(tf.multinomial(o_t, 1), [self.batch_size]), tf.int32)
                x_tp1 = tf.nn.embedding_lookup(self.g_embeddings, next_token)  # batch x emb_dim
                ln_prob += tf.multiply(tf.one_hot(next_token, self.num_vocabulary, 1.0, 0.0),
                                       tf.nn.log_softmax(o_t))
                return i + 1, x_tp1, h_t, ln_prob

            x_start = tf.tile(tf.expand_dims(x_start, axis=0), (self.fork_degree, 1))
            h_start = tf.tile(tf.expand_dims(h_start, axis=0), (self.fork_degree, 1))
            _, _, _, _, prefix_ln_p = control_flow_ops.while_loop(
                cond=lambda i, _1, _2, _4, _5: i < self.sequence_length,
                body=_g_recurrence_1,
                loop_vars=(prefix_len, x_start, h_start, ln_p)
            )
            prefix_ln_p *= self.unbiased_temperature
            return tf.reduce_logsumexp(prefix_ln_p)
        def get_powered_probs():
            pass
        # When current index i >= given_num, start roll-out, use the output as time step t as the input at time step t+1
        def _g_recurrence_2(i, x_t, h_tm1, given_num, gen_x):
            h_t = self.g_recurrent_unit(x_t, h_tm1)  # hidden_memory_tuple
            o_t = self.g_output_unit(h_t)  # batch x vocab , logits not prob

            next_token = tf.cast(tf.reshape(tf.multinomial(o_t, 1), [self.batch_size]), tf.int32)
            x_tp1 = tf.nn.embedding_lookup(self.g_embeddings, next_token)  # batch x emb_dim
            gen_x = gen_x.write(i, next_token)  # indices, batch_size
            return i + 1, x_tp1, h_t, given_num, gen_x

        i, x_t, h_tm1, given_num, self.gen_x = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, given_num, _4: i < given_num,
            body=_g_recurrence_1,
            loop_vars=(tf.constant(0, dtype=tf.int32),
                       tf.nn.embedding_lookup(self.g_embeddings, self.start_token), self.h0, self.given_num, gen_x))

        _, _, _, _, self.gen_x = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3, _4: i < self.sequence_length,
            body=_g_recurrence_2,
            loop_vars=(i, x_t, h_tm1, given_num, self.gen_x))

        self.gen_x = self.gen_x.stack()  # seq_length x batch_size
        self.gen_x = tf.transpose(self.gen_x, perm=[1, 0])  # batch_size x seq_length

    def temperature_generate(self, sess, temperature):
        outputs = sess.run(self.temp_gen_x, {self.temperature: temperature})
        return outputs


class SavableModel(object):
    def __init__(self, params, parent_folder_path, name):
        self.saving_path = parent_folder_path + '{}/'.format(name)
        self.saver = tf.train.Saver(params, save_relative_paths=True)

    def save_model(self, sess, name=''):
        if name != '':
            saving_path = self.saving_path + name + '/'
        else:
            saving_path = self.saving_path
        self.saver.save(sess, saving_path)

    def load_model(self, sess, name=''):
        import os
        if name != '':
            saving_path = self.saving_path + name + '/'
        else:
            saving_path = self.saving_path
        if not os.path.exists(saving_path):
            os.makedirs(saving_path)
        if not tf.train.checkpoint_exists(saving_path + 'checkpoint'):
            print('Saved {} not found! Randomly initialized.'.format(self.__class__.__name__))
        else:
            self.saver.restore(sess, saving_path)
            print('Model {} loaded from {}!'.format(self.__class__.__name__, saving_path))

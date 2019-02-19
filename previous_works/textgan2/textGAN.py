# -*- coding: utf-8 -*-
"""
Yizhe Zhang

TextGAN
"""
## 152.3.214.203/6006

import inspect
import os

CURR_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) + '/'

GPUID = 1
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPUID)

# from tensorflow.contrib import metrics
# from tensorflow.contrib.learn import monitors
# from tensorflow.contrib.learn.python.learn.metric_spec import MetricSpec

from .model import *
from .utils import prepare_data_for_cnn, get_minibatches_idx, restore_from_save, _clip_gradients_seperate_norm
from .denoise import *
from tensorflow.python.platform import tf_logging as logging

profile = False
# import tempfile
# from tensorflow.examples.tutorials.mnist import input_data

logging.set_verbosity(logging.INFO)
# tf.logging.verbosity(1)
# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS


# flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')


class Options(object):
    def __init__(self, v_size, mx_len):
        self.dis_steps = 10
        self.gen_steps = 1
        self.fix_emb = False
        self.reuse_w = False
        self.reuse_cnn = False
        self.reuse_discrimination = False  # reuse cnn for discrimination
        self.restore = True
        self.tanh = False  # activation fun for the top layer of cnn, otherwise relu
        self.model = 'cnn_rnn'  # 'cnn_deconv'  # 'cnn_rnn', 'rnn_rnn' , default: cnn_deconv

        self.permutation = 0
        self.substitution = 's'  # Deletion(d), Insertion(a), Substitution(s) and Permutation(p)

        self.W_emb = None
        self.cnn_W = None
        self.cnn_b = None
        self.maxlen = mx_len
        self.n_words = v_size
        self.filter_shape = [2, 3]
        self.filter_size = [100, 200]
        self.multiplier = 2
        self.embed_size = 32
        self.latent_size = 32
        self.lr = 1e-5

        self.layer = 3
        self.stride = [2, 2, 2]  # for two layer cnn/deconv , use self.stride[0]
        self.batch_size = 64
        self.max_epochs = 100
        self.n_gan = 128  # self.filter_size * 3
        self.L = 1000

        self.rnn_share_emb = True
        self.additive_noise_lambda = 0.0
        self.bp_truncation = None
        self.n_hid = 32

        self.optimizer = 'Adam'  # tf.train.AdamOptimizer(beta1=0.9) #'Adam' # 'Momentum' , 'RMSProp'
        self.clip_grad = None  # None  #100  #  20#
        self.attentive_emb = False
        self.decay_rate = 0.99
        self.relu_w = False

        self.save_path = os.path.join(CURR_PATH,
                                      "./save/" + "bird_" + str(self.n_gan) +
                                      "_dim_" + self.model + "_" + self.substitution + str(self.permutation))
        self.log_path = os.path.join(CURR_PATH, "./log")
        self.text_path = os.path.join(CURR_PATH, "./text")
        if not os.path.exists(self.text_path): os.mkdir(self.text_path)
        self.print_freq = 10
        self.valid_freq = 100
        self.sigma_range = [2]

        # batch norm & dropout
        self.batch_norm = False
        self.dropout = False
        self.dropout_ratio = 1

        self.discrimination = False
        self.H_dis = 300

        self.sent_len = self.maxlen + 2 * (self.filter_shape - 1)
        self.sent_len2 = np.int32(floor((self.sent_len - self.filter_shape) / self.stride[0]) + 1)
        self.sent_len3 = np.int32(floor((self.sent_len2 - self.filter_shape) / self.stride[1]) + 1)
        self.sent_len4 = np.int32(floor((self.sent_len3 - self.filter_shape) / self.stride[2]) + 1)
        self.sentence = self.maxlen - 1
        print('Use model %s' % self.model)
        print('Use %d conv/deconv layers' % self.layer)

    def __iter__(self):
        for attr, value in self.__dict__.iteritems():
            yield attr, value


def discriminator(x, opt, prefix='d_', is_prob=False, is_reuse=None):
    W_norm_d = embedding_only(opt, prefix=prefix, is_reuse=is_reuse)  # V E
    H = encoder(x, W_norm_d, opt, prefix=prefix + 'enc_', is_prob=is_prob, is_reuse=is_reuse)
    logits = discriminator_2layer(H, opt, is_reuse=is_reuse)

    return logits, H


def encoder(x, W_norm_d, opt, prefix='d_', is_prob=False, is_reuse=None, is_padded=True):
    if is_prob:
        x_emb = tf.tensordot(x, W_norm_d, [[2], [0]])
    else:
        x_emb = tf.nn.embedding_lookup(W_norm_d, x)  # batch L emb
    if not is_padded:  # pad the input with pad_emb
        pad_emb = tf.expand_dims(tf.expand_dims(W_norm_d[0], 0), 0)  # 1*v
        x_emb = tf.concat([tf.tile(pad_emb, [opt.batch_size, opt.filter_shape - 1, 1]), x_emb], 1)

    x_emb = tf.expand_dims(x_emb, 3)  # batch L emb 1
    # bp()
    if opt.layer == 3:
        H = conv_model_3layer(x_emb, opt, prefix=prefix, is_reuse=is_reuse)
    else:
        H = conv_model(x_emb, opt, prefix=prefix, is_reuse=is_reuse)
    return tf.squeeze(H)


def textGAN(x, opt):
    # res = {}
    res_ = {}

    with tf.variable_scope("pretrain"):
        # z = tf.random_uniform([opt.batch_size, opt.latent_size], minval=-1.,maxval=1.)
        z = tf.random_normal([opt.batch_size, opt.latent_size])
        W_norm = embedding_only(opt, is_reuse=None)
        _, syn_sent, logits = lstm_decoder_embedding(z, tf.ones_like(x), W_norm, opt, add_go=True, feed_previous=True,
                                                     is_reuse=None, is_softargmax=True, is_sampling=False)
        prob = [tf.nn.softmax(l * opt.L) for l in logits]
        prob = tf.stack(prob, 1)

        # _, syn_onehot, rec_sent, _ = lstm_decoder_embedding(z, x_org, W_norm, opt)
        # x_emb_fake = tf.tensordot(syn_onehot, W_norm, [[2],[0]])
        # x_emb_fake = tf.expand_dims(x_emb_fake, 3)

    with tf.variable_scope("d_net"):
        logits_real, H_real = discriminator(x, opt)

        ## Real Trail
        # x_emb, W_norm = embedding(x, opt, is_reuse = None)  # batch L emb
        # x_emb = tf.expand_dims(x_emb, 3)  # batch L emb 1
        # H_enc, res = conv_encoder(x_emb, opt, res, is_reuse = None)

    with tf.variable_scope("d_net"):
        logits_fake, H_fake = discriminator(prob, opt, is_prob=True, is_reuse=True)

        # H_enc_fake, res_ = conv_encoder(x_emb_fake, is_train, opt, res_, is_reuse=True)
        # logits_real = discriminator_2layer(H_enc, opt)
        # logits_syn = discriminator_2layer(H_enc_fake, opt, is_reuse=True)

    res_['syn_sent'] = syn_sent
    res_['real_f'] = tf.squeeze(H_real)
    # Loss

    D_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_real), logits=logits_real)) + \
             tf.reduce_mean(
                 tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(logits_fake), logits=logits_fake))

    fake_mean = tf.reduce_mean(H_fake, axis=0)
    real_mean = tf.reduce_mean(H_real, axis=0)
    mean_dist = tf.sqrt(tf.reduce_mean((fake_mean - real_mean) ** 2))
    res_['mean_dist'] = mean_dist

    # cov_fake = acc_fake_xx - tensor.dot(acc_fake_mean.dimshuffle(0, 'x'), acc_fake_mean.dimshuffle(0, 'x').T)  +identity
    # cov_real = acc_real_xx - tensor.dot(acc_real_mean.dimshuffle(0, 'x'), acc_real_mean.dimshuffle(0, 'x').T)  +identity

    # cov_fake_inv = tensor.nlinalg.matrix_inverse(cov_fake)
    # cov_real_inv = tensor.nlinalg.matrix_inverse(cov_real)
    # tensor.nlinalg.trace(tensor.dot(cov_fake_inv,cov_real) + tensor.dot(cov_real_inv,cov_fake))

    GAN_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake, labels=tf.ones_like(logits_fake))
    MMD_loss = compute_MMD_loss(tf.squeeze(H_fake), tf.squeeze(H_real), opt)
    G_loss = mean_dist  # MMD_loss # + tf.reduce_mean(GAN_loss) # mean_dist #
    res_['mmd'] = MMD_loss
    res_['gan'] = tf.reduce_mean(GAN_loss)
    # *tf.cast(tf.not_equal(x_temp,0), tf.float32)
    tf.summary.scalar('D_loss', D_loss)
    tf.summary.scalar('G_loss', G_loss)
    summaries = [
        "learning_rate",
        "loss",
        # "G_loss",
        # "D_loss"
        # "gradients",
        # "gradient_norm",
    ]
    global_step = tf.Variable(0, trainable=False)

    all_vars = tf.trainable_variables()
    g_vars = [var for var in all_vars if
              var.name.startswith('pretrain')]
    d_vars = [var for var in all_vars if
              var.name.startswith('d_')]
    print([g.name for g in g_vars])
    generator_op = layers.optimize_loss(
        G_loss,
        global_step=global_step,
        # aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N,
        # framework.get_global_step(),
        optimizer=opt.optimizer,
        clip_gradients=(lambda grad: _clip_gradients_seperate_norm(grad, opt.clip_grad)) if opt.clip_grad else None,
        learning_rate_decay_fn=lambda lr, g: tf.train.exponential_decay(learning_rate=lr, global_step=g,
                                                                        decay_rate=opt.decay_rate, decay_steps=3000),
        learning_rate=opt.lr,
        variables=g_vars,
        summaries=summaries
    )

    discriminator_op = layers.optimize_loss(
        D_loss,
        global_step=global_step,
        # aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N,
        # framework.get_global_step(),
        optimizer=opt.optimizer,
        clip_gradients=(lambda grad: _clip_gradients_seperate_norm(grad, opt.clip_grad)) if opt.clip_grad else None,
        learning_rate_decay_fn=lambda lr, g: tf.train.exponential_decay(learning_rate=lr, global_step=g,
                                                                        decay_rate=opt.decay_rate, decay_steps=3000),
        learning_rate=opt.lr,
        variables=d_vars,
        summaries=summaries
    )

    # optimizer = tf.train.AdamOptimizer(learning_rate=opt.lr)  # Or another optimization algorithm.
    # train_op = optimizer.minimize(
    #     loss,
    #     aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)

    return res_, G_loss, D_loss, generator_op, discriminator_op


class TextGANMMD:
    def __init__(self, wrapper, parser):
        self.wrapper = wrapper
        self.ixtoword = parser.id2vocab
        self.opt = Options(parser.vocab.shape[0], parser.max_length)
        try:
            params = np.load(os.path.join(CURR_PATH, './param_g.npz'))
            if params['Wemb'].shape == (self.opt.n_words, self.opt.embed_size):
                print('Use saved embedding.')
                self.opt.W_emb = params['Wemb']
            else:
                print('Emb Dimension mismatch: param_g.npz:' + str(params['Wemb'].shape) + ' opt: ' + str(
                    (self.opt.n_words, self.opt.embed_size)))
                self.opt.fix_emb = False
        except IOError:
            print('No embedding file found.')
            self.opt.fix_emb = False

        with tf.device('/gpu:1'):
            self.x_ = tf.placeholder(tf.int32, shape=[self.opt.batch_size, self.opt.sent_len])
            self.x_org_ = tf.placeholder(tf.int32, shape=[self.opt.batch_size, self.opt.sent_len])
            self.is_train_ = tf.placeholder(tf.bool, name='is_train_')
            self.res_, self.g_loss_, self.d_loss_, self.gen_op, self.dis_op = textGAN(self.x_, self.opt)
            self.merged = tf.summary.merge_all()
            # self.opt.is_train = False
            # res_val_, loss_val_, _ = auto_encoder(x_, x_org_, self.opt)
            # merged_val = tf.summary.merge_all()

        # tensorboard --logdir=run1:/tmp/tensorflow/ --port 6006
        # writer = tf.train.SummaryWriter(self.opt.log_path, graph=tf.get_default_graph())

        self.uidx = 0
        config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True,
                                graph_options=tf.GraphOptions(build_cost_model=1))
        # config = tf.ConfigProto(device_count={'GPU':0})
        config.gpu_options.per_process_gpu_memory_fraction = 0.4
        np.set_printoptions(precision=3)
        np.set_printoptions(threshold=np.inf)
        self.saver = tf.train.Saver()

        self.run_metadata = tf.RunMetadata()

        self.sess = tf.Session(config=config)
        self.train_writer = tf.summary.FileWriter(self.opt.log_path + '/train', self.sess.graph)
        self.test_writer = tf.summary.FileWriter(self.opt.log_path + '/test', self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        if self.opt.restore:
            try:
                # pdb.set_trace()

                t_vars = tf.trainable_variables()
                # print([var.name[:-2] for var in t_vars])
                self.loader = restore_from_save(t_vars, self.sess, self.opt)

            except Exception as e:
                print(e)
                print("No saving session, using random initialization")
                self.sess.run(tf.global_variables_initializer())

    def train_func(self, train_data, valid_data):
        self.train = train_data
        self.val = valid_data
        for epoch in range(self.opt.max_epochs):
            print("Starting epoch %d" % epoch)
            # if epoch >= 10:
            #     print("Relax embedding ")
            #     self.opt.fix_emb = False
            #     self.opt.batch_size = 2
            kf = get_minibatches_idx(len(self.train), self.opt.batch_size, shuffle=True)
            for _, train_index in kf:
                self.uidx += 1
                sents = [self.train[t] for t in train_index]

                sents_permutated = add_noise(sents, self.opt)

                # sents[0] = np.random.permutation(sents[0])
                x_batch = prepare_data_for_cnn(sents_permutated, self.opt)  # Batch L
                d_loss = 0
                g_loss = 0
                if profile:
                    if self.uidx % self.opt.dis_steps == 0:
                        _, d_loss = self.sess.run([self.dis_op, self.d_loss_], feed_dict={self.x_: x_batch},
                                                  options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                                                  run_metadata=self.run_metadata)
                    if self.uidx % self.opt.gen_steps == 0:
                        _, g_loss = self.sess.run([self.gen_op, self.g_loss_], feed_dict={self.x_: x_batch},
                                                  options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                                                  run_metadata=self.run_metadata)
                else:
                    if self.uidx % self.opt.dis_steps == 0:
                        _, d_loss = self.sess.run([self.dis_op, self.d_loss_], feed_dict={self.x_: x_batch})
                    if self.uidx % self.opt.gen_steps == 0:
                        _, g_loss = self.sess.run([self.gen_op, self.g_loss_], feed_dict={self.x_: x_batch})

                if self.uidx % self.opt.valid_freq == 0:
                    is_train = True
                    valid_index = np.random.choice(len(self.val), self.opt.batch_size)
                    val_sents = [self.val[t] for t in valid_index]

                    val_sents_permutated = add_noise(val_sents, self.opt)

                    x_val_batch = prepare_data_for_cnn(val_sents_permutated, self.opt)

                    d_loss_val = self.sess.run(self.d_loss_, feed_dict={self.x_: x_val_batch})
                    g_loss_val = self.sess.run(self.g_loss_, feed_dict={self.x_: x_val_batch})

                    res = self.sess.run(self.res_, feed_dict={self.x_: x_val_batch})
                    print("Validation d_loss %f, g_loss %f  mean_dist %f" % (d_loss_val, g_loss_val, res['mean_dist']))
                    print("Sent: " + ' '.join([self.ixtoword[str(x)] for x in res['syn_sent'][0] if x != 0]).strip())
                    print("MMD loss %f, GAN loss %f" % (res['mmd'], res['gan']))
                    np.savetxt(os.path.join(CURR_PATH, './text/rec_val_words.txt'),
                               res['syn_sent'], fmt='%i', delimiter=' ')
                    if self.opt.discrimination:
                        print("Real Prob %f Fake Prob %f" % (res['prob_r'], res['prob_f']))

                    # val_set = [prepare_for_bleu(s) for s in val_sents]
                    # [bleu2s, bleu3s, bleu4s] = cal_BLEU([prepare_for_bleu(s) for s in res['syn_sent']], {0: val_set})
                    # print
                    # 'Val BLEU (2,3,4): ' + ' '.join([str(round(it, 3)) for it in (bleu2s, bleu3s, bleu4s)])

                    summary = self.sess.run(self.merged, feed_dict={self.x_: x_val_batch})
                    self.test_writer.add_summary(summary, self.uidx)

                if self.uidx % self.opt.print_freq == 0:
                    # pdb.set_trace()
                    res = self.sess.run(self.res_, feed_dict={self.x_: x_batch})
                    median_dis = np.sqrt(
                        np.median([((x - y) ** 2).sum() for x in res['real_f'] for y in res['real_f']]))
                    print("Iteration %d: d_loss %f, g_loss %f, mean_dist %f, realdist median %f" % (
                        self.uidx, d_loss, g_loss, res['mean_dist'], median_dis))
                    np.savetxt(os.path.join(CURR_PATH, './text/rec_train_words.txt')
                               , res['syn_sent'], fmt='%i', delimiter=' ')
                    print("Sent: " + ' '.join([self.ixtoword[str(x)] for x in res['syn_sent'][0] if x != 0]).strip())
                    self.saver.save(self.sess, self.opt.save_path, global_step=epoch)
                    self.wrapper.update_scores(self.uidx)

                    summary = self.sess.run(self.merged, feed_dict={self.x_: x_batch})
                    self.train_writer.add_summary(summary, self.uidx)
                    # print res['x_rec'][0][0]
                    # print res['x_emb'][0][0]
                    if profile:
                        tf.contrib.tfprof.model_analyzer.print_model_analysis(
                            tf.get_default_graph(),
                            run_meta=self.run_metadata,
                            tfprof_options=tf.contrib.tfprof.model_analyzer.PRINT_ALL_TIMING_MEMORY)

            self.saver.save(self.sess, self.opt.save_path, global_step=epoch)
        self.wrapper.update_scores(self.uidx)

    def generate(self):
        # x_val_batch = prepare_data_for_cnn(self.val, self.opt)
        res = self.sess.run(self.res_['syn_sent'])
        np.savetxt(os.path.join(CURR_PATH, './text/rec_val_words.txt'),
                   res, fmt='%i', delimiter=' ')
        return [list(map(lambda oo: str(oo), o)) for o in res]

import os
import time
import logging
import json
import math
import numpy as np
import tensorflow as tf
import attention_layers

from util import softmax, orthogonal_initializer

class RCModel(object):
    """
    Implements the reading comprehension model.
    """
    def __init__(self, vocab, args):

        # logging
        self.logger = logging.getLogger("brc")

        # basic config
        self.hidden_size = args.hidden_size
        self.optim_type = args.optim
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.use_dropout = args.dropout_keep_prob < 1

        # the vocab
        self.vocab = vocab

        # session info
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
        sess_config = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session(config=sess_config)

        self._build_graph()

        # save info
        self.saver = tf.train.Saver()

        # initialize the model
        self.sess.run(tf.global_variables_initializer())

    def _build_graph(self):
        """
        Builds the computation graph with Tensorflow
        """
        start_t = time.time()
        self._setup_placeholders()
        self._mask()
        self._embed()
        self._encode()
        self._match()
        self._decode()
        self._compute_loss()
        #self._create_train_op()
        self.logger.info('Time to build graph: {} s'.format(time.time() - start_t))

    def _setup_placeholders(self):
        """
        Placeholders
        """
        self.d = tf.placeholder(tf.int32, [None, None])
        self.q = tf.placeholder(tf.int32, [None, None])
        self.c0 = tf.placeholder(tf.int32, [None, None])
        self.c1 = tf.placeholder(tf.int32, [None, None])
        self.c2 = tf.placeholder(tf.int32, [None, None])
        self.c3 = tf.placeholder(tf.int32, [None, None])
        self.answer = tf.placeholder(tf.int32, [None, None])
        self.dropout_keep_prob = tf.placeholder(tf.float32)

    def _mask(self):
        self.d_m = tf.cast(tf.cast(self.d, tf.bool), tf.int32)
        self.q_m = tf.cast(tf.cast(self.q, tf.bool), tf.int32)
        self.c0_m = tf.cast(tf.cast(self.c0, tf.bool), tf.int32)
        self.c1_m = tf.cast(tf.cast(self.c1, tf.bool), tf.int32)
        self.c2_m = tf.cast(tf.cast(self.c2, tf.bool), tf.int32)
        self.c3_m = tf.cast(tf.cast(self.c3, tf.bool), tf.int32)

    def _embed(self):
        """
        The embedding layer, question and passage share embeddings
        """
        with tf.device('/cpu:0'), tf.variable_scope('word_embedding'):
            self.word_embeddings = tf.get_variable(
                'word_embeddings',
                shape=(self.vocab.size(), self.vocab.embed_dim),
                initializer=tf.constant_initializer(self.vocab.embeddings),
                trainable=True
            )
            self.d_emb = tf.nn.dropout(tf.nn.embedding_lookup(self.word_embeddings, self.d), self.dropout_keep_prob)
            self.q_emb = tf.nn.dropout(tf.nn.embedding_lookup(self.word_embeddings, self.q), self.dropout_keep_prob)
            self.c0_emb = tf.nn.dropout(tf.nn.embedding_lookup(self.word_embeddings, self.c0), self.dropout_keep_prob)
            self.c1_emb = tf.nn.dropout(tf.nn.embedding_lookup(self.word_embeddings, self.c1), self.dropout_keep_prob)
            self.c2_emb = tf.nn.dropout(tf.nn.embedding_lookup(self.word_embeddings, self.c2), self.dropout_keep_prob)
            self.c3_emb = tf.nn.dropout(tf.nn.embedding_lookup(self.word_embeddings, self.c3), self.dropout_keep_prob)

    def _encode(self):
        with tf.variable_scope('article', initializer=tf.random_uniform_initializer(minval=0, maxval=0.1)):
            fwd_cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)
            back_cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)

            self.d_len = tf.reduce_sum(self.d_m, reduction_indices=1)
            h, _ = tf.nn.bidirectional_dynamic_rnn(
                fwd_cell, back_cell, self.d_emb, sequence_length=tf.to_int64(self.d_len), dtype=tf.float32)
            self.h_doc = tf.concat(h, 2)

        with tf.variable_scope('question', initializer=tf.random_uniform_initializer(minval=0, maxval=0.1)):
            fwd_cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)
            back_cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)

            self.q_len = tf.reduce_sum(self.q_m, reduction_indices=1)
            _, h = tf.nn.bidirectional_dynamic_rnn(
                fwd_cell, back_cell, self.q_emb, sequence_length=tf.to_int64(self.q_len), dtype=tf.float32)
            self.h_q = tf.concat(h, -1)
            #self.pri = self.h_q.get_shape()
            #h_doc = tf.concat(h, 2)

        with tf.variable_scope('c', initializer=tf.random_uniform_initializer(minval=0, maxval=0.1)):
            fwd_cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)
            back_cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)

            c0_len = tf.reduce_sum(self.c0_m, reduction_indices=1)
            c1_len = tf.reduce_sum(self.c1_m, reduction_indices=1)
            c2_len = tf.reduce_sum(self.c2_m, reduction_indices=1)
            c3_len = tf.reduce_sum(self.c3_m, reduction_indices=1)
            _, h_0 = tf.nn.bidirectional_dynamic_rnn(
                fwd_cell, back_cell, self.c0_emb, sequence_length=tf.to_int64(c0_len), dtype=tf.float32)
            _, h_1 = tf.nn.bidirectional_dynamic_rnn(
                fwd_cell, back_cell, self.c1_emb, sequence_length=tf.to_int64(c1_len), dtype=tf.float32)
            _, h_2 = tf.nn.bidirectional_dynamic_rnn(
                fwd_cell, back_cell, self.c2_emb, sequence_length=tf.to_int64(c2_len), dtype=tf.float32)
            _, h_3 = tf.nn.bidirectional_dynamic_rnn(
                fwd_cell, back_cell, self.c3_emb, sequence_length=tf.to_int64(c3_len), dtype=tf.float32)
            #h_query = tf.nn.dropout(tf.concat(2, h), FLAGS.dropout_keep_prob)
            self.h_c0 = tf.concat(h_0, -1)
            self.h_c1 = tf.concat(h_1, -1)
            self.h_c2 = tf.concat(h_2, -1)
            self.h_c3 = tf.concat(h_3, -1)


    def _match(self):
        self.batch_size = tf.shape(self.h_q)[0]
        self.att_d = attention_layers.bilinear(self.hidden_size, self.h_q, self.h_doc, self.batch_size)

    def _decode(self):
        with tf.variable_scope('predict'):
            W2 = tf.get_variable(shape=[self.hidden_size*2, self.hidden_size*2], dtype=tf.float32, name='W2',
                                 initializer=tf.random_uniform_initializer(minval=-0.01, maxval=0.01))
            alpha0 = tf.matmul(self.h_c0, W2)  # (bz, hz)
            pro0 = tf.matmul(tf.reshape(alpha0, [self.batch_size, 1, self.hidden_size * 2])
                             , tf.reshape(self.att_d, [self.batch_size, self.hidden_size * 2, 1]))
            alpha1 = tf.matmul(self.h_c1, W2)  # (bz, hz)
            pro1 = tf.matmul(tf.reshape(alpha1, [self.batch_size, 1, self.hidden_size * 2])
                             , tf.reshape(self.att_d, [self.batch_size, self.hidden_size * 2, 1]))
            alpha2 = tf.matmul(self.h_c2, W2)  # (bz, hz)
            pro2 = tf.matmul(tf.reshape(alpha2, [self.batch_size, 1, self.hidden_size * 2])
                             , tf.reshape(self.att_d, [self.batch_size, self.hidden_size * 2, 1]))
            alpha3 = tf.matmul(self.h_c3, W2)  # (bz, hz)
            pro3 = tf.matmul(tf.reshape(alpha3, [self.batch_size, 1, self.hidden_size * 2])
                             , tf.reshape(self.att_d, [self.batch_size, self.hidden_size * 2, 1]))

            self.score = tf.stack([tf.reshape(pro0, [self.batch_size])
                                      , tf.reshape(pro1, [self.batch_size])
                                      , tf.reshape(pro2, [self.batch_size])
                                      , tf.reshape(pro3, [self.batch_size])])
            self.pri = tf.reshape(pro0, [self.batch_size])

    def _compute_loss(self):
        """
        The loss function
        """
        self.pre_answer = tf.transpose(self.score)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pre_answer, labels=tf.to_float(self.answer)))

        global_step = tf.Variable(0, name="global_step", trainable=False)

        p_a = tf.argmax(self.score, 0)
        self.accuracy = tf.reduce_mean(tf.to_float(tf.equal(p_a, tf.argmax(self.answer, 1))))

        # train_op = tf.train.GradientDescentOptimizer(0.0).minimize(loss)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        capped_grads_and_vars = [(tf.clip_by_value(grad, -5, 5), var) for (grad, var) in grads_and_vars]
        self.train_op = optimizer.apply_gradients(capped_grads_and_vars, global_step=global_step)

        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy', self.accuracy)

    def _train_epoch(self, train_batches, dropout_keep_prob):
        """
        Trains the model for a single epoch.
        Args:
            train_batches: iterable batch data for training
            dropout_keep_prob: float value indicating dropout keep probability
        """
        total_num, total_loss, total_acc = 0, 0, 0
        log_every_n_batch, n_batch_loss = 50, 0

        for bitx, batch in enumerate(train_batches, 1):
            #print(batch['article_token_ids'])
            feed_dict = {self.d: batch['article_token_ids'],
                         self.q: batch['question_token_ids'],
                         self.c0: batch['option0_ids'],
                         self.c1: batch['option1_ids'],
                         self.c2: batch['option2_ids'],
                         self.c3: batch['option3_ids'],
                         self.answer: batch['answer'],
                         self.dropout_keep_prob: dropout_keep_prob}
            _, loss, accuracy = self.sess.run([self.train_op, self.loss, self.accuracy], feed_dict)
            #print("I want know:" + str(sss))
            total_loss += loss * len(batch['raw_data'])
            total_acc += accuracy
            total_num += len(batch['raw_data'])
            n_batch_loss += loss
            if log_every_n_batch > 0 and bitx % log_every_n_batch == 0:
                self.logger.info('Average loss from batch {} to {} is {}, Accuracy is {}'.format(
                    bitx - log_every_n_batch + 1, bitx, n_batch_loss / log_every_n_batch, total_acc / log_every_n_batch))
                n_batch_loss = 0
                total_acc = 0
        return 1.0 * total_loss / total_num

    def train(self, data, epochs, batch_size, save_dir,
              dropout_keep_prob=1.0, evaluate=True):
        """
        Train the model with data
        Args:
            data: the BRCDataset class implemented in dataset.py
            epochs: number of training epochs
            batch_size:
            save_dir: the directory to save the model
            save_prefix: the prefix indicating the model type
            dropout_keep_prob: float value indicating dropout keep probability
            evaluate: whether to evaluate the model on test set after each epoch
        """
        pad_id = self.vocab.get_id(self.vocab.pad_token)
        for epoch in range(1, epochs + 1):
            #self.restore(save_dir, str(restore_epoch))#restore model for continues training
            self.logger.info('Training the model for epoch {}'.format(epoch))
            train_batches = data.gen_mini_batches('train', batch_size, pad_id, shuffle=True)
            train_loss = self._train_epoch(train_batches, dropout_keep_prob)
            self.logger.info('Average train loss for epoch {} is {}'.format(epoch, train_loss))

            if evaluate:
                self.logger.info('Evaluating the model after epoch {}'.format(epoch))
                if data.dev_set is not None:
                    eval_batches = data.gen_mini_batches('dev', batch_size, pad_id, shuffle=False)
                    eva_loss, eva_acc = self.evaluate(eval_batches)
                    #self.logger.info('Dev eval loss {}'.format(eva_loss))
                    self.logger.info('Dev eval result: {}'.format(eva_acc))


                    self.save(save_dir, str(epoch))
                else:
                    self.logger.warning('No dev set is loaded for evaluation in the dataset!')
            else:
                self.save(save_dir,  '_' + str(epoch))

    def evaluate(self, eval_batches, result_dir=None, result_prefix=None, save_full_info=False):
        """
        Evaluates the model performance on eval_batches and results are saved if specified
        Args:
            eval_batches: iterable batch data
            result_dir: directory to save predicted answers, answers will not be saved if None
            result_prefix: prefix of the file for saving predicted answers,
                           answers will not be saved if None
            save_full_info: if True, the pred_answers will be added to raw sample and saved
        """
        total_loss, total_num, total_acc = 0, 0, 0

        for b_itx, batch in enumerate(eval_batches):
            feed_dict = {self.d: batch['article_token_ids'],
                         self.q: batch['question_token_ids'],
                         self.c0: batch['option0_ids'],
                         self.c1: batch['option1_ids'],
                         self.c2: batch['option2_ids'],
                         self.c3: batch['option3_ids'],
                         self.answer: batch['answer'],
                         self.dropout_keep_prob: 1.0}
            accuracy = self.sess.run([self.accuracy], feed_dict)
            #print(accuracy[0])

            total_acc += accuracy[0] * len(batch['raw_data'])
            total_num += len(batch['raw_data'])

        return total_loss / total_num, total_acc / total_num


    def save(self, model_dir, model_prefix):
        """
        Saves the model into model_dir with model_prefix as the model indicator
        """
        self.saver.save(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info('Model saved in {}, with prefix {}.'.format(model_dir, model_prefix))

    def restore(self, model_dir, model_prefix):
        """
        Restores the model into model_dir from model_prefix as the model indicator
        """
        self.saver.restore(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info('Model restored from {}, with prefix {}'.format(model_dir, model_prefix))
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

import os
import sys
import time

class Rhythm:

    def __init__(self, n_lookback, n_features, n_channels, n_outputs, learning_rate=0.0001, epoches=1000):

        self.epoches = epoches
        self.batch_size = 64

        self.n_lookback = n_lookback
        self.n_features = n_features
        self.n_channels = n_channels
        self.n_outputs = n_outputs

        self.learning_rate = learning_rate

        self.n_l1 = 64
        self.n_fc = 32

        self.build()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.train_writer = tf.summary.FileWriter('log/train/', self.sess.graph)
        self.test_writer = tf.summary.FileWriter('log/test/', self.sess.graph)

        self.saver = tf.train.Saver(max_to_keep=100, keep_checkpoint_every_n_hours=2)

    def build(self):

        def _build_layer(dataset, keep_prob, c_names, n_l1, n_fc, w_initializer, b_initializer, is_training):

            n_filter = 64

            with tf.variable_scope('conv1') as scope:
                # regularizer1 = tf.contrib.layers.l2_regularizer(scale=0.1)
                # conv1 = tf.layers.conv2d(dataset, 64, (8, 8), strides=(1, 1), padding='SAME', kernel_regularizer=regularizer1, activation=None)
                conv1_filter = tf.Variable(tf.truncated_normal([1, 1, self.n_channels, n_filter]))
                # conv1 = tf.layers.conv2d(dataset, 64, (1, 8), strides=(1, 1), padding='SAME', activation=None)
                conv1 = tf.nn.conv2d(dataset, conv1_filter, strides=[1, 1, 1, 1], padding='SAME')
                # conv1 = tf.contrib.layers.batch_norm(conv1, is_training=is_training, updates_collections=None)
                # conv1 = tf.layers.dropout(conv1, keep_prob, training=is_training, noise_shape=[tf.shape(conv1)[0], 1, tf.shape(conv1)[2], 1])
                # conv1 = tf.layers.max_pooling2d(conv1, 2, (1, 4), padding='SAME')

            with tf.variable_scope('conv2') as scope:
                conv2_filter = tf.Variable(tf.truncated_normal([3, 3, self.n_channels, n_filter]))
                conv2 = tf.nn.conv2d(dataset, conv2_filter, strides=[1, 1, 1, 1], padding='SAME')

            with tf.variable_scope('conv3') as scope:
                conv3_filter = tf.Variable(tf.truncated_normal([5, 5, self.n_channels, n_filter]))
                conv3 = tf.nn.conv2d(dataset, conv3_filter, strides=[1, 1, 1, 1], padding='SAME')

            with tf.variable_scope('conv4') as scope:
                conv4 = tf.nn.avg_pool(dataset, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')

            with tf.variable_scope('concat') as scope:
                fc = tf.concat([conv1, conv2, conv3, conv4], axis=3)
                bias = tf.Variable(tf.truncated_normal([3 * n_filter + self.n_channels]))
                fc = tf.nn.bias_add(fc, bias)
                fc = tf.nn.relu(fc)
                fc = tf.reshape(fc, shape=[-1, self.n_lookback, 8299])

            with tf.variable_scope('rnn') as scope:
                cell = tf.contrib.rnn.BasicLSTMCell(num_units=n_l1, state_is_tuple=True)
                state_in = cell.zero_state(tf.shape(fc)[0], tf.float32)
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
                rnn, state = tf.nn.dynamic_rnn(inputs=fc, cell=cell, dtype=tf.float32, initial_state=state_in)
                fc = state[1]

            # with tf.variable_scope('fc') as scope:
            #     fc = tf.layers.dense(fc, self.n_outputs)
            # with tf.variable_scope('l1') as scope:
            #     fc = tf.reshape(conv3, shape=[-1, 128 * 2])
            #     w1 = tf.get_variable('w1', shape=[128 * 2, n_l1], initializer=w_initializer, collections=c_names)
            #     b1 = tf.get_variable('b1', shape=[1, n_l1], initializer=b_initializer, collections=c_names)
            #     l1 = tf.nn.relu(tf.matmul(fc, w1) + b1)
            #     l1 = tf.nn.dropout(l1, keep_prob)
            #
            with tf.variable_scope('l2') as scope:
                w2 = tf.get_variable('w2', shape=[n_l1, n_fc], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', shape=[1, n_fc], initializer=b_initializer, collections=c_names)
                l2 = tf.nn.relu(tf.matmul(fc, w2) + b2)
                fc = tf.nn.dropout(l2, keep_prob)
            #
            with tf.variable_scope('l3') as scope:
                w3 = tf.get_variable('w3', shape=[n_fc, self.n_outputs], initializer=w_initializer, collections=c_names)
                b3 = tf.get_variable('b3', shape=[1, self.n_outputs], initializer=b_initializer, collections=c_names)
                fc = tf.matmul(fc, w3) + b3

            return fc

        self.keep_prob = tf.placeholder(tf.float32, shape=())
        self.dataset = tf.placeholder(tf.float32, shape=(None, None, self.n_features, self.n_channels), name='dataset')
        self.labels = tf.placeholder(tf.float32, shape=(None, self.n_outputs), name='predict')
        self.is_training = tf.placeholder(tf.bool, shape=())

        c_names, w_initializer, b_initializer = \
            ['net_params', tf.GraphKeys.GLOBAL_VARIABLES], tf.contrib.layers.xavier_initializer(), tf.random_normal_initializer()

        self.logits = _build_layer(self.dataset, self.keep_prob, c_names, self.n_l1, \
            self.n_fc, w_initializer, b_initializer, self.is_training)

        with tf.variable_scope('loss') as scope:
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits))

        # beta = 0.01
        # regularizer = tf.nn.l2_loss(w2)
        # self.loss = tf.reduce_mean(self.loss + beta * regularizer)

        global_step = tf.Variable(0, trainable=False)
        # learning_rate = tf.train.exponential_decay(self.learning_rate, global_step, 500, 0.96)
        learning_rate = self.learning_rate

        # self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.loss, global_step=global_step)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss, global_step=global_step)

        self.prediction = tf.nn.softmax(self.logits)
        self.accuancy = self.accuracy_stat()

        with tf.variable_scope('summary') as scope:
            summary_tags = ['accuancy', 'loss']

            self.summary = {}
            self.summary_ops = {}

            for tag in summary_tags:
                self.summary[tag] = tf.placeholder(tf.float32, None, name=tag.replace(' ', '_') + '_0')
                self.summary_ops[tag] = tf.summary.scalar(tag, self.summary[tag])

    def accuracy_stat(self):
        correct_pred = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.labels, 1))
        return tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def train(self, X_train, y_train, X_test, y_test, load_sess=False):

        # X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.01, random_state=1)
        # X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.01, random_state=2)

        total_batch = int(X_train.shape[0] / self.batch_size)
        X_batches = np.array_split(X_train, total_batch)
        predict_batches = np.array_split(y_train, total_batch)

        if load_sess:
            start = self.load()
        else:
            start = 0

        for epoch in range(start, self.epoches):

            total_accuancy = 0
            total_loss = 0
            self.start_time = time.time()

            for i in range(total_batch):

                observation = X_batches[i]
                predict = predict_batches[i]

                # dataset = [None] * (len(observation) - 4)
                #
                # for j in range(4, len(observation)):
                #     dataset[j-4] = [observation[j-4], observation[j-3], observation[j-2], observation[j-1], observation[j]]
                #
                # dataset = np.array(dataset)
                # predict = predict[4:]

                accuancy, _, cost = self.sess.run([self.accuancy, self.optimizer, self.loss], \
                    feed_dict={self.dataset: observation, self.labels: predict, self.keep_prob: 0.8, self.is_training: True})

                total_accuancy += accuancy
                total_loss += cost

                self.print_process(i+1, total_batch, total_accuancy, total_loss)

            avg_accuancy = total_accuancy / total_batch
            avg_loss = total_loss / total_batch

            test_accuancy, test_cost = self.sess.run([self.accuancy, self.loss], \
                feed_dict={self.dataset: X_test, self.labels: y_test, self.keep_prob: 1, self.is_training: False})

            train_summary = self.sess.run(self.summary_ops, feed_dict={
                self.summary['accuancy']: avg_accuancy,
                self.summary['loss']: avg_loss,
                })

            self.train_writer.add_summary(train_summary['accuancy'], epoch)
            self.train_writer.add_summary(train_summary['loss'], epoch)

            test_summary = self.sess.run(self.summary_ops, feed_dict={
                self.summary['accuancy']: test_accuancy,
                self.summary['loss']: test_cost,
                })

            self.test_writer.add_summary(test_summary['accuancy'], epoch)
            self.test_writer.add_summary(test_summary['loss'], epoch)

            print('\nEpoch: {}, accuancy: {:.4f}, cost: {:.6f}, test accuancy: {:.4f}, cost: {:.6f}'
                .format(epoch, avg_accuancy, avg_loss, test_accuancy, test_cost))

            # exam_accuancy, exam_cost = self.sess.run([self.accuancy, self.loss], \
            #     feed_dict={self.dataset: X_exam, self.labels: y_exam, self.keep_prob: 1, self.is_training: False})
            #
            # print('Exam accuancy: {:.4f}, cost: {:.6f}'
            #     .format(exam_accuancy, exam_cost))

            # self.best = 0
            # if avg_accuancy > 0.75 and avg_accuancy > self.best:
            self.save(epoch)
            self.best = avg_accuancy

            # if self.best > 0.75 and epoch % 2 == 0:
            #     valid_accuancy, valid_cost = self.sess.run([self.accuancy, self.loss], \
            #         feed_dict={self.dataset: X_valid, self.labels: y_valid, self.keep_prob: 1, self.is_training: False})

                # print('Accuancy: {:.4f}, cost: {:.6f}, test accuancy: {:.4f}, cost: {:.6f}, valid accuancy: {:.4f}, cost: {:.6f}'
                #     .format(avg_accuancy, avg_loss, test_accuancy, test_cost, valid_accuancy, valid_cost))

    def print_process(self, step, total, accuancy, loss):
        sys.stdout.write('\rProcessing: {}/{}, accuancy: {:.4f}, cost: {:.4f} [{}]' \
            .format(step, total, accuancy/step, loss/step, self.get_time()))
        sys.stdout.flush()

    def get_time(self):
        elapsed_time = (time.time() - self.start_time) / 60
        unit = 'm'

        if (elapsed_time > 60):
            unit = 'h'
            elapsed_time = elapsed_time / 60

        return '{:.2f}{}'.format(elapsed_time, unit)

    def save(self, step):
        save_path = './data/sess.ckpt'
        self.saver.save(self.sess, save_path, global_step=step)
        print('Saving sess to {}: {}'.format(save_path, step))

    def load(self):

        # checkpoint_dir = '/Users/cc/Project/Lean/Launcher/bin/Debug/python/oracle/data/'
        # checkpoint_dir = './data'
        checkpoint_dir = '/Users/cc/documents/ds/motherland/model/1'
        start_step = 0

        try:
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            start_step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
        except Exception as e:
            print('Failed to find sess: {}'.format(str(e)))
            ckpt = None

        if not ckpt or not ckpt.model_checkpoint_path:
            print('Cannot find any saved sess in checkpoint_dir')
            sys.exit(2)
        else:
            try:
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
                self.train_writer.add_session_log(tf.SessionLog(status=tf.SessionLog.START), global_step=start_step)
                self.test_writer.add_session_log(tf.SessionLog(status=tf.SessionLog.START), global_step=start_step)
                print('Sess restored successfully: {}'.format(ckpt.model_checkpoint_path))
            except Exception as e:
                print('Failed to load sess: {}'.format(str(e)))
                sys.exit(2)

        return start_step

    def valid(self, X, y):
        self.load()

        print('Testing...')
        accuancy = 0
        stat = np.zeros(6)
        actual = np.zeros(6)
        result = np.array([np.zeros(6), np.zeros(6), np.zeros(6), np.zeros(6), np.zeros(6), np.zeros(6)])

        predict_up_actual_up = 0
        actual_up = 1
        predict_down_actual_down = 0
        actual_down = 1

        for i in range(len(X)):
            predict = self.sess.run(self.prediction, \
                feed_dict={self.dataset: np.array([X[i]]), self.keep_prob: 1, self.is_training: False})

            expect = np.argmax(y[i], axis=0)
            predict = int(np.argmax(predict, axis=1))

            if not expect == 0:
                actual_up += 1

            if expect == 0:
                actual_down += 1

            if not expect == 0 and not predict == 0:
                predict_up_actual_up += 1

            if expect == 0 and predict == 0:
                predict_down_actual_down += 1

            if predict == expect:
                accuancy += 1
                stat[predict] += 1

            result[predict][expect] += 1
            actual[expect] += 1

            stat = stat.astype(int)
            actual = actual.astype(int)
            # print('accuancy: {}/{}, predict: {}, expect: {}'.format(accuancy, i+1, predict, expect))
            sys.stdout.write('\rStep: {}, Accuancy: {:.2f}, ' \
                'Up: {:.2f}, Down: {:.2f}, 0: {}/{}, 1: {}/{}, 2: {}/{}, ' \
                '3: {}/{}, 4: {}/{}, 5: {}/{}' \
                .format(i, accuancy/float(i+1), predict_up_actual_up/float(actual_up), predict_down_actual_down/float(actual_down), \
                    stat[0], actual[0], stat[1], actual[1], \
                    stat[2], actual[2], stat[3], actual[3], stat[4], actual[4], \
                    stat[5], actual[5]))
            sys.stdout.flush()

        # print('overall accuancy: {}'.format(accuancy/float(len(X))))
        print('\nPredicted\t[0]\t[1]\t[2]\t[3]\t[4]\t[5]')
        for i in range(len(result)):
            print('{}\t\t{}\t{}\t{}\t{}\t{}\t{}'.format(i, result[i][0], result[i][1], result[i][2], result[i][3], result[i][4], result[i][5]))

        print('\n')            

    def predict(self, X):
        # self.load()

        predict = self.sess.run(self.prediction, feed_dict={self.dataset: X, self.keep_prob: 1, self.is_training: False})

        return predict

if __name__ == '__main__':

    mode = 'train'
    # mode = 'valid'

    # X = pickle.load(open('./data/' + mode + '/data.pkl', 'rb')) # (24949, 9, 26, 4)
    # X = pickle.load(open('./data/' + mode + '/test.pkl', 'rb')) # (24949, 9, 26, 4)
    # y = pickle.load(open('./data/' + mode + '/predict.pkl', 'rb')) # (24949, 7)

    # X = joblib.load('./data/' + mode + '/data.pkl')
    # y = joblib.load('./data/' + mode + '/predict.pkl')

    if mode == 'train':
        X_train = joblib.load('./data/X_train.pkl')
        y_train = joblib.load('./data/y_train.pkl')
        X_test = joblib.load('./data/X_test.pkl')
        y_test = joblib.load('./data/y_test.pkl')
    elif mode == 'valid':
        X_valid = joblib.load('./data/X_valid.pkl')
        y_valid = joblib.load('./data/y_valid.pkl')

    print('Data loaded successfully')

    if mode == 'train':
        print('X_train.shape: {}, y_train.shape: {}'.format(X_train.shape, y_train.shape))
        print('X_test.shape: {}, y_test.shape: {}'.format(X_test.shape, y_test.shape))
    else:
        print('X_valid.shape: {}, y_valid.shape: {}'.format(X_valid.shape, y_valid.shape))

    n_lookback = 9 #X.shape[1]
    n_features = 43 #X.shape[2]
    n_channels = 1 #X.shape[3]
    n_outputs = 6 #y.shape[1]

    learning_rate = 0.0001
    epoches = 500

    print('Initializating model')
    rthythm = Rhythm(n_lookback, n_features, n_channels, n_outputs, learning_rate, epoches)
    # rthythm.load()

    if mode == 'train':
        print('Training start [{}]'.format(time.strftime('%X')))
        rthythm.train(X_train, y_train, X_test, y_test, True)
    else:
        print('Valid start')
        rthythm.valid(X_valid, y_valid)

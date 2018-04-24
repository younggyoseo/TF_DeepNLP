import tensorflow as tf
import numpy as np
import os
import time
import pickle
import random
from collections import Counter

class NotTrainedError(Exception):
    pass

class NotFitToCorpusError(Exception):
    pass

class QRNN():
    def __init__(self, batch_size=24, dropout_keep_prob=0.7, learning_rate=1e-3,
                 filter_windows=[2,2,2,2], l2_reg_lambda=4e-5, hidden_size=256,
                 zoneout_keep_prob=1.0):
        print("DEBUG: 04231523")
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout_keep_prob = dropout_keep_prob
        self.zoneout_keep_prob = zoneout_keep_prob
        self.filter_windows = filter_windows
        self.l2_reg_lambda = l2_reg_lambda
        self.hidden_size = hidden_size
        self.__words = None
        self.__word_to_id = None
        self.__word_embedding_size = 0
        self.__max_seq_length = 0
        self.__num_classes = 0
    
    def fit_to_corpus(self, train_data):
        self.__fit_to_corpus(train_data, is_training=True)
        self.__build_graph()

    def __fit_to_corpus(self, data, is_training):
        # TRAINING
        if is_training == True:
            train, train_label, dev, dev_label, w2v, word_to_idx = data
            self.__words = list(word_to_idx.keys())
            self.__word_to_id = word_to_idx
            self.__w2v = w2v
            self.__train_list = list(zip(train, train_label))
            self.__valid_list = list(zip(dev, dev_label))
            self.__word_embedding_size = w2v.shape[1]
            self.__max_seq_length = train.shape[1]
            self.__num_classes = len(set(train_label))

        elif is_training == False:
            test, test_label = data
            self.__test_list = list(zip(test, test_label))

    def __build_graph(self):
        self.__graph = tf.Graph()

        with self.__graph.as_default():
            self.__input = tf.placeholder(tf.int32, shape=[self.batch_size, self.__max_seq_length])
            self.__label = tf.placeholder(tf.int32, shape=[self.batch_size])
            self.__dropout = tf.placeholder(tf.float32)
            self.__zoneout = tf.placeholder(tf.float32)

            with tf.variable_scope("Embedding"):
                # Define Embedding Matrices.
                word_embeddings = tf.get_variable(
                    name = "word_embeddings",
                    shape = [self.vocab_size, self.__word_embedding_size],
                    initializer = tf.constant_initializer(self.__w2v, verify_shape=True))

                # Embedding Layer
                word_embedding= tf.nn.embedding_lookup(  
                    params = word_embeddings,
                    ids = self.__input)  # [batch_size, max_seq_length, word_embedding_size]

                # [batch_size, max_seq_length, word_embedding_size]
                word_embedding_drop = tf.nn.dropout(
                    x = word_embedding,
                    keep_prob = self.__dropout,
                    name = "word_embedding_drop")

            # QRNN Layers
            for idx, filter_window in enumerate(self.filter_windows):
                if idx == 0:
                    input_ = word_embedding_drop
                    prev_c = 0.0

                d = input_.shape.as_list()[2]

                with tf.variable_scope("QRNN_{}".format(idx+1)):
                    # Convolution
                    with tf.variable_scope("Convolution"):
                        input_padded = tf.pad(
                            tensor = input_,
                            paddings = [[0,0], [filter_window-1,0], [0,0]],
                            name = "input_padded")

                        W = tf.get_variable(
                            name = "W",
                            shape = [filter_window, d, self.hidden_size * 3],
                            initializer = tf.random_normal_initializer(stddev=0.1))

                        ZFO = tf.nn.conv1d(
                            value = input_padded,
                            filters = W,
                            stride = 1,
                            padding = "VALID")

                        # [batch_size, max_seq_length, hidden_size]
                        Z, F, O = tf.split(
                            value = ZFO,
                            num_or_size_splits = 3,
                            axis = 2)

                    with tf.variable_scope("fo-Pool"):
                        Z = tf.tanh(Z)
                        O = tf.sigmoid(O)
                        F = tf.sigmoid(F)
                        F = 1 - tf.nn.dropout(1-F, self.__zoneout) * self.__zoneout
                        F = tf.reshape(
                            tensor = F,
                            shape = [self.batch_size, self.__max_seq_length, self.hidden_size])
                        ZF = (1. - F) * Z

                        def _step(f, zf, o, prev_c):
                            c = tf.multiply(f, prev_c) + zf
                            h = tf.multiply(o, c)
                            return c, h

                        length = F.shape.as_list()[1]
                        h_list = []
                        for i in range(length):
                            prev_c, h = _step(F[:,i,:], ZF[:,i,:], O[:,i,:], prev_c)
                            h_list.append(h)

                        H_ = tf.concat([tf.expand_dims(x, axis=1) for x in h_list], axis=1)
                        H = tf.reshape(
                            tensor = H_,
                            shape = [self.batch_size, self.__max_seq_length, self.hidden_size],
                            name = "H")
                            
                        input_ = tf.nn.dropout(
                            x = H,
                            keep_prob = self.__dropout,
                            name = "input_")
            
            with tf.variable_scope("Fully-Connected"):
                # [batch_size * max_seq_length, hidden_size]
                lengths = tf.reduce_sum(tf.sign(self.__input), axis=1)
                indices = tf.range(0, self.batch_size) * self.__max_seq_length + (lengths - 1)
                H_flat = tf.reshape(
                    tensor = H,
                    shape = [-1, self.hidden_size],
                    name = "H_flat")
                
                # [batch_size, hidden_size]
                H_last = tf.gather(
                    params = H_flat,
                    indices = indices,
                    axis = 0,
                    name = "H_last")

                H_drop = tf.nn.dropout(
                    x = H_last,
                    keep_prob = self.__dropout,
                    name = "H_drop")

                W_out = tf.get_variable(
                    name = "W_out",
                    shape = [self.hidden_size, self.__num_classes],
                    initializer = tf.truncated_normal_initializer(stddev=0.1))

                b_out = tf.get_variable(
                    name = "b_out",
                    shape = [self.__num_classes],
                    initializer = tf.truncated_normal_initializer(stddev=0.1))

                logits = tf.nn.xw_plus_b(
                    x = H_drop,
                    weights = W_out,
                    biases = b_out,
                    name = "logits")

            with tf.variable_scope("Loss"):
                # Accuracy
                predictions = tf.argmax(
                    input = logits,
                    axis = 1,
                    output_type = tf.int32)

                correct_predictions = tf.equal(
                    x = predictions,
                    y = self.__label)

                self.__accuracy = tf.reduce_mean(
                    input_tensor = tf.cast(correct_predictions, "float"),
                    name="accuracy")

                # Loss
                single_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits = logits,
                    labels = self.__label)

                l2_loss = tf.nn.l2_loss(W_out)

                self.__total_loss = tf.reduce_mean(single_losses) + self.l2_reg_lambda * l2_loss

            with tf.variable_scope("Optimize"):
                self.__global_step = tf.get_variable(
                    name = "global_step",
                    shape = [],
                    initializer = tf.constant_initializer(0),
                    trainable = False,
                    dtype = tf.int32)

                self.__learning_rate = tf.get_variable(
                    name = "learning_rate",
                    shape = [],
                    initializer = tf.constant_initializer(self.learning_rate),
                    trainable = False)

                self.__optimizer = tf.train.RMSPropOptimizer(self.__learning_rate, epsilon=1e-8).minimize(
                    loss = self.__total_loss,
                    global_step = self.__global_step,
                    name = "optimizer")

    def train(self, num_epochs, save_dir, log_dir=None, load_dir=None, print_every=20):
        should_write_summaries = log_dir is not None
        with tf.Session(graph=self.__graph) as session:

            tf.set_random_seed(1004)

            if should_write_summaries:
                summary_writer = tf.summary.FileWriter(log_dir, graph=session.graph)
            if load_dir is not None:
                self.__load(session, load_dir)
                print('-' * 80)
                print('Restored model from checkpoint. Size: {}'.format(_model_size()))
                print('-' * 80)
            else:
                tf.global_variables_initializer().run()
                print('-' * 80)
                print('Created and Initialized fresh model. Size: {}'.format(_model_size()))
                print('-' * 80)

            saver = tf.train.Saver(max_to_keep=5)
            train_batches = self.__prepare_batches("train")
            valid_batches = self.__prepare_batches("valid")

            for epoch in range(1, num_epochs+1):
                epoch_start_time = time.time()
                avg_train_loss = 0.0
                train_accuracy = 0.0
                count = 0
                random.shuffle(train_batches)

                for batch in train_batches:
                    inputs, labels = batch
                    count += 1
                    start_time = time.time()

                    loss, step, accuracy, *_ = session.run([
                        self.__total_loss,
                        self.__global_step,
                        self.__accuracy,
                        self.__optimizer,
                    ], {
                        self.__input: inputs,
                        self.__label: labels,
                        self.__dropout: self.dropout_keep_prob,
                        self.__zoneout: self.zoneout_keep_prob
                    })

                    train_accuracy += accuracy / len(train_batches)
                    avg_train_loss += loss / len(train_batches)
                    time_elapsed = time.time() - start_time

                    if count % print_every == 0:
                        print("{:06d}: {} [{:05d}/{:05d}], train_loss = {:06.8f}, accuracy = {:06.8f}, secs/batch = {:.4f}".format(
                            step, epoch, count, len(train_batches), loss, accuracy, time_elapsed))

                print("Epoch training time:", time.time()-epoch_start_time)
            
                ''' evaluating '''
                avg_valid_loss = 0.0
                valid_accuracy = 0.0
                count = 0

                for batch in valid_batches:
                    inputs, labels = batch
                    count += 1
                    start_time = time.time()

                    loss, accuracy = session.run([
                        self.__total_loss,
                        self.__accuracy
                    ], {
                        self.__input: inputs,
                        self.__label: labels,
                        self.__dropout: 1.0,
                        self.__zoneout: 1.0
                    })

                    valid_accuracy += accuracy / len(valid_batches)
                    avg_valid_loss += loss / len(valid_batches)

                print("\nFinished Epoch {}".format(epoch))
                print("train_loss = {:06.8f}, train_accruacy = {:06.8f}".format(
                    avg_train_loss, train_accuracy))
                print("valid_loss = {:06.8f}, valid_accuracy = {:06.8f}\n".format(
                    avg_valid_loss, valid_accuracy))

                ''' save model '''
                saver.save(session, os.path.join(save_dir, 'epoch{:03d}_{:.4f}.model'.format(epoch, avg_valid_loss)))

                if should_write_summaries:
                    ''' save summary events '''
                    summary = tf.Summary(value=[
                        tf.Summary.Value(tag="train_loss", simple_value=avg_train_loss),
                        tf.Summary.Value(tag="valid_loss", simple_value=avg_valid_loss),
                        tf.Summary.Value(tag="train_accuracy", simple_value=train_accuracy),
                        tf.Summary.Value(tag="valid_accuracy", simple_value=valid_accuracy)
                    ])
                    summary_writer.add_summary(summary, step)
                
        if should_write_summaries:
            summary_writer.close()

    def test(self, test_data, load_dir):
        # fit to test data
        self.__fit_to_corpus(test_data, is_training=False)
        test_batches = self.__prepare_batches("test")

        with tf.Session(graph=self.__graph) as session:
            self.__load(session, load_dir)
            print("-"*80)
            print('Restored model from checkpoint for testing. Size:', _model_size())
            print("-"*80)

            ''' testing '''
            avg_test_loss = 0.0
            test_accuracy = 0.0
            count = 0

            start_time = time.time()
            for batch in test_batches:
                inputs, labels = batch
                count += 1

                loss, accuracy = session.run([
                    self.__total_loss,
                    self.__accuracy
                ], {
                    self.__input: inputs,
                    self.__label: labels,
                    self.__dropout: 1.0,
                    self.__zoneout: 1.0
                })

                avg_test_loss += loss / len(test_batches)
                test_accuracy += accuracy / len(test_batches)

            time_elapsed = time.time() - start_time

            print("test loss = {:06.8f}, test accuracy = {:06.8f}".format(avg_test_loss, test_accuracy))
            print("test samples: {:06d}, time elapsed: {:.4f}, time per one batch: {:.4f}".format(
                len(test_batches) * self.batch_size, time_elapsed, time_elapsed/count))

    def __prepare_batches(self, mode="train"):
        ''' mode = train/valid/test '''
        if mode == "train" and self.__train_list is None:
            raise NotFitToCorpusError(
                "Need to fit model to corpus before preparing training batches.")
        if mode == "valid" and self.__valid_list is None:
            raise NotFitToCorpusError(
                "Need to fit model to corpus before preparing valid batches.")
        if mode == "test" and self.__test_list is None:
            raise NotFitToCorpusError(
                "Need to fit model to corpus before preparing test batches.")

        if mode == "train":
            input_list = self.__train_list.copy()
        elif mode == "valid":
            input_list = self.__valid_list.copy()
        elif mode == "test":
            input_list = self.__test_list.copy()
        else:
            raise TypeError("mode should be 'train'/'valid'/'test'")
        
        reduced_length = (len(input_list) // self.batch_size) * self.batch_size
        input_list = input_list[:reduced_length]
        sentence_list, label_list = zip(*input_list)

        return list(_batchify(self.batch_size, sentence_list, label_list))

    def __load(self, session, load_dir):
        saver = tf.train.Saver(max_to_keep=5)
        ckpt = tf.train.get_checkpoint_state(load_dir)
        assert ckpt, "No checkpoint found"
        assert ckpt.model_checkpoint_path, "No model path found in checkpoint"
        saver.restore(session, ckpt.model_checkpoint_path)

    @property
    def vocab_size(self):
        return len(self.__words) + 1  # +1 is for padding.

    @property
    def words(self):
        if self.__words is None:
            raise NotFitToCorpusError("Need to fit model to corpus before accessing words.")
        return self.__words

    def id_for_word(self, word):
        if self.__word_to_id is None:
            raise NotFitToCorpusError("Need to fit model to corpus before looking up word ids")
        return self.__word_to_id[word]

def _batchify(batch_size, *sequences):
    for i in range(0, len(sequences[0]), batch_size):
        yield tuple(sequence[i:i+batch_size] for sequence in sequences)
                
def _model_size():
    ''' Calculates model size '''
    params = tf.trainable_variables()
    size = 0
    for x in params:
        sz = 1
        for dim in x.get_shape():
            sz *= dim.value
        size += sz
    return size
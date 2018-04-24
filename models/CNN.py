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

class CNN():
    def __init__(self, batch_size=50, dropout_keep_prob=0.5, learning_rate=0.001,
                 num_filter=100, filter_windows=[3,4,5], l2_constarint=3.):
        print("DEBUG: 04180000")
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout_keep_prob = dropout_keep_prob
        self.num_filter = num_filter
        self.filter_windows = filter_windows
        self.l2_constraint = l2_constarint
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

            # Define Embedding Matrices.
            word_embeddings_static = tf.get_variable(
                name = "word_embeddings_static",
                shape = [self.vocab_size, self.__word_embedding_size],
                initializer = tf.constant_initializer(self.__w2v, verify_shape=True),
                trainable = False)

            # Embedding Layer
            word_embedding_static = tf.nn.embedding_lookup(
                params = word_embeddings_static,
                ids = self.__input,
                name = "word_embedding_static")

            word_embedding_static_expanded = tf.expand_dims(
                input = word_embedding_static,
                axis = -1)

            # Convolution Layers
            pooled_outputs = []
            for filter_window in self.filter_windows:
                with tf.variable_scope("conv-maxpool-{}".format(filter_window)):
                    W = tf.get_variable(
                        name = "W",
                        shape = [filter_window, self.__word_embedding_size, 1, self.num_filter],
                        initializer = tf.contrib.layers.variance_scaling_initializer())

                    b = tf.get_variable(
                        name = "b",
                        shape = [self.num_filter],
                        initializer = tf.random_normal_initializer(stddev=0.1))

                    conv_static_1 = tf.nn.conv2d(
                        input = word_embedding_static_expanded,
                        filter = W,
                        strides = [1,1,1,1],
                        padding = "VALID",
                        name = "conv_static_2")

                    conv_static_2 = tf.nn.conv2d(
                        input = word_embedding_static_expanded,
                        filter = W,
                        strides = [1,1,1,1],
                        padding = "VALID",
                        name = "conv_static_1")  
                        # [batch_size, max_seq_length - filter_window + 1, 1, num_filter]

                    # Apply Nonlinearity and concatenate the results from the two filters
                    h_static_1 = tf.nn.relu(
                        features = tf.nn.bias_add(conv_static_1, b),
                        name = "h_static_1")
                    h_static_2 = tf.nn.relu(
                        features = tf.nn.bias_add(conv_static_2, b),
                        name = "h_static_2")
                
                    pooled_1 = tf.nn.max_pool(
                        value = h_static_1,
                        ksize = [1, self.__max_seq_length - filter_window + 1, 1, 1],
                        strides = [1,1,1,1],
                        padding = "VALID",
                        name = "pooled_1")  # [batch_size, 1, 1, filter_window]

                    pooled_2 = tf.nn.max_pool(
                        value = h_static_2,
                        ksize = [1, self.__max_seq_length - filter_window + 1, 1, 1],
                        strides = [1,1,1,1],
                        padding = "VALID",
                        name = "pooled_2")  # [batch_size, 1, 1, filter_window]
                    
                    pooled_outputs.append(pooled_1)
                    pooled_outputs.append(pooled_2)
            
            # penultimate layer
            feature_size = 2 * self.num_filter * len(self.filter_windows)

            h_pool = tf.concat(
                values = pooled_outputs, 
                axis = 3)

            h_pool_flat = tf.reshape(
                tensor = h_pool,
                shape = [self.batch_size, feature_size])
                
            h_drop = tf.nn.dropout(
                x = h_pool_flat,
                keep_prob = self.__dropout)

            # Fully Connected Layer
            W_fc = tf.get_variable(
                name = "W_fc",
                shape = [feature_size, self.__num_classes],
                initializer = tf.contrib.layers.variance_scaling_initializer())

            b_fc = tf.get_variable(
                name = "b_fc",
                shape = [self.__num_classes],
                initializer = tf.random_normal_initializer(stddev=0.1))

            # Constraint l2-norms of the weight vectors
            self.__l2_norm = tf.nn.l2_loss(W_fc) * 2
            self.__weight_rescale = tf.assign(
                ref = W_fc,
                value = tf.divide(tf.multiply(W_fc, self.l2_constraint), self.__l2_norm))

            logits = tf.nn.xw_plus_b(
                x = h_drop,
                weights = W_fc,
                biases = b_fc,
                name = "logits")

            # Optimize
            single_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits = logits,
                labels = self.__label)

            self.__total_loss = tf.reduce_mean(single_losses)

            self.__global_step = tf.get_variable(
                name = "global_step",
                shape = [],
                initializer = tf.constant_initializer(0),
                trainable = False,
                dtype = tf.int32)

            learning_rate = tf.train.exponential_decay(
                learning_rate = self.learning_rate, 
                global_step = self.__global_step,
                decay_steps = 500,
                decay_rate = 0.95,
                staircase = True)

            self.__optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
                loss = self.__total_loss,
                global_step = self.__global_step,
                name = "optimizer")

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

                    loss, step, accuracy, l2_norm, *_ = session.run([
                        self.__total_loss,
                        self.__global_step,
                        self.__accuracy,
                        self.__l2_norm,
                        self.__optimizer,
                    ], {
                        self.__input: inputs,
                        self.__label: labels,
                        self.__dropout: self.dropout_keep_prob
                    })

                    if l2_norm > self.l2_constraint:
                        session.run(self.__weight_rescale)

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
                        self.__dropout: 1.0
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
                    self.__dropout: 1.0
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
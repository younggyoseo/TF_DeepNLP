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

class QRNN_LM():
    def __init__(self, batch_size=20, dropout_keep_prob=0.5, learning_rate=1.0,
                 filter_windows=[2,2], l2_reg_lambda=2e-3, hidden_size=640,
                 num_unroll_steps=105, grad_clip=10.0, zoneout_keep_prob=0.9,
                 word_embedding_size=640):
        print("DEBUG: 04231523")
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout_keep_prob = dropout_keep_prob
        self.zoneout_keep_prob = zoneout_keep_prob
        self.filter_windows = filter_windows
        self.l2_reg_lambda = l2_reg_lambda
        self.hidden_size = hidden_size
        self.num_unroll_steps = num_unroll_steps
        self.grad_clip = grad_clip
        self.word_embedding_size = word_embedding_size
        self.__words = None
        self.__word_to_id = None
    
    def fit_to_corpus(self, train_data):
        self.__fit_to_corpus(train_data, is_training=True)
        self.__build_graph()

    def __fit_to_corpus(self, data, is_training):
        # TRAINING
        if is_training == True:
            train, dev, word_to_idx = data
            self.__words = list(word_to_idx.keys())
            self.__word_to_id = word_to_idx
            self.__train_list = train
            self.__valid_list = dev

        elif is_training == False:
            test = data[0]
            self.__test_list = test

    def __build_graph(self):
        self.__graph = tf.Graph()

        with self.__graph.as_default():
            self.__input = tf.placeholder(tf.int32,
                                          shape=[self.batch_size, self.num_unroll_steps])
            self.__target = tf.placeholder(tf.int32,
                                           shape=[self.batch_size, self.num_unroll_steps])
            self.__prev_c = tf.placeholder(tf.float32, 
                                            shape=[len(self.filter_windows), self.batch_size, self.hidden_size])
            self.__dropout = tf.placeholder(tf.float32)
            self.__zoneout = tf.placeholder(tf.float32)

            with tf.variable_scope("Embedding"):
                # Define Embedding Matrices.
                word_embeddings = tf.get_variable(
                    name = "word_embeddings",
                    shape = [self.vocab_size, self.word_embedding_size],
                    initializer = tf.random_uniform_initializer(-0.05, 0.05))

                self.__clear_word_embedding_padding = tf.scatter_update(word_embeddings, [0], tf.constant(0.0, shape=[1, self.word_embedding_size]))

                # Embedding Layer
                word_embedding= tf.nn.embedding_lookup(  
                    params = word_embeddings,
                    ids = self.__input)  # [batch_size, max_seq_length, word_embedding_size]

                # [batch_size, max_seq_length, word_embedding_size]
                word_embedding_drop = tf.nn.dropout(
                    x = word_embedding,
                    keep_prob = self.__dropout,
                    name = "word_embedding_drop")

            last_c = []
            # QRNN Layers
            for idx, filter_window in enumerate(self.filter_windows):
                if idx == 0:
                    input_ = word_embedding_drop

                prev_c = self.__prev_c[idx]
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
                            initializer = tf.random_uniform_initializer(-0.05, 0.05))

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
                            shape = [self.batch_size, self.num_unroll_steps, self.hidden_size])
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
                        
                        last_c.append(prev_c)
                        H_ = tf.concat([tf.expand_dims(x, axis=1) for x in h_list], axis=1)
                        H = tf.reshape(
                            tensor = H_,
                            shape = [self.batch_size, self.num_unroll_steps, self.hidden_size],
                            name = "H")
                            
                        input_ = tf.nn.dropout(
                            x = H,
                            keep_prob = self.__dropout,
                            name = "input_")

            self.__last_c = last_c

            with tf.variable_scope("Fully-Connected"):
                # [batch_size * max_seq_length, hidden_size]
                H_flat = tf.reshape(
                    tensor = H,
                    shape = [-1, self.hidden_size],
                    name = "H_flat")

                H_drop = tf.nn.dropout(
                    x = H_flat,
                    keep_prob = self.__dropout,
                    name = "H_drop")

                W_out = tf.get_variable(
                    name = "W_out",
                    shape = [self.hidden_size, self.vocab_size],
                    initializer = tf.random_uniform_initializer(-0.05, 0.05))

                b_out = tf.get_variable(
                    name = "b_out",
                    shape = [self.vocab_size],
                    initializer = tf.random_uniform_initializer(-0.05, 0.05))

                logits = tf.nn.xw_plus_b(
                    x = H_drop,
                    weights = W_out,
                    biases = b_out,
                    name = "logits")

                logits_reshaped = tf.reshape(
                    tensor = logits,
                    shape = [self.batch_size, self.num_unroll_steps, self.vocab_size],
                    name = "logits_reshaped")

            with tf.variable_scope("Loss"):
                # Loss
                single_losses = tf.contrib.seq2seq.sequence_loss(
                    logits = logits_reshaped, 
                    targets = self.__target,
                    weights = tf.ones_like(self.__target, dtype=tf.float32),
                    average_across_batch = False)

                l2_loss = tf.nn.l2_loss(W_out)

                __loss_scale_factor = tf.constant(
                    value = self.num_unroll_steps,
                    dtype = tf.float32)
                
                self.__report_loss = tf.reduce_mean(single_losses)
                self.__total_loss = __loss_scale_factor * tf.reduce_mean(single_losses) + self.l2_reg_lambda * l2_loss

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

                # gradient clipping
                tvars = tf.trainable_variables()
                grads, self.__grad_norm = tf.clip_by_global_norm(
                    t_list = tf.gradients(self.__total_loss, tvars),
                    clip_norm = self.grad_clip)

                self.__optimizer = tf.train.GradientDescentOptimizer(self.__learning_rate).apply_gradients(
                    grads_and_vars = zip(grads, tvars), 
                    global_step = self.__global_step, 
                    name='optimizer')

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
                count = 0
                prev_c = np.zeros([len(self.filter_windows), self.batch_size, self.hidden_size])

                for batch in train_batches:
                    inputs, targets = batch
                    count += 1
                    start_time = time.time()
            
                    loss, step, prev_c, grad_norm, *_ = session.run([
                        self.__report_loss,
                        self.__global_step,
                        self.__last_c,
                        self.__grad_norm,
                        self.__optimizer,
                        self.__clear_word_embedding_padding,
                    ], {
                        self.__input: inputs,
                        self.__target: targets,
                        self.__prev_c: prev_c,
                        self.__dropout: self.dropout_keep_prob,
                        self.__zoneout: self.zoneout_keep_prob
                    })

                    avg_train_loss += loss / len(train_batches)
                    time_elapsed = time.time() - start_time

                    if count % print_every == 0:
                        print("{:06d}: {} [{:05d}/{:05d}], train_loss/perplexity = {:06.8f}/{:06.7f}, secs/batch = {:.4f}, grad_norm={:.4f}".format(
                            step, epoch, count, len(train_batches), loss, np.exp(loss), time_elapsed, grad_norm))

                print("Epoch training time:", time.time()-epoch_start_time)
            
                ''' evaluating '''
                avg_valid_loss = 0.0
                count = 0
                prev_c = np.zeros([len(self.filter_windows), self.batch_size, self.hidden_size])

                for batch in valid_batches:
                    inputs, targets = batch
                    count += 1
                    start_time = time.time()

                    loss, prev_c = session.run([
                        self.__report_loss,
                        self.__last_c
                    ], {
                        self.__input: inputs,
                        self.__target: targets,
                        self.__prev_c: prev_c,
                        self.__dropout: 1.0,
                        self.__zoneout: 1.0
                    })

                    avg_valid_loss += loss / len(valid_batches)

                print("\nFinished Epoch {}".format(epoch))
                print("train_loss = {:06.8f}, perplexity = {:06.8f}".format(avg_train_loss, np.exp(avg_train_loss)))
                print("validation_loss = {:06.8f}, perplexity = {:06.8f}\n".format(avg_valid_loss, np.exp(avg_valid_loss)))

                ''' save model '''
                saver.save(session, os.path.join(save_dir, 'epoch{:03d}_{:.4f}.model'.format(epoch, avg_valid_loss)))

                if should_write_summaries:
                    ''' save summary events '''
                    summary = tf.Summary(value=[
                        tf.Summary.Value(tag="train_loss", simple_value=avg_train_loss),
                        tf.Summary.Value(tag="valid_loss", simple_value=avg_valid_loss),
                        tf.Summary.Value(tag="train_perplexity", simple_value=np.exp(avg_train_loss)),
                        tf.Summary.Value(tag="valid_perplexity", simple_value=np.exp(avg_valid_loss))
                    ])
                    summary_writer.add_summary(summary, step)

                if epoch >= 6:
                    current_learning_rate = session.run(self.__learning_rate)
                    current_learning_rate *= 0.95
                    session.run(self.__learning_rate.assign(current_learning_rate))
                
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
            count = 0
            prev_c = np.zeros([len(self.filter_windows), self.batch_size, self.hidden_size])

            start_time = time.time()
            for batch in test_batches:
                inputs, targets = batch
                count += 1

                loss, prev_c = session.run([
                    self.__report_loss,
                    self.__last_c
                ], {
                    self.__input: inputs,
                    self.__target: targets,
                    self.__prev_c: prev_c,
                    self.__dropout: 1.0,
                    self.__zoneout: 1.0
                })

                avg_test_loss += loss / len(test_batches)

            time_elapsed = time.time() - start_time

            print("test loss = {:06.8f}, perplexity = {:06.8f}".format(avg_test_loss, np.exp(avg_test_loss)))
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
        
        reduced_length = (len(input_list) // (self.batch_size * self.num_unroll_steps)) * self.batch_size * self.num_unroll_steps
        input_list = input_list[:reduced_length]
        
        target_list = np.zeros_like(input_list)
        target_list[:-1] = input_list[1:].copy()
        target_list[-1] = input_list[0].copy()

        input_list = input_list.reshape([self.batch_size, -1, self.num_unroll_steps])
        target_list = target_list.reshape([self.batch_size, -1, self.num_unroll_steps])

        input_list = np.transpose(input_list, axes=(1,0,2)).reshape(-1, self.num_unroll_steps)
        target_list = np.transpose(target_list, axes=(1,0,2)).reshape(-1, self.num_unroll_steps)

        return list(_batchify(self.batch_size, input_list, target_list))


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
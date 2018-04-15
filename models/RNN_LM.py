import tensorflow as tf
import numpy as np
import os
import time
import pickle
import random
from collections import Counter
from random import shuffle

class NotTrainedError(Exception):
    pass

class NotFitToCorpusError(Exception):
    pass

class RNN_LM():
    def __init__(self, word_embedding_size, hidden_size, num_unroll_steps=35, batch_size=128, cell="RNN", 
                 dropout_keep_prob=0.5, learning_rate=0.001):
        print("DEBUG: 04151140")
        self.word_embedding_size = word_embedding_size
        self.num_unroll_steps = num_unroll_steps
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout_keep_prob = dropout_keep_prob
        self.cell = cell
        self.__words = None
        self.__word_to_id = None
    
    def fit_to_corpus(self, train_corpus, valid_corpus):
        self.__fit_to_corpus(train_corpus, valid_corpus, is_training=True)
        self.__build_graph()

    def __fit_to_corpus(self, *corpus, is_training):
        corpus_list = list(corpus)
        # TRAINING
        if is_training == True:
            assert len(corpus_list) == 2, "Input should be train corpus and valid corpus."
            train_corpus, valid_corpus = corpus_list

            # train corpus
            word_list, word_counts = self.__corpus2wordlist(train_corpus, word_count=True)
            self.__words = ["<unk>", "<eos>"] + [word for word in word_counts if word not in {"<unk>", "<eos>"}]
            self.__word_to_id = {word: i for i, word in enumerate(self.__words)}
            self.__train_word_list = np.array([self.__word_to_id[word] if word in self.__word_to_id else 0 for word in word_list], dtype=np.int32)

            # valid corpus
            word_list = self.__corpus2wordlist(valid_corpus)
            self.__valid_word_list = np.array([self.__word_to_id[word] if word in self.__word_to_id else 0 for word in word_list], dtype=np.int32)

        # TESTING
        elif is_training == False:
            assert len(corpus_list) == 1, "Input should be test corpus."
            test_corpus = corpus_list[0]

            # test corpus
            word_list = self.__corpus2wordlist(test_corpus)
            self.__test_word_list = np.array([self.__word_to_id[word] if word in self.__word_to_id else 0 for word in word_list], dtype=np.int32)

    def __build_graph(self):
        self.__graph = tf.Graph()
        with self.__graph.as_default():
            self.__input = tf.placeholder(tf.int32, shape=[None, self.num_unroll_steps])
            self.__target = tf.placeholder(tf.int32, shape=[None, self.num_unroll_steps])
            self.__dropout = tf.placeholder(tf.float32, shape=[])

            word_embeddings = tf.get_variable("word_embeddings", [self.vocab_size, self.word_embedding_size],
                                              initializer = tf.random_uniform_initializer(-1.0, 1.0))
            W_output = tf.get_variable("W_output", [self.hidden_size, self.vocab_size],
                                            initializer = tf.contrib.layers.xavier_initializer())
            b_output = tf.get_variable("b_output", [self.vocab_size],
                                            initializer = tf.zeros_initializer())

            word_embedding = tf.nn.embedding_lookup(word_embeddings, self.__input)  # [batch_size, num_unroll_steps, word_embedding_size]

            if self.cell == "RNN":
                cell = tf.nn.rnn_cell.RNNCell(num_units=self.hidden_size)
            elif self.cell == "GRU":
                cell = tf.nn.rnn_cell.GRUCell(num_units=self.hidden_size)
            elif self.cell == "LSTM":
                cell = tf.nn.rnn_cell.LSTMCell(num_units=self.hidden_size)

            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.__dropout)

            tmp_outputs, _ = tf.nn.dynamic_rnn(cell, word_embedding, dtype=tf.float32)  # [batch_size, num_unroll_steps, hidden_size]
            outputs = tf.reshape(tmp_outputs, [-1, self.hidden_size], name="outputs")
            tmp_logits = tf.nn.xw_plus_b(outputs, W_output, b_output, name="tmp_logits")
            logits = tf.reshape(tmp_logits, [-1, self.num_unroll_steps, self.vocab_size], name="logits")
            self.__prediction = tf.argmax(logits, 2, output_type=tf.int32)

            single_losses = tf.contrib.seq2seq.sequence_loss(logits=logits, 
                                                             targets=self.__target,
                                                             weights=tf.ones_like(self.__target, dtype=tf.float32),
                                                             average_across_batch=False)

            self.__total_loss = tf.reduce_mean(single_losses)
            self.__global_step = tf.Variable(0, name="global_step", trainable=False)
            learning_rate = tf.train.exponential_decay(self.learning_rate, self.__global_step,
                                                           10000, 0.95, staircase=True)

            self.__optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
                self.__total_loss, global_step=self.__global_step)

    def train(self, num_epochs, save_dir, log_dir=None, load_dir=None, print_every=20):

        should_write_summaries = log_dir is not None
        with tf.Session(graph=self.__graph) as session:
            tf.set_random_seed(1004)
            random.seed(1004)
            if should_write_summaries:
                summary_writer = tf.summary.FileWriter(log_dir, graph=session.graph)
            if load_dir is not None:
                self.__load(session, load_dir)
                print('-'*80 + '\n' + 'Restored model from checkpoint. Size: {}\n'.format(_model_size()) + '-'*80)
            else:
                tf.global_variables_initializer().run()
                print('-'*80 + '\n' + 'Created and Initialized fresh model. Size: {}\n'.format(_model_size()) + '-'*80)

            saver = tf.train.Saver(max_to_keep=5)
            train_batches = self.__prepare_batches("train")
            valid_batches = self.__prepare_batches("valid")

            for epoch in range(1, num_epochs+1):
                epoch_start_time = time.time()
                avg_train_loss = 0.0
                total_cnt = 0
                count = 0
                shuffle(train_batches)
                for batch in train_batches:
                    inputs, targets = batch
                    total_cnt += len(inputs)
                    count += 1
                    start_time = time.time()

                    loss, step, _ = session.run([
                        self.__total_loss,
                        self.__global_step,
                        self.__optimizer
                    ], {
                        self.__input: inputs,
                        self.__target: targets,
                        self.__dropout: self.dropout_keep_prob
                    })
                    avg_train_loss += (loss * len(inputs))
                    time_elapsed = time.time() - start_time

                    if count % print_every == 0:
                        print("{:06d}: {} [{:05d}/{:05d}], train_loss/perplexity = {:06.8f}/{:06.7f} secs/batch = {:.4f}".format(
                            step, epoch, count, len(train_batches), loss, np.exp(loss), time_elapsed))

                avg_train_loss = (avg_train_loss / total_cnt)
                print("Epoch training time:", time.time()-epoch_start_time)
            
                ''' evaluating '''
                print("\nEvaluating..\n")
                avg_valid_loss = 0.0
                total_cnt = 0
                count = 0
                for batch in valid_batches:
                    inputs, targets = batch
                    total_cnt += len(inputs)
                    count += 1
                    start_time = time.time()
                    loss = session.run(
                        self.__total_loss
                    , {
                        self.__input: inputs,
                        self.__target: targets,
                        self.__dropout: 1.0
                    })
                    avg_valid_loss += (loss * len(inputs))

                    # if count % print_every == 0:
                    #     print("[{:05d}/{:05d}], validation loss = {:06.8f}, perflexity = {:06.8f}".format(
                    #         count, len(valid_batches), loss, np.exp(loss)))

                avg_valid_loss = (avg_valid_loss / total_cnt)
                print("Finished Epoch {}".format(epoch))
                print("train_loss = {:06.8f}, perflexity = {:06.8f}".format(avg_train_loss, np.exp(avg_train_loss)))
                print("validation_loss = {:06.8f}, perflexity = {:06.8f}\n".format(avg_valid_loss, np.exp(avg_valid_loss)))

                ''' save model '''
                saver.save(session, os.path.join(save_dir, 'epoch{:03d}_{:.4f}.model'.format(epoch, avg_valid_loss)))

                if should_write_summaries:
                    ''' save summary events '''
                    summary = tf.Summary(value=[
                        tf.Summary.Value(tag="train_loss", simple_value=avg_train_loss),
                        tf.Summary.Value(tag="valid_loss", simple_value=avg_valid_loss)
                    ])
                    summary_writer.add_summary(summary, step)
                
        if should_write_summaries:
            summary_writer.close()

    def test(self, test_corpus, load_dir):
        # fit to test data
        self.__fit_to_corpus(test_corpus, is_training=False)
        test_batches = self.__prepare_batches("test")

        with tf.Session(graph=self.__graph) as session:
            self.__load(session, load_dir)
            print("-"*80)
            print('Restored model from checkpoint for testing. Size:', _model_size())
            print("-"*80)

            ''' testing '''
            avg_test_loss = 0.0
            total_cnt = 0
            count = 0
            start_time = time.time()
            for batch in test_batches:
                inputs, targets = batch
                total_cnt += len(inputs)
                count += 1

                loss = session.run(
                    self.__total_loss
                , {
                    self.__input: inputs,
                    self.__target: targets,
                    self.__dropout: 1.0
                })

                avg_test_loss += (loss * len(inputs))

            avg_test_loss = avg_test_loss / total_cnt
            time_elapsed = time.time() - start_time

            print("test loss = {:06.8f}, perplexity = {:06.8f}".format(avg_test_loss, np.exp(avg_test_loss)))
            print("test samples: {:06d}, time elapsed: {:.4f}, time per one batch: {:.4f}".format(total_cnt, time_elapsed, time_elapsed/count))


    def __prepare_batches(self, mode="train"):
        ''' mode = train/valid/test '''
        if mode == "train" and self.__train_word_list is None:
            raise NotFitToCorpusError(
                "Need to fit model to corpus before preparing training batches.")
        if mode == "valid" and self.__valid_word_list is None:
            raise NotFitToCorpusError(
                "Need to fit model to corpus before preparing valid batches.")
        if mode == "test" and self.__test_word_list is None:
            raise NotFitToCorpusError(
                "Need to fit model to corpus before preparing test batches.")

        if mode == "train":
            input_list = self.__train_word_list.copy()
        elif mode == "valid":
            input_list = self.__valid_word_list.copy()
        elif mode == "test":
            input_list = self.__test_word_list.copy()
        else:
            raise TypeError("mode should be 'train'/'valid'/'test'")
            
        target_list = np.zeros_like(input_list)
        target_list[:-1] = input_list[1:].copy()
        target_list[-1] = input_list[0].copy()

        reduced_length = (len(target_list) // self.num_unroll_steps) * self.num_unroll_steps

        input_list = input_list[:reduced_length].reshape([-1, self.num_unroll_steps])
        target_list = target_list[:reduced_length].reshape([-1, self.num_unroll_steps])
        return list(_batchify(self.batch_size, input_list, target_list))

    def __load(self, session, load_dir):
        saver = tf.train.Saver(max_to_keep=5)
        ckpt = tf.train.get_checkpoint_state(load_dir)
        assert ckpt, "No checkpoint found"
        assert ckpt.model_checkpoint_path, "No model path found in checkpoint"
        saver.restore(session, ckpt.model_checkpoint_path)

    def __corpus2wordlist(self, corpus, word_count=False):
        word_list = []
        if word_count: 
            word_counts = Counter()
        for sequence in corpus:
            if word_count: word_counts.update(sequence)
            for word in sequence:
                word_list.append(word)
            word_list.append("<eos>")
        if word_count:
            return word_list, word_counts
        else:
            return word_list


    @property
    def vocab_size(self):
        return len(self.__words)

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
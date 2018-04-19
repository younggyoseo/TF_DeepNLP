import tensorflow as tf
import numpy as np
import os
import time
import pickle
from collections import Counter

class NotTrainedError(Exception):
    pass

class NotFitToCorpusError(Exception):
    pass

class CharRNN():
    def __init__(self, hidden_size, num_unroll_steps=35, batch_size=128, cell="RNN", 
                 dropout_keep_prob=0.5, learning_rate=0.001, grad_clip=5., num_layers=1):
        print("DEBUG: 04160104")
        self.num_unroll_steps = num_unroll_steps
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout_keep_prob = dropout_keep_prob
        self.grad_clip = grad_clip
        self.num_layers = num_layers
        self.cell = cell
        self.__chars = None
        self.__char_to_id = None
    
    def fit_to_corpus(self, corpus):
        self.__fit_to_corpus(corpus)
        self.__build_graph()

    def __fit_to_corpus(self, corpus):
        char_counts = Counter()
        for sequence in corpus:
            char_counts.update(sequence)
        count_pairs = sorted(char_counts.items(), key=lambda x: -x[1])
        self.__chars, _ = zip(*count_pairs)
        self.__char_to_id = {char: i for i, char in enumerate(self.__chars)}
        self.__train_char_list = np.array(
            [self.__char_to_id[ch] for sequence in corpus for ch in sequence], dtype=np.int32)
        
    def __build_graph(self, sampling=False):
        if sampling:
            self.__sampling_graph = tf.Graph()
            self.batch_size = 1
            self.num_unroll_steps = 1
        self.__graph = tf.Graph()
        graph = self.__graph if not sampling else self.__sampling_graph

        with graph.as_default():
            self.__input = tf.placeholder(tf.int32,
                                          shape=[self.batch_size, self.num_unroll_steps])
            self.__target = tf.placeholder(tf.int32,
                                           shape=[self.batch_size, self.num_unroll_steps])
            self.__dropout = tf.placeholder(tf.float32)

            W_output = tf.get_variable(
                name = "W_output",
                shape = [self.hidden_size, self.vocab_size],
                initializer = tf.contrib.layers.xavier_initializer())

            b_output = tf.get_variable(
                name = "b_output",
                shape = [self.vocab_size],
                initializer = tf.zeros_initializer())

            x_onehot = tf.one_hot(
                indices = self.__input,
                depth = self.vocab_size,
                on_value = 1.0,
                off_value = 0.0,
                name = "x_onehot")

            def make_cell():
                if self.cell == "RNN":
                    cell = tf.nn.rnn_cell.RNNCell(num_units=self.hidden_size)
                elif self.cell == "GRU":
                    cell = tf.nn.rnn_cell.GRUCell(num_units=self.hidden_size)
                elif self.cell == "LSTM":
                    cell = tf.nn.rnn_cell.LSTMCell(num_units=self.hidden_size)
                return cell
            
            cells = tf.contrib.rnn.MultiRNNCell(
                [tf.contrib.rnn.DropoutWrapper(
                    cell = make_cell(),
                    output_keep_prob = self.__dropout)
                for _ in range(self.num_layers)])

            self.__initial_state = cells.zero_state(
                    batch_size = self.batch_size,
                    dtype = tf.float32)

            # [batch_size, num_unroll_steps, hidden_size]
            outputs, self.__final_state = tf.nn.dynamic_rnn(
                cell = cells,
                inputs = x_onehot,
                initial_state = self.__initial_state,
                dtype = tf.float32)

            # [batch_size * num_unroll_steps, hidden_size]
            outputs_reshaped = tf.reshape(
                tensor = outputs, 
                shape = [-1, self.hidden_size], 
                name = "outputs_reshaped")  

            # [batch_size * num_unroll_steps, vocab_size]
            logits = tf.nn.xw_plus_b(
                x = outputs_reshaped,
                weights = W_output, 
                biases = b_output, 
                name = "logits")

            # probability for sampling
            self.__proba = tf.nn.softmax(
                logits = logits,
                name = "proba")

            # [batch_size, num_unroll_steps, vocab_size]
            logits_reshaped = tf.reshape(
                tensor = logits, 
                shape = [-1, self.num_unroll_steps, self.vocab_size],
                name = "logits_reshaped")  

            self.__prediction = tf.argmax(
                input = logits_reshaped,
                axis = 2, 
                output_type=tf.int32)

            single_losses = tf.contrib.seq2seq.sequence_loss(
                logits = logits_reshaped, 
                targets = self.__target,
                weights = tf.ones_like(self.__target, dtype=tf.float32),
                average_across_batch = False)

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
                decay_steps = 10000,
                decay_rate = 0.95,
                staircase = True)

            # gradient clipping
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(
                t_list = tf.gradients(self.__total_loss, tvars),
                clip_norm = self.grad_clip)

            self.__optimizer = tf.train.AdamOptimizer(learning_rate).apply_gradients(
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
            train_batches = self.__prepare_batches()

            for epoch in range(1, num_epochs+1):
                epoch_start_time = time.time()
                avg_train_loss = 0.0
                count = 0
                new_state = session.run(self.__initial_state)

                for batch in train_batches:
                    inputs, targets = batch
                    count += 1
                    start_time = time.time()

                    loss, step, new_state, _ = session.run([
                        self.__total_loss,
                        self.__global_step,
                        self.__final_state,
                        self.__optimizer
                    ], {
                        self.__input: inputs,
                        self.__target: targets,
                        self.__initial_state: new_state,
                        self.__dropout: self.dropout_keep_prob
                    })

                    avg_train_loss += loss / len(train_batches)
                    time_elapsed = time.time() - start_time

                    if count % print_every == 0:
                        print("{:06d}: {} [{:05d}/{:05d}], train_loss/perplexity = {:06.8f}/{:06.7f} secs/batch = {:.4f}".format(
                            step, epoch, count, len(train_batches), loss, np.exp(loss), time_elapsed))

                print("Epoch training time:", time.time()-epoch_start_time)

                print("Finished Epoch {}".format(epoch))
                print("train_loss = {:06.8f}, perflexity = {:06.8f}".format(avg_train_loss, np.exp(avg_train_loss)))

                ''' save model '''
                saver.save(session, os.path.join(save_dir, 'epoch{:03d}_{:.4f}.model'.format(epoch, avg_train_loss)))

                if should_write_summaries:
                    ''' save summary events '''
                    summary = tf.Summary(value=[
                        tf.Summary.Value(tag="train_loss", simple_value=avg_train_loss),
                        tf.Summary.Value(tag="train_perplexity", simple_value=np.exp(avg_train_loss))
                    ])
                    summary_writer.add_summary(summary, step)
                
        if should_write_summaries:
            summary_writer.close()

    def sample(self, output_length, load_dir, starter_seq="The "):
        # build graph for sampling
        self.__build_graph(sampling=True)

        with tf.Session(graph=self.__sampling_graph) as session:
            observed_seq = [ch for ch in starter_seq]
            # load the parameters
            self.__load(session, load_dir)
            
            # run the model using the starter word
            new_state = session.run(self.__initial_state)
            for ch in starter_seq:
                x = np.zeros((1,1))
                x[0,0] = self.__char_to_id[ch]
                proba, new_state = session.run([
                    self.__proba,
                    self.__final_state
                ], {
                    self.__input : x,
                    self.__initial_state: new_state,
                    self.__dropout : 1.0
                })

            char_id = self.__get_top_char(proba, self.vocab_size)
            observed_seq.append(self.__chars[char_id])

            for _ in range(output_length):
                x[0,0] = char_id
                proba, new_state = session.run([
                    self.__proba,
                    self.__final_state
                ], {
                    self.__input : x,
                    self.__initial_state: new_state,
                    self.__dropout : 1.0
                })
                char_id = self.__get_top_char(proba, self.vocab_size)
                observed_seq.append(self.__chars[char_id])

        return ''.join(observed_seq)


    def __prepare_batches(self):
        if self.__train_char_list is None:
            raise NotFitToCorpusError(
                "Need to fit model to corpus before preparing training batches.")
        input_list = self.__train_char_list.copy()
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

    def __get_top_char(self, proba, vocab_size, top_n=5):
        p = np.squeeze(proba)
        p[np.argsort(p)[:-top_n]] = 0.0
        p = p / np.sum(p)
        char_id = np.random.choice(vocab_size, 1, p=p)[0]
        return char_id

    @property
    def vocab_size(self):
        return len(self.__chars)

    @property
    def chars(self):
        if self.__chars is None:
            raise NotFitToCorpusError("Need to fit model to corpus before accessing words.")
        return self.__chars

    def id_for_char(self, char):
        if self.__char_to_id is None:
            raise NotFitToCorpusError("Need to fit model to corpus before looking up word ids")
        return self.__char_to_id[char]

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
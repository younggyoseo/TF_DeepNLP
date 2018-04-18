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

class WordRNN():
    def __init__(self, word_embedding_size, hidden_size, num_unroll_steps=35, batch_size=128, cell="RNN", 
                 dropout_keep_prob=0.5, learning_rate=0.001, grad_clip=5., num_layers=1):
        print("DEBUG: 04152210")
        self.word_embedding_size = word_embedding_size
        self.num_unroll_steps = num_unroll_steps
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout_keep_prob = dropout_keep_prob
        self.grad_clip = grad_clip
        self.num_layers = num_layers
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
            self.__train_word_list = np.array(
                [self.__word_to_id[word] if word in self.__word_to_id else 0 for word in word_list], dtype=np.int32)

            # valid corpus
            word_list = self.__corpus2wordlist(valid_corpus)
            self.__valid_word_list = np.array(
                [self.__word_to_id[word] if word in self.__word_to_id else 0 for word in word_list], dtype=np.int32)

        # TESTING
        elif is_training == False:
            assert len(corpus_list) == 1, "Input should be test corpus."
            test_corpus = corpus_list[0]

            # test corpus
            word_list = self.__corpus2wordlist(test_corpus)
            self.__test_word_list = np.array(
                [self.__word_to_id[word] if word in self.__word_to_id else 0 for word in word_list], dtype=np.int32)

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

            word_embeddings = tf.get_variable(
                name = "word_embeddings", 
                shape = [self.vocab_size, self.word_embedding_size],
                initializer = tf.truncated_normal_initializer(stddev=0.1))

            W_output = tf.get_variable(
                name = "W_output",
                shape = [self.hidden_size, self.vocab_size],
                initializer = tf.contrib.layers.xavier_initializer())

            b_output = tf.get_variable(
                name = "b_output",
                shape = [self.vocab_size],
                initializer = tf.zeros_initializer())

            word_embedding = tf.nn.embedding_lookup(  
                params = word_embeddings,
                ids = self.__input)  # [batch_size, num_unroll_steps, word_embedding_size]

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
                inputs = word_embedding,
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
            train_batches = self.__prepare_batches("train")
            valid_batches = self.__prepare_batches("valid")

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
            
                ''' evaluating '''
                print("\nEvaluating..\n")
                avg_valid_loss = 0.0
                count = 0
                new_state = session.run(self.__initial_state)

                for batch in valid_batches:
                    inputs, targets = batch
                    count += 1
                    start_time = time.time()

                    loss, new_state = session.run([
                        self.__total_loss,
                        self.__final_state
                    ], {
                        self.__input: inputs,
                        self.__target: targets,
                        self.__initial_state: new_state,
                        self.__dropout: 1.0
                    })

                    avg_valid_loss += loss / len(valid_batches)

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
            count = 0
            new_state = session.run(self.__initial_state)

            start_time = time.time()
            for batch in test_batches:
                inputs, targets = batch
                count += 1

                loss, new_state = session.run([
                    self.__total_loss,
                    self.__final_state
                ], {
                    self.__input: inputs,
                    self.__target: targets,
                    self.__initial_state: new_state,
                    self.__dropout: 1.0
                })

                avg_test_loss += loss / len(test_batches)

            time_elapsed = time.time() - start_time

            print("test loss = {:06.8f}, perplexity = {:06.8f}".format(avg_test_loss, np.exp(avg_test_loss)))
            print("test samples: {:06d}, time elapsed: {:.4f}, time per one batch: {:.4f}".format(
                len(test_batches) * self.batch_size, time_elapsed, time_elapsed/count))

    def sample(self, output_length, load_dir, starter_word="the"):
        # build graph for sampling
        self.__build_graph(sampling=True)

        with tf.Session(graph=self.__sampling_graph) as session:
            observed_seq = [starter_word]
            # load the parameters
            self.__load(session, load_dir)
            
            # run the model using the starter word
            new_state = session.run(self.__initial_state)
            x = np.zeros((1,1))
            x[0,0] = self.__word_to_id[observed_seq[0]]
            proba, new_state = session.run([
                self.__proba,
                self.__final_state
            ], {
                self.__input : x,
                self.__initial_state: new_state,
                self.__dropout : 1.0
            })

            word_id = self.__get_top_word(proba, self.vocab_size)
            observed_seq.append(self.__words[word_id])

            for _ in range(output_length):
                x[0,0] = word_id
                proba, new_state = session.run([
                    self.__proba,
                    self.__final_state
                ], {
                    self.__input : x,
                    self.__initial_state: new_state,
                    self.__dropout : 1.0
                })
                word_id = self.__get_top_word(proba, self.vocab_size)
                observed_seq.append(self.__words[word_id])

        return ' '.join(observed_seq)


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
        
        reduced_length = (len(input_list) // (self.batch_size * self.num_unroll_steps)) * self.batch_size * self.num_unroll_steps
        input_list = input_list[:reduced_length]
        
        target_list = np.zeros_like(input_list)
        target_list[:-1] = input_list[1:].copy()
        target_list[-1] = input_list[0].copy()

        input_list = input_list.reshape([-1, self.num_unroll_steps])
        target_list = target_list.reshape([-1, self.num_unroll_steps])
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

    def __get_top_word(self, proba, vocab_size, top_n=5):
        p = np.squeeze(proba)
        # make "<unk>" and <"eos"> do not appear in sampled sentence for "pretty" sentence.
        p[0] = 0.0
        p[1] = 0.0
        p[np.argsort(p)[:-top_n]] = 0.0
        p = p / np.sum(p)
        word_id = np.random.choice(vocab_size, 1, p=p)[0]
        return word_id

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
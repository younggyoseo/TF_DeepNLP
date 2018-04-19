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

class LSTMCharCNN():
    def __init__(self, hidden_size=300, num_unroll_steps=35, batch_size=20,  grad_clip=5.,
                 dropout_keep_prob=0.5, learning_rate=1.0, lstm_num_layer=2, highway_num_layer=1,
                 char_embedding_size=15, filter_windows=[1,2,3,4,5,6],
                 num_filters=[25,50,75,100,125,150]):
        self.hidden_size = hidden_size
        self.num_unroll_steps = num_unroll_steps
        self.batch_size = batch_size
        self.grad_clip = grad_clip
        self.dropout_keep_prob = dropout_keep_prob
        self.learning_rate = learning_rate
        self.lstm_num_layer = lstm_num_layer
        self.highway_num_layer = highway_num_layer
        self.char_embedding_size = char_embedding_size
        self.filter_windows = filter_windows
        self.num_filters = num_filters

        self.__chars = None
        self.__char_to_id = None
        self.__words = None
        self.__word_to_id = None
    
    def fit_to_corpus(self, corpus):
        self.__fit_to_corpus(corpus, True)
        self.__build_graph()

    def __fit_to_corpus(self, corpus, is_training):
        # TRAINING
        if is_training == True:
            train_word, valid_word, train_char, valid_char, word_to_idx, char_to_idx, max_word_length = corpus
            self.__words = list(word_to_idx.keys())
            self.__word_to_id = word_to_idx
            self.__chars = list(char_to_idx.keys())
            self.__char_to_id = char_to_idx
            self.__train_list = list(zip(train_word, train_char))
            self.__valid_list = list(zip(valid_word, valid_char))
            self.__max_word_length = max_word_length

        else:
            test_word, test_char = data
            self.__test_list = list(zip(test_word, test_char))

    def __build_graph(self, sampling=False):
        if sampling:
            self.__sampling_graph = tf.Graph()
            self.batch_size = 1
            self.num_unroll_steps = 1
        self.__graph = tf.Graph()
        graph = self.__graph if not sampling else self.__sampling_graph

        with graph.as_default():
            self.__chars_input = tf.placeholder(tf.int32, 
                shape = [self.batch_size, self.num_unroll_steps, self.__max_word_length])
            self.__words_input = tf.placeholder(tf.int32,
                shape=[self.batch_size, self.num_unroll_steps])
            self.__dropout = tf.placeholder(tf.float32, shape=[])

            with tf.variable_scope("embedding"):
                char_embeddings = tf.get_variable(
                    name = "char_embeddings",
                    shape = [self.char_vocab_size, self.char_embedding_size],
                    initializer = tf.random_uniform_initializer(-0.05, 0.05))

                # this op and comments are directly borrowed from https://github.com/mkroutikov/tf-lstm-char-cnn/blob/master/model.py
                '''this op clears embedding vector of first symbol (symbol at position 0, which is by convention the position
                of the padding symbol). It can be used to mimic Torch7 embedding operator that keeps padding mapped to
                zero embedding vector and ignores gradient updates. For that do the following in TF:
                1. after parameter initialization, apply this op to zero out padding embedding vector
                2. after each gradient update, apply this op to keep padding at zero'''
                self.__clear_char_embedding_padding = tf.scatter_update(char_embeddings, [0], tf.constant(0.0, shape=[1, self.char_embedding_size]))
                    
                # [batch_size, num_unroll_steps, max_word_length, char_embedding_size]
                char_embedding = tf.nn.embedding_lookup(
                    params = char_embeddings,
                    ids = self.__chars_input,
                    name = "char_embedding")

                # [batch_size * num_unroll_steps, max_word_length, char_embedding_size]
                char_embedding_reshaped = tf.reshape(
                    tensor = char_embedding,
                    shape = [-1, self.__max_word_length, self.char_embedding_size],
                    name = "char_embedding_reshaped")

                char_embedding_expanded = tf.expand_dims(
                    input = char_embedding_reshaped,
                    axis = 1)

            # Convolution Layer
            pooled_outputs = []
            with tf.variable_scope("Convolution"):
                for filter_window, num_filter in zip(self.filter_windows, self.num_filters):
                    with tf.variable_scope("conv-maxpool-{}".format(filter_window)):
                        reduced_length = self.__max_word_length - filter_window + 1
                        W = tf.get_variable(
                            name = "W",
                            shape = [1, filter_window, self.char_embedding_size, num_filter],
                            initializer = tf.random_uniform_initializer(-0.05, 0.05))
                        
                        b = tf.get_variable(
                            name = "b",
                            shape = [num_filter],
                            initializer = tf.random_uniform_initializer(-0.05, 0.05))

                        # [batch_size*num_unroll_steps, 1, max_word_length - filter_window + 1, num_filter]
                        conv = tf.nn.conv2d(
                            input = char_embedding_expanded,
                            filter = W,
                            strides = [1,1,1,1],
                            padding = "VALID",
                            name = "conv")

                        h = tf.nn.tanh(
                            x = conv + b,
                            name = "h")

                        # [batch_size * num_unroll_steps, 1, 1, num_filter]
                        pooled = tf.nn.max_pool(
                            value = h,
                            ksize = [1, 1, reduced_length, 1],
                            strides = [1,1,1,1],
                            padding="VALID",
                            name = "pool")

                        pooled_outputs.append(tf.squeeze(pooled, [1,2]))
            
            feature_size = np.sum(self.num_filters)
            # [batch_size * num_unroll_steps, feature_size]
            h_pool = tf.concat(
                values = pooled_outputs,
                axis = 1)

            # Highway Network
            for i in range(1, self.highway_num_layer+1):
                with tf.variable_scope("highway-{}".format(i)):
                    if i == 1:
                        _input = h_pool

                    W_h = tf.get_variable(
                        name = "W_h",
                        shape = [feature_size, feature_size],
                        initializer = tf.random_uniform_initializer(-0.05, 0.05))

                    W_t = tf.get_variable(
                        name = "W_t",
                        shape = [feature_size, feature_size],
                        initializer = tf.random_uniform_initializer(-0.05, 0.05))

                    b_h = tf.get_variable(
                        name = "b_h",
                        shape = [feature_size],
                        initializer = tf.random_uniform_initializer(-0.05, 0.05))
                    
                    b_t = tf.get_variable(
                        name = "b_t",
                        shape = [feature_size],
                        initializer = tf.random_uniform_initializer(-2.05, -1.95),
                        trainable = False)

                    g = tf.nn.relu(
                        features = tf.matmul(_input, W_h) + b_h,
                        name = "g")
                    
                    # transform gate
                    t = tf.sigmoid(
                        x = tf.matmul(_input, W_t) + b_t,
                        name = "t")

                    _input = t * g + (1. - t) * _input
                    z = _input

            # LSTM Layer
            def make_cell():
                cell = tf.nn.rnn_cell.BasicLSTMCell(
                    num_units=self.hidden_size,
                    forget_bias=0.0,
                    reuse=False)
                cell = tf.contrib.rnn.DropoutWrapper(
                    cell = cell,
                    output_keep_prob = self.__dropout)
                return cell

            with tf.variable_scope("LSTM"):
                cells = tf.contrib.rnn.MultiRNNCell(
                    [make_cell() for _ in range(self.lstm_num_layer)])

                self.__initial_state = cells.zero_state(
                    batch_size = self.batch_size,
                    dtype = tf.float32)

                z_reshaped = tf.reshape(
                    tensor = z,
                    shape = [self.batch_size, self.num_unroll_steps, feature_size])
                
                outputs, self.__final_state = tf.nn.dynamic_rnn(
                    cell = cells,
                    inputs = z_reshaped,
                    initial_state = self.__initial_state,
                    dtype = tf.float32)

            with tf.variable_scope("loss"):
                # [batch_size * num_unroll_steps, hidden_size]
                outputs_reshaped = tf.reshape(
                    tensor = outputs, 
                    shape = [-1, self.hidden_size], 
                    name = "outputs_reshaped")

                W_output = tf.get_variable(
                    name = "W_output",
                    shape = [self.hidden_size, self.word_vocab_size],
                    initializer = tf.random_uniform_initializer(-0.05, 0.05))

                b_output = tf.get_variable(
                    name = "b_output",
                    shape = [self.word_vocab_size],
                    initializer = tf.random_uniform_initializer(-0.05, 0.05))

                # [batch_size * num_unroll_steps, word_vocab_size]
                logits = tf.matmul(outputs_reshaped, W_output) + b_output

                # probability for sampling
                self.__proba = tf.nn.softmax(
                    logits = logits,
                    name = "proba")

                # [batch_size, num_unroll_steps, word_vocab_size]
                logits_reshaped = tf.reshape(
                    tensor = logits, 
                    shape = [self.batch_size, self.num_unroll_steps, self.word_vocab_size],
                    name = "logits_reshaped")  

                single_losses = tf.contrib.seq2seq.sequence_loss(
                    logits = logits_reshaped, 
                    targets = self.__words_input,
                    weights = tf.ones_like(self.__words_input, dtype=tf.float32),
                    average_across_batch = False)

                # According to https://github.com/mkroutikov/tf-lstm-char-cnn, We should scale the loss in train time
                # in order to replicate the training process of torch7. I'm don't know how exactly this works,
                # I tried to followed his instruction.
                self.__loss_scale_factor = tf.constant(
                    value = self.num_unroll_steps,
                    dtype = tf.float32)

                self.__total_loss = tf.reduce_mean(single_losses) * self.__loss_scale_factor

            with tf.variable_scope("optimize"):
                self.__global_step = tf.get_variable(
                    name = "global_step",
                    shape = [],
                    initializer = tf.constant_initializer(0),
                    trainable = False,
                    dtype = tf.int32)

                # gradient clipping
                tvars = tf.trainable_variables()
                grads, self.__grad_norm = tf.clip_by_global_norm(
                    t_list = tf.gradients(self.__total_loss, tvars),
                    clip_norm = self.grad_clip)

                self.__learning_rate = tf.get_variable(
                    name = "learning_rate",
                    shape = [],
                    initializer = tf.constant_initializer(self.learning_rate),
                    trainable = False)

                self.__optimizer = tf.train.GradientDescentOptimizer(self.__learning_rate).apply_gradients(
                    grads_and_vars = zip(grads, tvars), 
                    global_step = self.__global_step, 
                    name='optimizer')

    def train(self, num_epochs, save_dir, log_dir=None, load_dir=None, print_every=20, new_learning_rate=None):
        should_write_summaries = log_dir is not None
        with tf.Session(graph=self.__graph) as session:

            if should_write_summaries:
                summary_writer = tf.summary.FileWriter(log_dir, graph=session.graph)
            if load_dir is not None:
                self.__load(session, load_dir)
                print('-' * 80)
                print('Restored model from checkpoint. Size: {}'.format(_model_size()))
                print('-' * 80)
            else:
                tf.global_variables_initializer().run()
                session.run(self.__clear_char_embedding_padding)
                print('-' * 80)
                print('Created and Initialized fresh model. Size: {}'.format(_model_size()))
                print('-' * 80)
            
            if new_learning_rate is not None and load_dir is not None:
                session.run(self.__learning_rate.assign(new_learning_rate))

            saver = tf.train.Saver(max_to_keep=5)
            train_batches = self.__prepare_batches("train")
            valid_batches = self.__prepare_batches("valid")
            best_valid_loss = None

            for epoch in range(1, num_epochs+1):
                epoch_start_time = time.time()
                avg_train_loss = 0.0
                count = 0
                new_state = session.run(self.__initial_state)
                
                for batch in train_batches:
                    chars, words = batch
                    count += 1
                    start_time = time.time()

                    loss, _, new_state, grad_norm, step, _ = session.run([
                        self.__total_loss,
                        self.__optimizer,
                        self.__final_state,
                        self.__grad_norm,
                        self.__global_step,
                        self.__clear_char_embedding_padding,
                    ], {
                        self.__chars_input: chars,
                        self.__words_input: words,
                        self.__initial_state: new_state,
                        self.__dropout: self.dropout_keep_prob
                    })

                    # we should rescale loss by scale_loss_factor(=num_unroll_steps)
                    loss = loss / self.num_unroll_steps
                    avg_train_loss += 0.05 * (loss - avg_train_loss)
                    time_elapsed = time.time() - start_time

                    if count % print_every == 0:
                        print("{:06d}: {} [{:05d}/{:05d}], train_loss/perflexity = {:06.8f}/{:06.7f}, secs/batch = {:.4f}, grad.norm={:06.8f}".format(
                            step, epoch, count, len(train_batches), loss, np.exp(loss), time_elapsed, grad_norm))

                print("Epoch training time:", time.time()-epoch_start_time)
            
                ''' evaluating '''
                avg_valid_loss = 0.0
                count = 0
                new_state = session.run(self.__initial_state)

                for batch in valid_batches:
                    chars, words = batch
                    count += 1
                    start_time = time.time()

                    loss, new_state = session.run([
                        self.__total_loss,
                        self.__final_state,
                    ], {
                        self.__chars_input: chars,
                        self.__words_input: words,
                        self.__initial_state: new_state,
                        self.__dropout: 1.0
                    })

                    # we should rescale loss by scale_loss_factor(=num_unroll_steps)
                    loss = loss / self.num_unroll_steps
                    avg_valid_loss += loss / len(valid_batches)

                print("\nFinished Epoch {}".format(epoch))
                print("train_loss = {:06.8f}, perplexity = {:06.7f}".format(
                    avg_train_loss, np.exp(avg_train_loss)))
                print("valid_loss = {:06.8f}, perplexity = {:06.7f}\n".format(
                    avg_valid_loss, np.exp(avg_valid_loss)))

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

                
                if best_valid_loss is not None and np.exp(avg_valid_loss) > np.exp(best_valid_loss) - 1.0:
                    print('decaying learning rate by half..')
                    current_learning_rate = session.run(self.__learning_rate)
                    print('learning rate was:', current_learning_rate)
                    current_learning_rate = current_learning_rate * 0.5
                    if current_learning_rate < 1e-5:
                        print('learning_rate too small - stopping now')
                        break

                    session.run(self.__learning_rate.assign(current_learning_rate))
                    print('new learning rate is:', current_learning_rate)
                else:
                    best_valid_loss = avg_valid_loss
                
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
            new_state = session.run(self.__initial_state)

            start_time = time.time()
            for batch in test_batches:
                chars, words = batch
                count += 1

                loss, new_state = session.run([
                    self.__total_loss,
                    self.__final_state
                ], {
                    self.__chars_input : chars,
                    self.__words_input : words,
                    self.__initial_state : new_state,
                    self.__dropout : 1.0
                })

                # we should rescale loss by scale_loss_factor(=num_unroll_steps)
                loss = loss / self.num_unroll_steps
                avg_test_loss += loss / len(test_batches)

            time_elapsed = time.time() - start_time

            print("test loss = {:06.8f}, perplexity = {:06.7f}".format(avg_test_loss, np.exp(avg_test_loss)))
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
            x = np.zeros((1,1,self.__max_word_length), dtype=np.int32)
            char_array = [self.__char_to_id[char] for char in '{' + starter_word + '}']
            x[0,0,:len(char_array)] = char_array

            new_state = session.run(self.__initial_state)
            proba, new_state = session.run([
                self.__proba,
                self.__final_state
            ], {
                self.__chars_input : x,
                self.__initial_state: new_state,
                self.__dropout : 1.0
            })

            word_id = self.__get_top_word(proba, self.word_vocab_size)
            observed_seq.append(self.__words[word_id])

            for _ in range(output_length):
                x = np.zeros((1,1,self.__max_word_length), dtype=np.int32)
                char_array = [self.__char_to_id[char] for char in '{' + self.__words[word_id] + '}']
                x[0,0,:len(char_array)] = char_array

                proba, new_state = session.run([
                    self.__proba,
                    self.__final_state
                ], {
                    self.__chars_input : x,
                    self.__initial_state: new_state,
                    self.__dropout : 1.0
                })
                word_id = self.__get_top_word(proba, self.word_vocab_size)
                observed_seq.append(self.__words[word_id])

        return ' '.join(observed_seq)

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
        word_list, char_list = zip(*input_list)

        target_word_list = np.zeros_like(word_list)
        target_word_list[:-1] = word_list[1:]
        target_word_list[-1] = word_list[0]

        char_list = np.reshape(char_list, [self.batch_size, -1, self.num_unroll_steps, self.__max_word_length])
        target_word_list = np.reshape(target_word_list, [self.batch_size, -1, self.num_unroll_steps])

        char_list = np.transpose(char_list, axes=(1,0,2,3)).reshape(-1, self.num_unroll_steps)
        target_word_list = np.transpose(target_word_list, axes=(1,0,2)).reshape(-1, self.num_unroll_steps)
        
        return list(_batchify(self.batch_size, char_list, target_word_list))

    def __load(self, session, load_dir):
        saver = tf.train.Saver(max_to_keep=5)
        ckpt = tf.train.get_checkpoint_state(load_dir)
        assert ckpt, "No checkpoint found"
        assert ckpt.model_checkpoint_path, "No model path found in checkpoint"
        saver.restore(session, ckpt.model_checkpoint_path)

    def __get_top_word(self, proba, vocab_size, top_n=5):
        p = np.squeeze(proba)
        p[0] = 0.0
        p[np.argsort(p)[:-top_n]] = 0.0
        p = p / np.sum(p)
        word_id = np.random.choice(vocab_size, 1, p=p)[0]
        return word_id

    @property
    def word_vocab_size(self):
        return len(self.words)

    @property
    def char_vocab_size(self):
        return len(self.chars)

    @property
    def words(self):
        if self.__words is None:
            raise NotFitToCorpusError("Need to fit model to corpus before accessing words.")
        return self.__words

    @property
    def chars(self):
        if self.__chars is None:
            raise NotFitToCorpusError("Need to fit model to corpus before accessing chars.")
        return self.__chars

    def id_for_char(self, char):
        if self.__char_to_id is None:
            raise NotFitToCorpusError("Need to fit model to corpus before looking up word ids")
        return self.__char_to_id[char]

def _batchify(batch_size, *sequences):
    for i in range(0, len(sequences[0]), batch_size):
        yield tuple(np.squeeze(sequence[i:i+batch_size],0) for sequence in sequences)
                
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
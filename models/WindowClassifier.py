import tensorflow as tf
import numpy as np
import os
import time
import pickle
import random
from collections import Counter
from random import shuffle
from sklearn.metrics import classification_report

class NotTrainedError(Exception):
    pass

class NotFitToCorpusError(Exception):
    pass

class WindowClassifier():
    def __init__(self, word_embedding_size, window_size, hidden_size, batch_size=128,
                 learning_rate=0.001):
        print("DEBUG: 04132118")
        if isinstance(window_size, tuple):
            self.left_context, self.right_context = window_size
        elif isinstance(window_size, int):
            self.left_context = self.right_context = window_size
        else:
            raise ValueError("`context_size` should be an int or a tuple of two ints")

        self.word_embedding_size = word_embedding_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.__words = None
        self.__word_to_id = None
        self.__embeddings = None
    
    def fit_to_data(self, train_data, valid_data, is_training=True):
        self.__fit_to_data(train_data, valid_data, left_size=self.left_context, right_size=self.right_context, 
                           is_training=is_training)
        self.__build_graph()

    def __fit_to_data(self, *data, left_size, right_size, is_training):
        data_list = list(data)
        # TRAINING
        if is_training == True:
            assert len(data_list) == 2, "Input should be train data and valid data."
            train_data, valid_data = data_list

            # train data
            word_window_list, entity_list, word_counts = self.__data2list(train_data, left_size, right_size, word_count=True)
            self.__entities = ['B-LOC', 'B-MISC', 'B-ORG', 'B-PER', 'I-LOC', 'I-MISC', 'I-ORG', 'I-PER', 'O']
            self.__entity_to_id = {entity: i for i, entity in enumerate(self.__entities)}
            self.__words = ["<PAD>", "<UNK>"] + [word for word in word_counts]
            self.__word_to_id = {word: i for i, word in enumerate(self.__words)}

            self.__train_window_list = list(zip(*(
                [[self.__word_to_id[word] if word in self.__word_to_id else 1 for word in window]
                for window in word_window_list],
                [self.__entity_to_id[entity] for entity in entity_list])))

            # valid data
            word_window_list, entity_list = self.__data2list(valid_data, left_size, right_size)
            self.__valid_window_list = list(zip(*(
                [[self.__word_to_id[word] if word in self.__word_to_id else 1 for word in window]
                for window in word_window_list],
                [self.__entity_to_id[entity] for entity in entity_list])))

        # TESTING
        elif is_training == False:
            assert len(data_list) == 1, "Input should be test data."
            test_data = data_list[0]

            # test data
            word_window_list, entity_list = self.__data2list(test_data, left_size, right_size)
            self.__test_window_list = list(zip(*(
                [[self.__word_to_id[word] if word in self.__word_to_id else 1 for word in window]
                for window in word_window_list],
                [self.__entity_to_id[entity] for entity in entity_list])))

    def __build_graph(self):
        self.__graph = tf.Graph()
        with self.__graph.as_default():
            context_length = self.right_context + self.left_context
            concatenated_size = self.word_embedding_size * (context_length + 1)

            self.__word_input = tf.placeholder(tf.int32, shape=[None, context_length + 1])
            self.__entity_input = tf.placeholder(tf.int32, shape=[None])

            word_embeddings = tf.get_variable("word_embeddings", [self.vocab_size, self.word_embedding_size],
                                              initializer = tf.random_uniform_initializer(-1.0, 1.0))
            W_hidden = tf.get_variable("W_hidden", [concatenated_size, self.hidden_size],
                                            initializer = tf.contrib.layers.xavier_initializer())
            b_hidden = tf.get_variable("b_hidden", [self.hidden_size],
                                            initializer = tf.zeros_initializer())
            W_output = tf.get_variable("W_output", [self.hidden_size, self.entity_size],
                                            initializer = tf.contrib.layers.xavier_initializer())
            b_output = tf.get_variable("b_output", [self.entity_size],
                                            initializer = tf.zeros_initializer())

            # this op and comments are borrowed from https://github.com/mkroutikov/tf-lstm-char-cnn/blob/master/model.py
            # this op clears embedding vector of "<PAD>" and "<UNK>"
            # 1. after parameter initialization, apply this op to zero out padding embedding vector
            # 2. after each gradient update, apply this op to keep padding at zero
            self.__clear_word_embedding_padding = tf.scatter_update(
                word_embeddings, [0, 1], tf.constant(0.0, shape=[2, self.word_embedding_size]))

            embedding_list = [tf.nn.embedding_lookup(word_embeddings, self.__word_input[:, entry])
                              for entry in range(context_length+1)]
            concatenated_embedding = tf.concat(embedding_list, 1, "concatenated_embedding")

            hidden = tf.nn.xw_plus_b(concatenated_embedding, W_hidden, b_hidden, name="hidden")
            logits = tf.nn.xw_plus_b(hidden, W_output, b_output, name="logits")

            self.__prediction = tf.argmax(logits, 1, output_type=tf.int32)

            single_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                           labels=self.__entity_input)
            self.__total_loss = tf.reduce_mean(single_losses)
            tf.summary.scalar("NER_loss", self.__total_loss)
            self.__global_step = tf.Variable(0, name="global_step", trainable=False)
            learning_rate = tf.train.exponential_decay(self.learning_rate, self.__global_step,
                                                           10000, 0.95, staircase=True)

            self.__optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
                self.__total_loss, global_step=self.__global_step)

            self.__summary = tf.summary.merge_all()
            self.__word_embeddings = word_embeddings

    def train(self, num_epochs, save_dir, log_dir=None, load_dir=None, print_every=1000):
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
                session.run(self.__clear_word_embedding_padding)
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
                    words, entities = batch
                    total_cnt += len(words)
                    count += 1
                    start_time = time.time()

                    loss, step, *_ = session.run([
                        self.__total_loss,
                        self.__global_step,
                        self.__optimizer,
                        self.__clear_word_embedding_padding
                    ], {
                        self.__word_input: words,
                        self.__entity_input: entities
                    })
                    avg_train_loss += (loss * len(words))
                    time_elapsed = time.time() - start_time

                    if count % print_every == 0:
                        print("{:06d}: {} [{:05d}/{:05d}], train_loss = {:06.8f}, secs/batch = {:.4f}".format(
                            step, epoch, count, len(train_batches), loss, time_elapsed))

                avg_train_loss = (avg_train_loss / total_cnt)
                print("Epoch training time:", time.time()-epoch_start_time)

                ''' evaluating '''
                print("\nEvaluating..")
                avg_valid_loss = 0.0
                true_entity_list = []
                pred_entity_list = []
                total_cnt = 0
                count = 0
                for batch in valid_batches:
                    words, entities = batch
                    total_cnt += len(words)
                    count += 1
                    start_time = time.time()

                    loss, pred_entities = session.run([
                        self.__total_loss,
                        self.__prediction,
                    ], {
                        self.__word_input: words,
                        self.__entity_input: entities
                    })
                    avg_valid_loss += (loss * len(words))
                    true_entity_list.extend(entities)
                    pred_entity_list.extend(pred_entities)

                # print table
                target_names = self.__entities[:-1]  # remove "O" from targets.
                labels = list(range(len(target_names)))
                print(classification_report(true_entity_list, pred_entity_list, labels, target_names))

                avg_valid_loss = (avg_valid_loss / total_cnt)
                print("Finished Epoch {}".format(epoch))
                print("train_loss = {:06.8f}, validation_loss = {:06.8f}\n".format(avg_train_loss, avg_valid_loss))

                ''' save model '''
                saver.save(session, os.path.join(save_dir, 'epoch{:03d}_{:.4f}.model'.format(epoch, avg_valid_loss)))

                if should_write_summaries:
                    ''' save summary events '''
                    summary = tf.Summary(value=[
                        tf.Summary.Value(tag="train_loss", simple_value=avg_train_loss),
                        tf.Summary.Value(tag="valid_loss", simple_value=avg_valid_loss)
                    ])
                    summary_writer.add_summary(summary, step)

            self.__embeddings = self.__word_embeddings.eval()
            if should_write_summaries:
                summary_writer.close()

    def test(self, test_data, load_dir):
        # fit to test data
        self.__fit_to_data(test_data, left_size=self.left_context, right_size=self.right_context, is_training=False)
        test_batches = self.__prepare_batches("test")

        with tf.Session(graph=self.__graph) as session:
            self.__load(session, load_dir)
            print("-"*80)
            print('Restored model from checkpoint for testing. Size:', _model_size())
            print("-"*80)

            ''' testing '''
            true_entity_list = []
            pred_entity_list = []
            total_cnt = 0
            count = 0
            start_time = time.time()
            for batch in test_batches:
                words, entities = batch
                total_cnt += len(words)
                count += 1

                pred_entities = session.run(
                    self.__prediction
                , {
                    self.__word_input: words,
                    self.__entity_input: entities
                })

                true_entity_list.extend(entities)
                pred_entity_list.extend(pred_entities)

            time_elapsed = time.time() - start_time
            target_names = self.__entities[:-1]  # remove "O" from targets.
            labels = list(range(len(target_names)))

            print(classification_report(true_entity_list, pred_entity_list, labels, target_names))
            print("test samples: {:06d}, time elapsed: {:.4f}, time per one batch: {:.4f}".format(total_cnt, time_elapsed, time_elapsed/count))


    def embedding_for(self, word_str_or_id):
        if isinstance(word_str_or_id, str):
            return self.embeddings[self.__word_to_id[word_str_or_id]]
        elif isinstance(word_str_or_id, int):
            return self.embeddings[word_str_or_id]

    def __prepare_batches(self, mode="train"):
        ''' mode = train/valid/test '''
        if mode == "train" and self.__train_window_list is None:
            raise NotFitToCorpusError(
                "Need to fit model to data before preparing training batches.")
        if mode == "valid" and self.__valid_window_list is None:
            raise NotFitToCorpusError(
                "Need to fit model to data before preparing valid batches.")
        if mode == "test" and self.__test_window_list is None:
            raise NotFitToCorpusError(
                "Need to fit model to data before preparing test batches.")

        if mode == "train":
            window_list = self.__train_window_list
        elif mode == "valid":
            window_list = self.__valid_window_list
        elif mode == "test":
            window_list = self.__test_window_list
        else:
            raise TypeError("mode should be 'train'/'valid'/'test'")

        words, entities = zip(*window_list)
        return list(_batchify(self.batch_size, words, entities))

    def __data2list(self, data, left_size, right_size, word_count=False):
        if word_count: word_counts = Counter()
        word_window_list = []
        entity_list = []

        for word_region, entity_region in data:
            if word_count: word_counts.update(word_region)
            for l_context, word, r_context in _context_windows(word_region, left_size, right_size):
                l_list = ["<PAD>"] * (self.left_context - len(l_context)) + [word for word in l_context]
                r_list = [word for word in r_context] + ["<PAD>"] * (self.right_context - len(r_context))
                context_list = l_list + [word] + r_list
                word_window_list.append(context_list)
            
            for l_context, entity, r_context in _context_windows(entity_region ,left_size, right_size):
                entity_list.append(entity)
        
        if word_count:
            return word_window_list, entity_list, word_counts
        else:
            return word_window_list, entity_list

    def __load(self, session, load_dir):
        saver = tf.train.Saver(max_to_keep=5)
        ckpt = tf.train.get_checkpoint_state(load_dir)
        assert ckpt, "No checkpoint found"
        assert ckpt.model_checkpoint_path, "No model path found in checkpoint"
        saver.restore(session, ckpt.model_checkpoint_path)

    @property
    def vocab_size(self):
        return len(self.__words)

    @property
    def entity_size(self):
        return len(self.__entities)
    
    @property
    def words(self):
        if self.__words is None:
            raise NotFitToCorpusError("Need to fit model to corpus before accessing words.")
        return self.__words

    @property
    def embeddings(self):
        if self.__embeddings is None:
            raise NotTrainedError("Need to train model before accessing embeddings")
        return self.__embeddings
    
    def id_for_word(self, word):
        if self.__word_to_id is None:
            raise NotFitToCorpusError("Need to fit model to corpus before looking up word ids")
        return self.__word_to_id[word]
    
    def id_for_entity(self, entity):
        if self.__entity_to_id is None:
            raise NotFitToCorpusError("Need to fit model to corpus before looking up entity ids")
        return self.__entity_to_id[entity]

def _context_windows(region ,left_size, right_size):
    for i, word in enumerate(region):
        start_index = i - left_size
        end_index = i + right_size
        left_context = _window(region, start_index, i - 1)
        right_context = _window(region, i + 1, end_index)
        yield (left_context, word, right_context)

def _window(region, start_index, end_index):
    """
    Returns the list of words starting from `start_index`, going to `end_index`
    taken from region. If `start_index` is a negative number, or if `end_index`
    is greater than the index of the last word in region, this function will pad
    its return value with `NULL_WORD`.
    """
    last_index = len(region) + 1
    selected_tokens = region[max(start_index, 0):min(end_index, last_index) + 1]
    return selected_tokens

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
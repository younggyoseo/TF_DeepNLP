import tensorflow as tf
import numpy as np
import os
import time
import pickle
from collections import Counter
from random import shuffle
from sklearn.metrics import classification_report

class NotTrainedError(Exception):
    pass

class NotFitToCorpusError(Exception):
    pass

class WindowClassifier():
    def __init__(self, word_embedding_size, window_size, hidden_size, batch_size=128,
                 learning_rate=0.001, negative_sample_size=5):
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
        self.negative_sample_size = negative_sample_size
        self.__words = None
        self.__word_to_id = None
        self.__embeddings = None
    
    def fit_to_data(self, data, is_training=True):
        self.__fit_to_data(data, self.left_context, self.right_context, is_training)
        self.__build_graph()

    def __fit_to_data(self, data, left_size, right_size, is_training):
        word_counts = Counter()
        word_window_list = []
        entity_list = []

        for word_region, entity_region in data:
            word_counts.update(word_region)
            for l_context, word, r_context in _context_windows(word_region ,left_size, right_size):
                l_list = ["<PAD>"] * (self.left_context - len(l_context)) + [word for word in l_context]
                r_list = [word for word in r_context] + ["<PAD>"] * (self.right_context - len(r_context))
                context_list = l_list + [word] + r_list
                word_window_list.append(context_list)

            for l_context, entity, r_context in _context_windows(entity_region, left_size, right_size):
                entity_list.append(entity)

        if len(word_window_list) == 0:
            raise ValueError("No window data in corpus. Did you try to reuse a generator?")
        
        # In test phase, Do not change vocabulary.
        if is_training:
            self.__entities = ['B-LOC', 'B-MISC', 'B-ORG', 'B-PER', 'I-LOC', 'I-MISC', 'I-ORG', 'I-PER', 'O']
            self.__entity_to_id = {entity: i for i, entity in enumerate(self.__entities)}
            self.__words = ["<PAD>", "<UNK>"] + [word for word in word_counts]
            self.__word_to_id = {word: i for i, word in enumerate(self.__words)}

        # if word is not in vocabulary, label it as 1("<UNK>")
        self.__window_list = list(zip(*(
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

    def train(self, num_epochs, save_dir, log_dir=None, load_dir=None,
              summary_batch_interval=5000, saver_batch_interval=5000, print_every=1000):
        should_write_summaries = log_dir is not None and summary_batch_interval
        should_save_models = save_dir is not None and saver_batch_interval

        batches = self.__prepare_batches()
        config = tf.ConfigProto(allow_soft_placement = True)
        with tf.Session(graph=self.__graph, config=config) as session:
            if should_write_summaries:
                print("Writing TensorBoard summaries to {}".format(log_dir))
                summary_writer = tf.summary.FileWriter(log_dir, graph=session.graph)
            if should_save_models:
                print("Saving TensorFlow models to {}".format(save_dir))
                saver = tf.train.Saver(max_to_keep=5)

            if load_dir is not None:
                self.__load(session, load_dir)
                print("-"*80)
                print('Restored model from checkpoint. Size:', _model_size())
                print('Total number of batches:', len(batches))
                print("-"*80)
            else:
                tf.global_variables_initializer().run()
                session.run(self.__clear_word_embedding_padding)
                print('-'*80)
                print('Created and Initialized fresh model. Size:', _model_size())
                print('Total number of batches:', len(batches))
                print('-'*80)

            for epoch in range(num_epochs):
                shuffle(batches)
                if epoch == 0:
                    batch_start_time = time.time()
                losses = []
                for batch in batches:
                    words, entities = batch
                    if len(words) != self.batch_size:
                        continue
                    feed_dict = {
                        self.__word_input: words,
                        self.__entity_input: entities}
                    loss, step, *_ = session.run([self.__total_loss, self.__global_step, 
                                                  self.__optimizer, self.__clear_word_embedding_padding],
                                                  feed_dict=feed_dict)
                    losses.append(loss)

                    if (step + 1) % print_every == 0:
                        print("step: {}, epoch:{}, time/batch: {:.4}, avg_loss: {:.4}".format(
                            step + 1, epoch+1, (time.time() - batch_start_time)/print_every, np.mean(losses)))
                        batch_start_time = time.time()
                        losses.clear()

                    if should_write_summaries and (step + 1) % summary_batch_interval == 0:
                        summary_str = session.run(self.__summary, feed_dict=feed_dict)
                        summary_writer.add_summary(summary_str, step)
                        print("Saved summaries at step {}".format(step + 1))

                    if should_save_models and (step + 1) % saver_batch_interval == 0:
                        saver.save(session, os.path.join(save_dir, "WC_NER-{}.model".format(step+1)))
                        print("Saved a model at step {}".format(step + 1))

            # last save
            saver.save(session, os.path.join(save_dir, "WC_NER-{}.model".format(step+1)))

            self.__embeddings = self.__word_embeddings.eval()
            if should_write_summaries:
                summary_writer.close()

    def test(self, test_data, load_dir):
        # fit to test data
        self.fit_to_data(test_data, is_training=False)

        batches = self.__prepare_batches()
        config = tf.ConfigProto(allow_soft_placement = True)
        with tf.Session(graph=self.__graph, config=config) as session:
            self.__load(session, load_dir)
            print("-"*80)
            print('Restored model from checkpoint for testing. Size:', _model_size())
            print("-"*80)

            true_entity_list = []
            pred_entity_list = []

            for batch in batches:
                words, entities = batch
                feed_dict = {
                    self.__word_input: words,
                    self.__entity_input: entities}
                pred_entities = session.run(self.__prediction, feed_dict)
            
                true_entity_list.extend(entities)
                pred_entity_list.extend(pred_entities)
            
            target_names = self.__entities[:-1]  # remove "O" from targets.
            labels = list(range(len(target_names)))

            # print results
            print(classification_report(true_entity_list, pred_entity_list, labels, target_names))
            

    
    def embedding_for(self, word_str_or_id):
        if isinstance(word_str_or_id, str):
            return self.embeddings[self.__word_to_id[word_str_or_id]]
        elif isinstance(word_str_or_id, int):
            return self.embeddings[word_str_or_id]

    def __prepare_batches(self):
        if self.__window_list is None:
            raise NotFitToCorpusError(
                "Need to fit model to corpus before preparing training batches.")

        words, entities = zip(*self.__window_list)
        return list(_batchify(self.batch_size, words, entities))

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
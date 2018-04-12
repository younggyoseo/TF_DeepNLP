import tensorflow as tf
import numpy as np
import os
import time
import pickle
from collections import Counter
from random import shuffle

class NotTrainedError(Exception):
    pass

class NotFitToCorpusError(Exception):
    pass

class Doc2VecModel():
    def __init__(self, word_embedding_size, doc_embedding_size, context_size, max_vocab_size=100000,
                 min_occurrences=1, scaling_factor=3/4, batch_size=512, learning_rate=0.05,
                 negative_sample_size=5):

        if isinstance(context_size, tuple):
            self.left_context, self.right_context = context_size
        elif isinstance(context_size, int):
            self.left_context = self.right_context = context_size
        else:
            raise ValueError("`context_size` should be an int or a tuple of two ints")

        self.word_embedding_size = word_embedding_size
        self.doc_embedding_size = doc_embedding_size
        self.max_vocab_size = max_vocab_size
        self.min_occurrences = min_occurrences
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.scaling_factor = scaling_factor
        self.negative_sample_size = negative_sample_size
        self.__corpus_length = None
        self.__words = None
        self.__word_to_id = None
        self.__cooccurrence_matrix = None
        self.__embeddings = None
    
    def fit_to_corpus(self, corpus):
        self.__fit_to_corpus(corpus, self.max_vocab_size, self.min_occurrences,
                             self.left_context, self.right_context)
        self.__build_graph()

    def __fit_to_corpus(self, corpus, vocab_size, min_occurrences, left_size, right_size):
        self.__corpus = corpus  # this is inserted in order to label points in TSNE figure.
        word_counts = Counter()
        window_list = []
        self.__corpus_length = len(corpus)

        for doc_idx, region in enumerate(corpus):
            word_counts.update(region)
            for l_context, word, r_context in _context_windows(region, left_size, right_size):
                l_list = ["<PAD>"] * (self.left_context - len(l_context)) + [word for word in l_context]
                r_list = [word for word in r_context] + ["<PAD>"] * (self.right_context - len(r_context))
                context_list = l_list + r_list
                window_list.append((word, (*context_list, doc_idx)))
        if len(window_list) == 0:
            raise ValueError("No window data in corpus. Did you try to reuse a generator?")

        self.__words, self.__word_counts = list(zip(
            *[(word,count) for idx, (word,count) in enumerate(word_counts.most_common(vocab_size)) 
            if count >= min_occurrences])) #and np.random.random() > subsampling_probs[idx]]))
        # add "<PAD>" to self.__word_to_id
        self.__word_to_id = {"<PAD>" : 0}
        self.__word_to_id.update({word: i for i, word in enumerate(self.__words)})

        # local utility method for membership test of context_list
        def membership_test(seq, set):
            for entry in seq:
                if entry not in seq:
                    return False
            return True

        self.__window_list = [
            (self.__word_to_id[word], tuple(self.__word_to_id[context_word] for context_word in context_list) + (doc_idx,))
            for word, (*context_list, doc_idx) in window_list
            if word in self.__word_to_id and membership_test(context_list, self.__word_to_id)]

    def __build_graph(self):
        self.__graph = tf.Graph()
        with self.__graph.as_default():
            context_length = self.right_context + self.left_context
            concatenated_size = self.word_embedding_size + self.doc_embedding_size

            self.__focal_input = tf.placeholder(tf.int32, shape=[self.batch_size],
                                                name="focal_words")
            self.__context_input = tf.placeholder(tf.int32, shape=[self.batch_size, context_length + 1],  # doc idx
                                                name="context_words")
            
            word_embeddings = tf.get_variable("word_embeddings", [self.vocab_size, self.word_embedding_size],
                                               initializer = tf.random_uniform_initializer(-1.0, 1.0))
            doc_embeddings = tf.get_variable("doc_embeddings", [self.vocab_size, self.doc_embedding_size],
                                               initializer = tf.random_uniform_initializer(-1.0, 1.0))
            nce_weights = tf.get_variable("nce_weights", [self.vocab_size, concatenated_size],
                                               initializer = tf.contrib.layers.xavier_initializer())
            nce_biases = tf.get_variable("nce_biases", [self.vocab_size],
                                               initializer = tf.zeros_initializer())

            # this op clears embedding vector of "<PAD>"
            # 1. after parameter initialization, apply this op to zero out padding embedding vector
            # 2. after each gradient update, apply this op to keep padding at zero
            self.__clear_word_embedding_padding = tf.scatter_update(
                word_embeddings, [0], tf.constant(0.0, shape=[1, self.word_embedding_size]))

            # word vectors
            word_embedding = tf.zeros([self.batch_size, self.word_embedding_size])
            for entry in range(context_length):
                word_embedding += tf.nn.embedding_lookup(word_embeddings, self.__context_input[:, entry])

            # doc vector
            doc_indices = tf.slice(self.__context_input, [0,context_length],[self.batch_size,1])
            doc_embedding = tf.nn.embedding_lookup(doc_embeddings, doc_indices)  #[batch_size, 1, doc_embedding_size]

            # concatenate the sum of word vectors with the document vector, [batch_size, concatenated_size]
            concatenated_embedding = tf.concat([word_embedding, tf.squeeze(doc_embedding)], 1)

            focal_matrix = tf.reshape(tf.cast(self.__focal_input, tf.int64), [self.batch_size, 1])
            sampler = (tf.nn.fixed_unigram_candidate_sampler(
                                    true_classes=focal_matrix,
                                    num_true=1,
                                    num_sampled=self.negative_sample_size,
                                    unique=True,
                                    range_max=self.vocab_size,
                                    distortion=self.scaling_factor,
                                    unigrams=list(self.__word_counts)))

            single_losses = tf.nn.nce_loss(weights=nce_weights,
                                           biases=nce_biases,
                                           labels=focal_matrix,
                                           inputs=concatenated_embedding,
                                           num_sampled=self.negative_sample_size,
                                           num_classes=self.vocab_size,
                                           sampled_values=sampler)

            self.__total_loss = tf.reduce_mean(single_losses)
            tf.summary.scalar("Word2Vec_loss", self.__total_loss)
            self.__global_step = tf.Variable(0, name="global_step", trainable=False)
            learning_rate = tf.train.exponential_decay(self.learning_rate, self.__global_step,
                                                           10000, 0.95, staircase=True)

            self.__optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
                self.__total_loss, global_step=self.__global_step)

            self.__summary = tf.summary.merge_all()
            self.__doc_embeddings = doc_embeddings 
            
    def train(self, num_epochs, log_dir=None, save_dir=None, load_dir=None,
              summary_batch_interval=5000, saver_batch_interval=5000, tsne_epoch_interval=None, 
              print_every=1000):
        should_write_summaries = log_dir is not None and summary_batch_interval
        should_generate_tsne = log_dir is not None and tsne_epoch_interval
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
                ckpt = tf.train.get_checkpoint_state(load_dir)
                assert ckpt, "No checkpoint found"
                assert ckpt.model_checkpoint_path, "No model path found in checkpoint"
                saver.restore(session, ckpt.model_checkpoint_path)
                print('-'*80)
                print('Restored model from checkpoint. Size:', _model_size())
                print('Total number of batches:', len(batches))
                print('-'*80)
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
                    focals, contexts = batch
                    if len(contexts) != self.batch_size:
                        continue
                    feed_dict = {
                        self.__focal_input: focals,
                        self.__context_input: contexts}
                    loss, step, *_ = session.run([self.__total_loss, self.__global_step, 
                                                  self.__optimizer, self.__clear_word_embedding_padding],
                                                  feed_dict=feed_dict)
                    losses.append(loss)

                    if (step + 1) % print_every == 0:
                        print("step: {}, epoch:{}, time/batch: {}, avg_loss: {}".format(
                            step + 1, epoch+1, (time.time() - batch_start_time)/print_every, np.mean(losses)))
                        batch_start_time = time.time()
                        losses.clear()

                    if should_write_summaries and (step + 1) % summary_batch_interval == 0:
                        summary_str = session.run(self.__summary, feed_dict=feed_dict)
                        summary_writer.add_summary(summary_str, step)
                        print("Saved summaries at step {}".format(step + 1))

                    if should_save_models and (step + 1) % saver_batch_interval == 0:
                        saver.save(session, os.path.join(save_dir, "word2vec-{}.model".format(step+1)))
                        print("Saved a model at step {}".format(step + 1))

                if should_generate_tsne and (epoch + 1) % tsne_epoch_interval == 0:
                    current_embeddings = self.__doc_embeddings.eval()
                    output_path = os.path.join(log_dir, "epoch{:03d}.png".format(epoch + 1))
                    self.generate_tsne(output_path, embeddings=current_embeddings)
            self.__embeddings = self.__doc_embeddings.eval()
            
            if should_write_summaries:
                summary_writer.close()

    
    def embedding_for(self, word_str_or_id):
        if isinstance(word_str_or_id, str):
            return self.embeddings[self.__word_to_id[word_str_or_id]]
        elif isinstance(word_str_or_id, int):
            return self.embeddings[word_str_or_id]

    def __prepare_batches(self):
        if self.__window_list is None:
            raise NotFitToCorpusError(
                "Need to fit model to corpus before preparing training batches.")

        focals, contexts = zip(*self.__window_list)
        return list(_batchify(self.batch_size, focals, contexts))

    @property
    def corpus_length(self):
        if self.__corpus_length is None:
            raise NotFitToCorpusError(
                "Need to fit model to corpus before accessing corpus_length.")
        return self.__corpus_length

    @property
    def vocab_size(self):
        return len(self.__words)
    
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
    @property
    def corpus(self):
        if self.__corpus is None:
            raise NotFitToCorpusError("Need to fit model to corpus before accessing corpus.")
        return self.__corpus
    
    def id_for_word(self, word):
        if self.__word_to_id is None:
            raise NotFitToCorpusError("Need to fit model to corpus before looking up word ids")
        return self.__word_to_id[word]

    def generate_tsne(self, path=None, size=(100,100), doc_count=1000, embeddings=None, corpus=None):
        if embeddings is None:
            embeddings = self.embeddings

        from sklearn.manifold import TSNE
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)

        low_dim_embs = tsne.fit_transform(embeddings[:doc_count, :])

        if corpus is None:
            labels = [' '.join(line) for line in self.__corpus[:doc_count]]
        else:
            labels = corpus[:doc_count]
        return _plot_with_labels(low_dim_embs, labels, path, size)



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
                
def _plot_with_labels(low_dim_embs, labels, path, size):
    import matplotlib.pyplot as plt
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    figure = plt.figure(figsize=size)
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5,2), textcoords='offset points', ha='right',
                     va='bottom')

    if path is not None:
        figure.savefig(path)
        plt.close(figure)

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
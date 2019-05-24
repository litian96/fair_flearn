import json
import numpy as np
import tensorflow as tf
from tqdm import trange

from tensorflow.contrib import rnn

from flearn.utils.model_utils import batch_data
from flearn.utils.language_utils import line_to_indices, val_to_vec
from flearn.utils.tf_utils import graph_size, process_grad

with open('flearn/models/sent140/embs.json', 'r') as inf:
    embs = json.load(inf)
id2word = embs['vocab']
word2id = {v: k for k,v in enumerate(id2word)}
word_emb = np.array(embs['emba'])

def process_x(raw_x_batch, max_words=25):
    x_batch = [e[4] for e in raw_x_batch]
    x_batch = [line_to_indices(e, word2id, max_words) for e in x_batch]
    x_batch = np.array(x_batch)
    return x_batch

def process_y(raw_y_batch):
    y_batch = [1 if e=='4' else 0 for e in raw_y_batch]
    y_batch = [val_to_vec(2, e) for e in y_batch]
    y_batch = np.array(y_batch) 
    return y_batch


class Model(object):

    def __init__(self, seq_len, num_classes, n_hidden, q, optimizer, seed):
        #params
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.n_hidden = n_hidden
        self.emb_arr = word_emb

        # create computation graph
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(123+seed)
            self.features, self.labels, self.pred, self.x, self.train_op, self.grads, self.eval_metric_ops, self.loss = self.create_model(q, optimizer)
            self.saver = tf.train.Saver()
        self.sess = tf.Session(graph=self.graph)

        # find memory footprint and compute cost of the model
        self.size = graph_size(self.graph)
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
            metadata = tf.RunMetadata()
            opts = tf.profiler.ProfileOptionBuilder.float_operation()
            self.flops = tf.profiler.profile(self.graph, run_meta=metadata, cmd='scope', options=opts).total_float_ops
    
    def create_model(self, q, optimizer):
        features = tf.placeholder(tf.int32, [None, self.seq_len], name='features')
        labels = tf.placeholder(tf.int32, [None, self.num_classes], name='labels')

        embs = tf.Variable(self.emb_arr, dtype=tf.float32, trainable=False)
        x = tf.nn.embedding_lookup(embs, features)
        
        stacked_lstm = rnn.MultiRNNCell([rnn.BasicLSTMCell(self.n_hidden) for _ in range(2)])
        outputs, _ = tf.nn.dynamic_rnn(stacked_lstm, x, dtype=tf.float32)
        fc1 = tf.layers.dense(inputs=outputs[:,-1,:], units=30)
        pred = tf.layers.dense(inputs=fc1, units=self.num_classes)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=labels))
        grads_and_vars = optimizer.compute_gradients(loss)
        grads, var_list = zip(*grads_and_vars)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())
        
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(labels, 1))
        eval_metric_ops = tf.count_nonzero(correct_pred)
        
        return features, labels, pred, x, train_op, grads, eval_metric_ops, loss

    def set_params(self, model_params=None):
        if model_params is not None:
            with self.graph.as_default():
                all_vars = tf.trainable_variables()
                for variable, value in zip(all_vars, model_params):
                    variable.load(value, self.sess)

    def get_params(self):
        with self.graph.as_default():
            model_params = self.sess.run(tf.trainable_variables())
        return model_params

    def get_loss(self, data):
        x_vecs = process_x(data['x'], self.seq_len)
        labels = process_y(data['y'])
        with self.graph.as_default():
            loss = self.sess.run(self.loss, feed_dict={self.features: x_vecs, self.labels: labels})
        return loss
    
    def solve_inner(self, data, loss, num_epochs=1, batch_size=32):
        '''
        Args:
            data: dict of the form {'x': [list], 'y': [list]}
        Return:
            comp: number of FLOPs computed while training given data
            update: list of np.ndarray weights, with each weight array
        corresponding to a variable in the resulting graph
        '''
        for _ in trange(num_epochs, desc='Epoch: ', leave=False):
            for X, y in batch_data(data, batch_size):
                input_data = process_x(X, self.seq_len)
                target_data = process_y(y)
                with self.graph.as_default():
                    self.sess.run(self.train_op,
                        feed_dict={self.features: input_data, self.labels: target_data})
        soln = self.get_params()
        comp = num_epochs * (len(data['y'])//batch_size) * batch_size * self.flops
        return soln, comp

    def solve_sgd(self, mini_batch_data):
        with self.graph.as_default():
            input_data = process_x(mini_batch_data[0], self.seq_len)
            target_data = process_y(mini_batch_data[1])
            grads, loss, _ = self.sess.run([self.grads, self.loss, self.train_op],
                                    feed_dict={self.features: input_data, self.labels: target_data})
        soln = self.get_params()
        return grads, loss, soln
    
    def test(self, data):
        '''
        Args:
            data: dict of the form {'x': [list], 'y': [list]}
        '''
        x_vecs = process_x(data['x'], self.seq_len)
        labels = process_y(data['y'])
        with self.graph.as_default():
            tot_correct, loss = self.sess.run([self.eval_metric_ops, self.loss],
                feed_dict={self.features: x_vecs, self.labels: labels})
        return tot_correct, loss
    
    def close(self):
        self.sess.close()

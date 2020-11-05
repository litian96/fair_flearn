import numpy as np
from tqdm import trange
import json

import os
import sys
import tensorflow as tf

from tensorflow.contrib import rnn



utils_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
utils_dir = os.path.join(utils_dir, 'utils')
sys.path.append(utils_dir)

from model_utils import batch_data, batch_data2
from language_utils import letter_to_vec, word_to_indices
from tf_utils import graph_size
from tf_utils import process_sparse_grad, process_sparse_grad2

def process_x(raw_x_batch):
    """
    Convert a batch of words into a list.

    Args:
        raw_x_batch: (todo): write your description
    """
    x_batch = [word_to_indices(word) for word in raw_x_batch]
    x_batch = np.array(x_batch)
    return x_batch

def process_y(raw_y_batch):
    """
    Process raw y_y.

    Args:
        raw_y_batch: (todo): write your description
    """
    y_batch = [letter_to_vec(c) for c in raw_y_batch]
    return y_batch

class Model(object):
    def __init__(self, seq_len, num_classes, n_hidden, q, optimizer, seed):
        """
        Initialize the network.

        Args:
            self: (todo): write your description
            seq_len: (int): write your description
            num_classes: (int): write your description
            n_hidden: (int): write your description
            q: (int): write your description
            optimizer: (todo): write your description
            seed: (int): write your description
        """
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.n_hidden = n_hidden

        self.graph = tf.Graph()
        with self.graph.as_default():
            #tf.set_random_seed(123 + seed)
            tf.set_random_seed(456 + seed)
            self.features, self.labels, self.train_op, self.grads, self.eval_metric_ops, self.loss = self.create_model(optimizer)
            self.saver = tf.train.Saver()
        self.sess = tf.Session(graph=self.graph)

        self.size = graph_size(self.graph)

        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
            metadata = tf.RunMetadata()
            opts = tf.profiler.ProfileOptionBuilder.float_operation()
            self.flops = tf.profiler.profile(self.graph, run_meta=metadata, cmd='scope', options=opts).total_float_ops

    def create_model(self, optimizer):
        """
        Create the model.

        Args:
            self: (todo): write your description
            optimizer: (todo): write your description
        """

        features = tf.placeholder(tf.int32, [None, self.seq_len])
        embedding = tf.get_variable("embedding", [self.num_classes, 8])
        x = tf.nn.embedding_lookup(embedding, features)
        labels = tf.placeholder(tf.int32, [None, self.num_classes])
        
        stacked_lstm = rnn.MultiRNNCell([rnn.BasicLSTMCell(self.n_hidden) for _ in range(2)])
        outputs, _ = tf.nn.dynamic_rnn(stacked_lstm, x, dtype=tf.float32)
        pred = tf.layers.dense(inputs=outputs[:,-1,:], units=self.num_classes)
        
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=labels))

        grads_and_vars = optimizer.compute_gradients(loss)
        grads, _ = zip(*grads_and_vars)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())

        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(labels, 1))
        eval_metric_ops = tf.count_nonzero(correct_pred)

        return features, labels, train_op, grads, eval_metric_ops, loss


    def set_params(self, model_params=None):
        """
        Set all variables.

        Args:
            self: (todo): write your description
            model_params: (dict): write your description
        """
        if model_params is not None:
            with self.graph.as_default():
                all_vars = tf.trainable_variables()
                for variable, value in zip(all_vars, model_params):
                    variable.load(value, self.sess)

    def get_params(self):
        """
        Get the model instance.

        Args:
            self: (todo): write your description
        """
        with self.graph.as_default():
            model_params = self.sess.run(tf.trainable_variables())
        return model_params

    def get_loss(self, data):
        """
        Get loss loss : param data : : return :

        Args:
            self: (todo): write your description
            data: (todo): write your description
        """
        with self.graph.as_default():
            loss = self.sess.run(self.loss, feed_dict={self.features: process_x(data['x']), self.labels: process_y(data['y'])})
        return loss

    '''
    def get_gradients(self, data, model_len):

        grads = np.zeros(model_len)
        num_samples = len(data['y'])

        processed_samples = 0

        if num_samples < 50:
            input_data = process_x(data['x'])
            target_data = process_y(data['y'])
            with self.graph.as_default():
                model_grads = self.sess.run(self.grads, 
                    feed_dict={self.features: input_data, self.labels: target_data})
            grads = process_sparse_grad(model_grads)
            processed_samples = num_samples

        else:  # in order to fit into memory, compute gradients in a batch of size 50, and subsample a subset of points to approximate
            for i in range(min(int(num_samples / 50), 4)):
                input_data = process_x(data['x'][50*i:50*(i+1)])
                target_data = process_y(data['y'][50*i:50*(i+1)])

                with self.graph.as_default():
                    model_grads = self.sess.run(self.grads,
                    feed_dict={self.features: input_data, self.labels: target_data})
            
                flat_grad = process_sparse_grad(model_grads)
                grads = np.add(grads, flat_grad)

            grads = grads * 1.0 / min(int(num_samples/50), 4)
            processed_samples = min(int(num_samples / 50), 4) * 50

        return processed_samples, grads
    '''
    
    def solve_inner(self, data, num_epochs=1, batch_size=32):
        '''
        Args:
            data: dict of the form {'x': [list], 'y': [list]}
        Return:
            soln: trainable variables of the lstm model
            comp: number of FLOPs computed while training given data
        '''
        for idx in trange(num_epochs, desc='Epoch: ', leave=False):
            for X,y in batch_data(data, batch_size):
                input_data = process_x(X)
                target_data = process_y(y)
                with self.graph.as_default():
                    self.sess.run(self.train_op,
                        feed_dict={self.features: input_data, self.labels: target_data})
        soln = self.get_params()
        comp = num_epochs * (len(data['y'])//batch_size) * batch_size * self.flops
        return soln, comp
    

    def solve_sgd(self, mini_batch_data):
        '''
        use multiple sgd to simulate gd, in order to by-pass GPU memory limits of running LSTM models
        '''
    
        grads = []   # store the grads of each mini-batch
        final_grads = [] # store the average gradient across all mini-batches (should be the same with GD)
        for X,y in batch_data2(mini_batch_data, batch_size=80):
            input_data = process_x(X)
            target_data = process_y(y)
            with self.graph.as_default():
                grad = self.sess.run(self.grads,
                                    feed_dict={self.features: input_data, self.labels: target_data}) # already applied gradients
                grads.append(process_sparse_grad2(grad))

        for layer in range(len(grads[0])):
            tmp = grads[0][layer]
            for j in range(1, len(grads)):
                tmp += grads[j][layer]
            final_grads.append(tmp * 1.0 / len(grads))
        dummy = 0
        return final_grads, dummy, dummy



    def test(self, data):
        '''
        Args:
            data: dict of the form {'x': [list], 'y': [list]}
        Return:
            tot_correct: total #samples that are predicted correctly
            loss: loss value on `data`
        '''
        x_vecs = process_x(data['x'])
        labels = process_y(data['y'])
        with self.graph.as_default():
            tot_correct, loss = self.sess.run([self.eval_metric_ops, self.loss],
                feed_dict={self.features: x_vecs, self.labels: labels})
        return tot_correct, loss
    
    def close(self):
        """
        Closes the connection.

        Args:
            self: (todo): write your description
        """
        self.sess.close()


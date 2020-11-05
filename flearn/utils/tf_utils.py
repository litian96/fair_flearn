import numpy as np

import tensorflow as tf

def __num_elems(shape):
    '''Returns the number of elements in the given shape

    Args:
        shape: TensorShape
    
    Return:
        tot_elems: int
    '''
    tot_elems = 1
    for s in shape:
        tot_elems *= int(s)
    return tot_elems

def graph_size(graph):
    '''Returns the size of the given graph in bytes

    The size of the graph is calculated by summing up the sizes of each
    trainable variable. The sizes of variables are calculated by multiplying
    the number of bytes in their dtype with their number of elements, captured
    in their shape attribute

    Args:
        graph: TF graph
    Return:
        integer representing size of graph (in bytes)
    '''
    tot_size = 0
    with graph.as_default():
        vs = tf.trainable_variables()
        for v in vs:
            tot_elems = __num_elems(v.shape)
            dtype_size = int(v.dtype.size)
            var_size = tot_elems * dtype_size
            tot_size += var_size
    return tot_size

def process_sparse_grad(grads):
    '''
    Args:
        grads: grad returned by LSTM model (only for the shakespaere dataset)
    Return:
        a flattened grad in numpy (1-D array)
    '''

    indices = grads[0].indices
    values =  grads[0].values
    first_layer_dense = np.zeros((80,8))
    for i in range(indices.shape[0]):
        first_layer_dense[indices[i], :] = values[i, :]

    client_grads = first_layer_dense
    for i in range(1, len(grads)):
        client_grads = np.append(client_grads, grads[i]) # output a flattened array

    return client_grads

def process_sparse_grad2(grads):
    '''
    Args:
        grads: grad returned by LSTM model (only for the shakespaere dataset)
    Return:
        a list of arrays (the same as returned by self.get_params())
    '''
    client_grads = []
    indices = grads[0].indices
    values =  grads[0].values
    first_layer_dense = np.zeros((80,8))
    for i in range(indices.shape[0]):
        first_layer_dense[indices[i], :] = values[i, :]

    client_grads.append(first_layer_dense)
    for i in range(1, len(grads)):
        client_grads.append(grads[i]) 

    return client_grads


def process_grad(grads):
    '''
    Args:
        grads: grad 
    Return:
        a flattened grad in numpy (1-D array)
    '''

    client_grads = grads[0]

    for i in range(1, len(grads)):
        client_grads = np.append(client_grads, grads[i]) # output a flattened array


    return client_grads

def cosine_sim(a, b):
    '''Returns the cosine similarity between two arrays a and b
    '''  
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product * 1.0 / (norm_a * norm_b)  


def softmax(x):
    """
    Return the softmax.

    Args:
        x: (array): write your description
    """
    ex = np.exp(x)
    sum_ex = np.sum( np.exp(x))
    return ex/sum_ex



def norm_grad(grad_list):
    """
    Computes the gradient of gradients.

    Args:
        grad_list: (list): write your description
    """
    # input: nested gradients
    # output: square of the L-2 norm

    client_grads = grad_list[0] # shape now: (784, 26)

    for i in range(1, len(grad_list)):
        client_grads = np.append(client_grads, grad_list[i]) # output a flattened array

    return np.sum(np.square(client_grads))


def norm_grad_sparse(grads):
    """
    Compute the gradients.

    Args:
        grads: (array): write your description
    """
    # input: sparse gradients (usually assicated with trainable embedding variables)
    # output: square of the L-2 norm

    indices = grads[0].indices
    values =  grads[0].values
    first_layer_dense = np.zeros((80,8))
    for i in range(indices.shape[0]):
        first_layer_dense[indices[i], :] = values[i, :]

    client_grads = first_layer_dense

    for i in range(1, len(grads)):
        client_grads = np.append(client_grads, grads[i]) # output a flattened array

    return np.sum(np.square(client_grads))




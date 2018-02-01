from data_preprocessing_Yazabi import *
import tensorflow as tf 


n_nodes_input = len(train_x.columns) # number of input features; 561
n_nodes_hl = 30     # number of units in hidden layer
n_classes = len(np.unique(Y_train_numeric))   # number of activities; 6

    
def neural_network_model(data, dropout_keep_prob):

    # define weights and biases for all each layer
    hidden_layer = {'weights':tf.Variable(tf.truncated_normal([n_nodes_input, n_nodes_hl], stddev=0.3)),
                      'biases':tf.Variable(tf.constant(0.1, shape=[n_nodes_hl]))}
    output_layer = {'weights':tf.Variable(tf.truncated_normal([n_nodes_hl, n_classes], stddev=0.3)),
                    'biases':tf.Variable(tf.constant(0.1, shape=[n_classes]))}
    # feed forward and activations
    l1 = tf.add(tf.matmul(data, hidden_layer['weights']), hidden_layer['biases'])
    l1 = tf.nn.sigmoid(l1)
    l1 = tf.nn.dropout(l1, dropout_keep_prob)
    output = tf.matmul(l1, output_layer['weights']) + output_layer['biases']
    
    return output

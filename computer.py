# Implementation of a feed forward neural network with one hidden layer.
# Requires: numpy, sklearn>=0.18.1, tensorflow>=1.0

# NOTE: In order to make the code simple, we rewrite x * W_1 + b_1 = x' * W_1'
# where x' = [x | 1] and W_1' is the matrix W_1 appended with a new row with elements b_1's.
# Similarly, for h * W_2 + b_2

import csv
import tensorflow as tf
import numpy as np

class Computer:

    def __init__(self):
        #feed forward neural net architecture
        self.num_input_nodes = 4
        self.num_hidden_nodes = 14
        self.num_output_nodes = 5

        #input and output placeholders
        self.x = tf.placeholder(tf.float64, shape = [None, self.num_input_nodes])
        self.y = tf.placeholder(tf.float64, shape = [None, self.num_output_nodes])

        #weights and biases
        self.hidden_weights = tf.Variable(tf.random_uniform([self.num_input_nodes, self.num_hidden_nodes], 0, 1, dtype = tf.float64))
        self.hidden_biases = tf.Variable(tf.random_uniform([self.num_hidden_nodes], 0, 1, dtype = tf.float64))
        self.output_weights = tf.Variable(tf.random_normal([self.num_hidden_nodes, self.num_output_nodes], 0, 1, dtype = tf.float64))
        self.output_biases = tf.Variable(tf.random_normal([self.num_output_nodes], 0, 1, dtype = tf.float64))

        #learning parameters
        self.num_epochs = 30
        #self.batch_size = 5
        self.learning_rate = 0.1

        #record player's moves
        self.player_moves = []

##        self.hidden_weights = tf.Variable(np.full((self.num_input_nodes, self.num_hidden_nodes), 0.5), dtype = tf.float64)
##        self.hidden_biases = tf.Variable(np.full((self.num_hidden_nodes), 1), dtype = tf.float64)
##        self.output_weights = tf.Variable(np.full((self.num_hidden_nodes, self.num_output_nodes), 0.5), dtype = tf.float64)
##        self.output_biases = tf.Variable(np.full((self.num_output_nodes), 1), dtype = tf.float64)

    def dense_to_one_hot(self, labels_dense):
        """
        Convert player moves {0,1,2,3,4} to vectors of the form
        (1,0,0,0,0), (0,1,0,0,0), etc.
        """
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * self.num_output_nodes
        labels_one_hot = np.zeros((num_labels, self.num_output_nodes))
        labels_one_hot.flat[index_offset + labels_dense] = 1
    
        return labels_one_hot

    def forward_prop(self, x):
        hidden = tf.add(tf.matmul(x, self.hidden_weights), self.hidden_biases)
        hidden = tf.nn.relu(hidden)
        output = tf.add(tf.matmul(hidden, self.output_weights), self.output_biases)

        return output

    def predict(self, x):
        """
        Returns the class with the highest probability given an input vector
        """
        
        return tf.argmax(tf.nn.softmax(self.forward_prop(x)), axis = 1)

    def train(self, x, y):
        """
        Desc:
        Determines the parameters of the neural network by minimizing the cross
        entropy of the predicted classes. The cross entropy is the log likelihood
        function of the multinomial distribution. The individual
        probabilities are modeled as p(Y_i = j) = e^(X_i * B_j)/ (\sum_j e^(X_i * B_j))

        Args:
        x is a matrix of the training inputs. The number of rows equals the number of training samples,
        and the number of columns equals the number of input nodes

        y is a row vector of the classes for the training samples. y is converted to one hot vectors
        """

        #convert y to one hot vectors
        y_ = comp.dense_to_one_hot(y)

        #calculate the cost of the inputs and update the parameters
        output = self.forward_prop(self.x) 
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.y, logits = output)) 
        update = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cost)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        for i in range(self.num_epochs):
            sess.run(update, feed_dict={self.x: x, self.y: y_})
            train_accuracy = np.mean(np.argmax(y_, axis = 1) == sess.run(self.predict(x)))
            print("Epoch = %d, train accuracy = %.2f%%" % (i + 1, 100. * train_accuracy))


        print(sess.run(self.predict(np.float64([[0,1,2,3], [1,2,3,4], [2,3,4,0], [3,4,0,1], [4,0,1,2]]))))
        print(self.hidden_weights.eval(sess))

        sess.close()
            
    
##    def test(self):
##        x = tf.placeholder(tf.float64, [None, self.num_input_nodes])
##        init_vars = tf.global_variables_initializer()
##        y = comp.predict(x)
##        y_ = comp.forward_prop(x)
##
##        sess = tf.Session()
##        sess.run(init_vars)
##
##        print(sess.run(self.hidden_weights))
##        print(sess.run(self.hidden_biases))
##        print(sess.run(self.output_weights))
##        print(sess.run(self.output_biases))
##
##        print(sess.run(y_, {x: [[0,1,2,3], [1,2,3,4], [2,3,4,0]]}))
##        print(sess.run(y, {x: [[0, 1, 2, 3], [1,2,3,4], [2,3,4,0]]}))
##        
##        #print(sess.run(self.output_weights))
##        #sess.run(updates, feed_dict={X: train_X[i: i + 1], y: train_y[i: i + 1]})
##        sess.close()
        



##
##
X_ = [] #data
Y_ = [] #labels

with open("/Users/brianlubeck/Documents/DataScience/rpsls/player_data.csv",newline="") as f:
    csv_f = csv.reader(f)

    for row in csv_f:
        X_.append([int(i) for i in row[0:4]])
        Y_.append(int(row[4]))

X = np.float64(X_)
Y = np.int32(Y_)


comp = Computer()
comp.train(X,Y)
#comp.test()

##
##RANDOM_SEED = 42
##tf.set_random_seed(RANDOM_SEED)
##
##def init_weights(shape):
##    """ Weight initialization """
##   weights = tf.random_normal(shape, stddev=0.1)
##    return tf.Variable(weights)
##
##def forwardprop(X, w_1, w_2):
##    """
##    Forward-propagation.
##    IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
##    """
##    h = tf.nn.relu(tf.matmul(X, w_1))
##    yhat = tf.matmul(h, w_2)
##    return yhat
##
##def main():
##    train_X, test_X, train_y, test_y = get_iris_data()
##
##    # Layer's sizes
##    x_size = train_X.shape[1]   # Number of input nodes: 4 features and 1 bias
##    h_size = 256                # Number of hidden nodes
##    y_size = train_y.shape[1]   # Number of outcomes (3 iris flowers)
##
##    # Symbols
##    X = tf.placeholder("float", shape=[None, x_size])
##    y = tf.placeholder("float", shape=[None, y_size])
##
##    # Weight initializations
##    w_1 = init_weights((x_size, h_size))
##    w_2 = init_weights((h_size, y_size))
##
##    # Forward propagation
##    yhat    = forwardprop(X, w_1, w_2)
##    predict = tf.argmax(yhat, axis=1)
##
##    # Backward propagation
##    cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
##    updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
##
##    # Run SGD
##    sess = tf.Session()
##    init = tf.global_variables_initializer()
##    sess.run(init)
##
##    for epoch in range(100):
##        # Train with each example
##        for i in range(len(train_X)):
##            sess.run(updates, feed_dict={X: train_X[i: i + 1], y: train_y[i: i + 1]})
##
##        train_accuracy = np.mean(np.argmax(train_y, axis=1) ==
##                                 sess.run(predict, feed_dict={X: train_X, y: train_y}))
##        test_accuracy  = np.mean(np.argmax(test_y, axis=1) ==
##                                 sess.run(predict, feed_dict={X: test_X, y: test_y}))
##
##        print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%"
##              % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))
##
##    sess.close()
##
##
##from sklearn import datasets
##from sklearn.model_selection import train_test_split
##
##
##
##

##
##def get_iris_data():
##    """ Read the iris data set and split them into training and test sets """
##    iris   = datasets.load_iris()
##    data   = iris["data"]
##    target = iris["target"]
##
##    # Prepend the column of 1s for bias
##    N, M  = data.shape
##    all_X = np.ones((N, M + 1))
##    all_X[:, 1:] = data
##
##    # Convert into one-hot vectors
##    num_labels = len(np.unique(target))
##    all_Y = np.eye(num_labels)[target]  # One liner trick!
##    return train_test_split(all_X, all_Y, test_size=0.33, random_state=RANDOM_SEED)
##

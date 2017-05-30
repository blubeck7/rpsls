import tensorflow as tf
import numpy as np

class MultiClassifier:
    """
    Mulitple Classifier Neural Network. Returns the probability of each class
    for a given input.
    """

    def __init__(self, num_input_nodes, num_hidden_nodes, num_output_nodes):
        #feed forward neural net architecture
        self.num_input_nodes = num_input_nodes
        self.num_hidden_nodes = num_hidden_nodes
        self.num_output_nodes = num_output_nodes

        #input and output placeholders
        self.x = tf.placeholder(tf.float64, shape = [None, self.num_input_nodes])
        self.y = tf.placeholder(tf.float64, shape = [None, self.num_output_nodes])

        #weights and biases
        self.hidden_weights = tf.Variable(tf.random_normal([self.num_input_nodes, self.num_hidden_nodes], 0, 1, dtype = tf.float64))
        self.hidden_biases = tf.Variable(tf.random_normal([self.num_hidden_nodes], 0, 1, dtype = tf.float64))
        self.output_weights = tf.Variable(tf.random_normal([self.num_hidden_nodes, self.num_output_nodes], 0, 1, dtype = tf.float64))
        self.output_biases = tf.Variable(tf.random_normal([self.num_output_nodes], 0, 1, dtype = tf.float64))

        self.cur_hw = np.zeros((self.num_input_nodes, self.num_hidden_nodes), dtype = np.float64)
        self.cur_hb = np.zeros((self.num_hidden_nodes), dtype = np.float64)
        self.cur_ow = np.zeros((self.num_hidden_nodes, self.num_output_nodes), dtype = np.float64)
        self.cur_ob = np.zeros((self.num_output_nodes), dtype = np.float64)

        #learning parameters
        self.num_epochs = 25
        #self.batch_size = 5
        self.learning_rate = 0.1

    def dense_to_one_hot(self, labels_dense):
        """
        Convert vectors in the form {0,1,2,3,4,...} to vectors of the form
        (1,0,0,0,0,...), (0,1,0,0,0,..., etc.
        """
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * self.num_output_nodes
        labels_one_hot = np.zeros((num_labels, self.num_output_nodes))
        labels_one_hot.flat[index_offset + labels_dense] = 1
    
        return labels_one_hot

    def forward_prop(self):
        hidden = tf.add(tf.matmul(self.x, self.hidden_weights), self.hidden_biases)
        hidden = tf.nn.relu(hidden)
        output = tf.add(tf.matmul(hidden, self.output_weights), self.output_biases)

        return output

    def predict(self, x):
        """
        Returns the class with the highest probability given an input vector
        """

        predict = tf.argmax(tf.nn.softmax(self.forward_prop()), axis = 1)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        
        #Load previously trained weights
        sess.run(self.hidden_weights.assign(self.cur_hw))
        sess.run(self.hidden_biases.assign(self.cur_hb))
        sess.run(self.output_weights.assign(self.cur_ow))
        sess.run(self.output_biases.assign(self.cur_ob))

        preds = sess.run(predict, feed_dict = {self.x: x})

        sess.close()

        return preds

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
        y_ = self.dense_to_one_hot(y)

        #calculate the cost of the inputs and update the parameters
        output = self.forward_prop() 
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.y, logits = output)) 
        update = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cost)
        predict = tf.argmax(tf.nn.softmax(self.forward_prop()), axis = 1)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        for i in range(self.num_epochs):
            _, c, p = sess.run([update, cost, predict], feed_dict={self.x: x, self.y: y_})
            #train_accuracy = np.mean(np.argmax(y_, axis = 1) == p)
            
            print("Epoch = %d, cost = %.5f" % (i + 1, c))
            #print("Epoch = %d, train accuracy = %.2f%%" % (i + 1, 100. * train_accuracy))

        #save weights and biases for prediction

        self.cur_hw = sess.run(self.hidden_weights)
        self.cur_hb = sess.run(self.hidden_biases)
        self.cur_ow = sess.run(self.output_weights)
        self.cur_ob = sess.run(self.output_biases)

        #print(self.predict(np.float64([[0,1,2,3], [1,2,3,4], [2,3,4,0], [3,4,0,1], [4,0,1,2]])))
        #print(self.hidden_weights.eval(sess))

        sess.close()

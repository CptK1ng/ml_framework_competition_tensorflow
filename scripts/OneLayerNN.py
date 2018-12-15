import tensorflow as tf
from tensorflow import nn
from scripts.DataLoader import DataLoader
import numpy as np


class OneLayerNeuralNet():

    def __init__(self, path_to_data, hl_size):
        self.data_loader = DataLoader(path_to_data="../data/training.csv")
        self.hl_size=hl_size

    # 96px x 96px = 9216 size for input layer
    x = tf.placeholder(tf.float32, [None, 9216])
    y = tf.placeholder(tf.float32, [30])

    def one_layer_network_model(self):
        fully_connected_layer = {'weights': tf.Variable(tf.random_normal([9216, self.hl_size])),
                                 'biases': tf.Variable(tf.random_normal([self.hl_size]))}

        l1 = tf.add(tf.matmul(self.data, fully_connected_layer['weights']), fully_connected_layer['biases'])
        l1 = tf.nn.relu(l1)

        # Output are 30 Keypount,(15 x and y coordinates for the facial keypoints
        output_layer = {'weights': tf.Variable(tf.random_normal([self.hl_size, 30])),
                        'biases': tf.Variable(tf.random_normal([30])), }

        output = tf.matmul(l1, output_layer['weights']) + output_layer['biases']
        return output

    def train(self,x, epochs):
        prediction = self.one_layer_network_model(x)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
        optimizer = tf.train.AdamOptimizer().minimize(cost)

        hm_epochs = epochs
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())

            for epoch in range(hm_epochs):
                epoch_loss = 0
                x = np.split()
                for _ in range(int(mnist.train.num_examples / batch_size)):
                    epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                    _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                    epoch_loss += c

                print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
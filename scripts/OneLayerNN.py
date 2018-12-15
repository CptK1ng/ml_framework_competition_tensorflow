import tensorflow as tf
from tensorflow import nn
from scripts.DataLoader import DataLoader
import numpy as np


class OneLayerNeuralNet():

    def __init__(self, path_to_data, hl_size):
        self.data_loader = DataLoader(path_to_data=path_to_data)
        self.hl_size=hl_size
        # 96px x 96px = 9216 size for input layer
        self.x = tf.placeholder(tf.float32, [None, 9216])
        self.y = tf.placeholder(tf.float32, [None,30])
        self.train(self.x,50,100)



    def one_layer_network_model(self,data):
        fully_connected_layer = {'weights': tf.Variable(tf.random_normal([9216, self.hl_size])),
                                 'biases': tf.Variable(tf.random_normal([self.hl_size]))}

        l1 = tf.add(tf.matmul(data, fully_connected_layer['weights']), fully_connected_layer['biases'])
        l1 = tf.nn.relu(l1)

        # Output are 30 Keypount,(15 x and y coordinates for the facial keypoints
        output_layer = {'weights': tf.Variable(tf.random_normal([self.hl_size, 30])),
                        'biases': tf.Variable(tf.random_normal([30])), }

        output = tf.matmul(l1, output_layer['weights']) + output_layer['biases']
        return output

    def train(self,x, epochs, batch_size):
        prediction = self.one_layer_network_model(self.x)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=self.y ))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

        hm_epochs = epochs
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())

            x_data = self.data_loader.images
            x_data = [np.ravel(x) for x in x_data]
            y_data = self.data_loader.keypoints
            for epoch in range(hm_epochs):
                epoch_loss = 0
                total_batches = int(len(self.data_loader.images) / batch_size)
                x = np.array_split(x_data, total_batches)
                y = np.array_split(y_data, total_batches)

                for i in range(total_batches):
                    batch_x, batch_y = np.array(x[i]), np.array(y[i])
                    _, c = sess.run([optimizer, cost], feed_dict={self.x: batch_x, self.y: batch_y})
                    epoch_loss += c

                print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)
            #TODO correct muss angepasst werden. Argmax ist glaub ich nicht nötig
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            print('Accuracy:', accuracy.eval({x: self.data_loader.images, y: self.data_loader.keypoints}))



OneLayerNeuralNet(path_to_data="../data/training.csv", hl_size=500)
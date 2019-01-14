import tensorflow as tf
from scripts.DataLoader import DataLoader
import numpy as np
import datetime


class OneLayerNeuralNet():

    def __init__(self, path_to_data, hl_size):
        self.data_loader = DataLoader(path_to_data=path_to_data)
        self.hl_size = hl_size
        # 96px x 96px = 9216 size for input layer
        self.x = tf.placeholder(tf.float32, [None, 9216], name="x")
        self.y = tf.placeholder(tf.float32, [None, 30], name="lables")

    def one_layer_network_model(self, data):
        with tf.name_scope("fcn"):
            fully_connected_layer = {'weights': tf.Variable(tf.random_normal([9216, self.hl_size]), name="W"),
                                     'biases': tf.Variable(tf.random_normal([self.hl_size]), name="B")}
            l1 = tf.add(tf.matmul(data, fully_connected_layer['weights']), fully_connected_layer['biases'])
            l1 = tf.nn.relu(l1)
            tf.summary.histogram("weights", fully_connected_layer['weights'])
            tf.summary.histogram("biases", fully_connected_layer['biases'])
            tf.summary.histogram("activations", l1)

        with tf.name_scope("output"):
            # Output are 30 Keypount,(15 x and y coordinates for the facial keypoints
            output_layer = {'weights': tf.Variable(tf.random_normal([self.hl_size, 30]), name="W"),
                            'biases': tf.Variable(tf.random_normal([30]), name="B")}
            output = tf.matmul(l1, output_layer['weights']) + output_layer['biases']
            tf.summary.histogram("weights", output_layer['weights'])
            tf.summary.histogram("biases", output_layer['biases'])
            tf.summary.histogram("activations", output)

        return output

    def train(self, epochs, batch_size, save_model=False):
        prediction = self.one_layer_network_model(self.x)
        # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=self.y ))
        with tf.name_scope("cost"):
            cost = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.y, predictions=prediction))
            tf.summary.scalar("mse", cost)

        with tf.name_scope("train"):
            # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
            optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

        hm_epochs = epochs

        time = str(datetime.datetime.now().time()).split('.')[0]
        saver = tf.train.Saver()
        best_epoch_loss = 1000
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            merged_summary = tf.summary.merge_all()
            writer = tf.summary.FileWriter(
                '../tmp/facial_keypoint/one_layer/{}hl/{}epochs_{}bs_Adam_lr01'.format(self.hl_size, epochs,
                                                                                       batch_size))
            writer.add_graph(sess.graph)

            x_data = self.data_loader.images
            # Making one dimensional array from 2 dim image
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

                s = sess.run(merged_summary, feed_dict={self.x: batch_x, self.y: batch_y})
                writer.add_summary(s, epoch)

                if epoch_loss < best_epoch_loss and save_model:
                    save_path = saver.save(sess, "../tmp/savepoints/onelayer/{}/model.ckpt".format(time))
                    tf.train.write_graph(sess.graph.as_graph_def(), '..',
                                         'tmp/savepoints/onelayer/{}/one_layer.pbtxt'.format(time),
                                         as_text=True)

                    best_epoch_loss = epoch_loss
                    print("Model saved in path: %s" % save_path)

                print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)


olnn = OneLayerNeuralNet(path_to_data="../data/training.csv", hl_size=500)
olnn.train(500, 32, save_model=True)

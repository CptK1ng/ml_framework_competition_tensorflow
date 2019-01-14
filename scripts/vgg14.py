import tensorflow as tf
from setuptools.command.test import test
from sklearn.model_selection import train_test_split

from tensorflow.python import debug as tf_debug
from scripts.DataLoader import DataLoader
import numpy as np
import datetime


class VGG():

    def __init__(self, path_to_data, ):
        self.data_loader = DataLoader(path_to_data=path_to_data)
        # 96px x 96px = 9216 size for input layer
        self.x = tf.placeholder(tf.float32, [None, 9216], name="x")
        self.y = tf.placeholder(tf.float32, [None, 30], name="labels")
        self.x_image = tf.reshape(self.x, [-1, 96, 96, 1])
        tf.summary.image('input', self.x_image, 3)

    def conv_conv_layer(self, input, size_in, size_out, name="conv"):
        with tf.name_scope(name):
            weights = tf.Variable(tf.truncated_normal([3, 3, size_in, size_out + 2], stddev=0.1), name="weights")
            biases = tf.Variable(tf.constant(0.1, shape=[size_out + 2]), name="biases")
            conv = tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding="VALID")
            act = tf.nn.relu(conv + biases)
            tf.summary.histogram("weights", weights)
            tf.summary.histogram("biases", biases)
            tf.summary.histogram("activations", act)

            weights2 = tf.Variable(tf.truncated_normal([3, 3, size_out + 2, size_out], stddev=0.1), name="weights")
            biases2 = tf.Variable(tf.constant(0.1, shape=[size_out]), name="biases")
            conv2 = tf.nn.conv2d(act, weights2, strides=[1, 1, 1, 1], padding="VALID")
            act2 = tf.nn.relu(conv2 + biases2)
            tf.summary.histogram("weights2", weights2)
            tf.summary.histogram("biases2", biases2)
            tf.summary.histogram("activations2", act2)

            # TODO Angegeben war stride 1, macht aber bei maxpooling mit [2,2] wenig Sinn
            return tf.nn.max_pool(act2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    def fc_layer_w_drpout(self, input, size_in, size_out, dropout_keep = 0.7, name="fc"):
        with tf.name_scope(name):
            weights = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="Weights")
            biases = tf.Variable(tf.constant(0.1, shape=[size_out]), name="Biases")
            act = tf.matmul(input, weights) + biases
            output = tf.nn.dropout(act, keep_prob=dropout_keep)

            tf.summary.histogram("weights", weights)
            tf.summary.histogram("biases", biases)
            tf.summary.histogram("activations", act)
            tf.summary.histogram("dropout", output)

            return output


    def vgg_model(self, x):
        conv1 = self.conv_conv_layer(x, 1, 64, "conv1")
        conv2 = self.conv_conv_layer(conv1, 64, 128, "conv2")

        conv3 = self.conv_conv_layer(conv2, 128, 256, "conv3")
        flattened = tf.layers.Flatten()(conv3)  #
        fcl1 = self.fc_layer_w_drpout(flattened, int(flattened.shape[1]), 512, dropout_keep=0.7, name="fcl1")
        output = self.fc_layer_w_drpout(fcl1, 512, 30, dropout_keep=0.7, name="fcl2")

        return output

    def train(self, learning_rate, epochs, batch_size, optimizer = "adam", save_model=False):
        prediction = self.vgg_model(self.x_image)

        with tf.name_scope("loss"):
            loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.y, predictions=prediction))
            tf.summary.scalar("mse", loss)

        with tf.name_scope("train"):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
            if optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
            elif optimizer =="sgd":
                optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

        summ = tf.summary.merge_all()
        best_epoch_loss = 999999999999
        saver = tf.train.Saver()
        time = str(datetime.datetime.now().time()).split('.')[0]

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter(
                '../tmp/facial_keypoint/vgg/{}epochs_{}bs_Adam_lr{}_{}'.format(epochs, batch_size, learning_rate,
                                                                                  time))
            writer.add_graph(sess.graph)

            # Training procedure
            # Making one dimensional array from 2 dim image
            x_data = np.array([np.ravel(x) for x in self.data_loader.images])
            y_data = self.data_loader.keypoints
            x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)
            for epoch in range(epochs):
                epoch_loss = 0

                total_batches = int(len(x_train) / batch_size)
                x = np.array_split(x_train, total_batches)
                y = np.array_split(y_train, total_batches)

                for i in range(total_batches):
                    batch_x, batch_y = np.array(x[i]), np.array(y[i])
                    _, c = sess.run([optimizer, loss], feed_dict={self.x: batch_x, self.y: batch_y})
                    epoch_loss += c

                if epoch % 5 == 0:
                    batch_x, batch_y = np.array(x[i]), np.array(y[i])
                    s = sess.run(summ, feed_dict={self.x: batch_x, self.y: batch_y})
                    writer.add_summary(s, epoch)

                if epoch_loss < best_epoch_loss and save_model:
                    save_path = saver.save(sess,
                                           "../tmp/savepoints/vgg/{}/model.ckpt".format(time))
                    tf.train.write_graph(sess.graph.as_graph_def(), '..', 'tmp/savepoints/vgg/{}/vgg.pbtxt'.format(time), as_text=True)

                    best_epoch_loss = epoch_loss
                    print("Model saved in path: %s" % save_path)

                print('Epoch', epoch, 'completed out of', epochs, 'loss:', epoch_loss)


ml_network = VGG(path_to_data="../data/training.csv")
ml_network.train(1e-2, 100, 50, optimizer='adam', save_model=True)

import tensorflow as tf
from sklearn.model_selection import train_test_split

from scripts.DataLoader import DataLoader
import numpy as np
import datetime
import time
from PIL import Image
import matplotlib.pyplot as plt


class LeNet():

    def __init__(self, path_to_data=None):
        self.data_loader = DataLoader(path_to_data=path_to_data)
        # 96px x 96px = 9216 size for input layer
        self.x = tf.placeholder(tf.float32, [None, 9216], name="x")
        self.y = tf.placeholder(tf.float32, [None, 30], name="labels")
        self.x_image = tf.reshape(self.x, [-1, 96, 96, 1])
        self.mean_loss = 10000

        tf.summary.image('input', self.x_image, 3)

    def getimage(self, array):
        img = Image.new('RGB', (96, 96), "black")
        pixels = img.load()  # create the pixel map

        cot = array
        for i in range(img.size[0]):  # for every pixel:
            for j in range(img.size[1]):
                pixels[i, j] = (int(cot[i][j]), int(cot[i][j]), int(cot[i][j]))  # set the colour accordingly
        return img

    def conv_layer(self, input, size_in, size_out, name="conv"):
        with tf.name_scope(name):
            weights = tf.Variable(tf.truncated_normal([3, 3, size_in, size_out], stddev=0.1), name="weights")
            biases = tf.Variable(tf.constant(0.1, shape=[size_out]), name="biases")
            conv = tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding="VALID")
            act = tf.nn.relu(conv + biases)
            tf.summary.histogram("weights", weights)
            tf.summary.histogram("biases", biases)
            tf.summary.histogram("activations", act)
            return tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    def fc_layer(self, input, size_in, size_out, name="fc"):
        with tf.name_scope(name):
            weights = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="Weights")
            biases = tf.Variable(tf.constant(0.1, shape=[size_out]), name="Biases")
            act = tf.matmul(input, weights) + biases
            tf.summary.histogram("weights", weights)
            tf.summary.histogram("biases", biases)
            tf.summary.histogram("activations", act)
            return act

    def le_net_model(self, x):
        conv1 = self.conv_layer(x, 1, 16, "conv1")
        conv2 = self.conv_layer(conv1, 16, 32, "conv2")
        conv3 = self.conv_layer(conv2, 32, 64, "conv3")
        conv4 = self.conv_layer(conv3, 64, 128, "conv4")
        conv5 = self.conv_layer(conv4, 128, 256, "conv5")

        # Flatten the array to make it processable for fc layers
        flattened = tf.layers.Flatten()(conv5)  #
        fcl1 = self.fc_layer(flattened, int(flattened.shape[1]), 500, "fcl1")
        fcl2 = self.fc_layer(fcl1, 500, 500, "fcl1")
        fcl3 = self.fc_layer(fcl2, 500, 500, "fcl2")
        fcl4 = self.fc_layer(fcl3, 500, 500, "fcl3")
        output = self.fc_layer(fcl4, 500, 30, "output")

        return output

    def train(self, learning_rate, epochs, batch_size, save_model=False, repeat_training_n_times=1):
        prediction = self.le_net_model(self.x_image)

        with tf.name_scope("loss"):
            # == loss = tf.reduce_sum(tf.pow(prediction - Y,2))/(n_instances)
            loss = tf.reduce_sum(tf.losses.mean_squared_error(labels=self.y, predictions=prediction))
            tf.summary.scalar("sse", loss)

        with tf.name_scope("train"):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        day_time = str(datetime.datetime.now().time()).split('.')[0]

        with tf.Session() as sess:
            # Training procedure
            # Making one dimensional array from 2 dim image

            all_train_losses, all_test_losses = [], []
            for run in range(repeat_training_n_times):
                x_data = np.array([np.ravel(x) for x in self.data_loader.images])
                y_data = self.data_loader.keypoints
                x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)

                train_writer = tf.summary.FileWriter(
                    '../tmp/facial_keypoint/le_net/{}epochs_{}bs_Adam_lr{}_{}/{}/train'.format(epochs, batch_size,
                                                                                               learning_rate,
                                                                                               day_time, run))
                test_writer = tf.summary.FileWriter(
                    '../tmp/facial_keypoint/le_net/{}epochs_{}bs_Adam_lr{}_{}/{}/test'.format(epochs, batch_size,
                                                                                              learning_rate,
                                                                                              day_time, run))
                print("Start Training No. ", run)
                sess.run(tf.global_variables_initializer())
                train_writer.add_graph(sess.graph)

                merged_summary = tf.summary.merge_all()
                best_epoch_loss = 1000
                saver = tf.train.Saver()

                training_losses, test_losses = [], []
                start_time = time.perf_counter()
                for epoch in range(epochs):
                    if epoch % 20 == 0:
                        print("Training epoch ", epoch)

                    losses = []
                    epoch_loss = 0

                    total_batches = int(len(x_train) / batch_size)
                    x = np.array_split(x_train, total_batches)
                    y = np.array_split(y_train, total_batches)

                    for i in range(total_batches):
                        batch_x, batch_y = np.array(x[i]), np.array(y[i])
                        _, c = sess.run([optimizer, loss], feed_dict={self.x: batch_x, self.y: batch_y})
                        epoch_loss += c

                    epoch_loss /= total_batches

                    # Test current weights on test data every 5 epochs
                    if epoch % 1 == 0:
                        l, s = sess.run([loss, merged_summary], feed_dict={self.x: x_test, self.y: y_test})
                        test_writer.add_summary(s, epoch)
                        test_losses.append(l)
                    # Writing all information to SummaryWriter
                    if epoch % 1 == 0:
                        batch_x, batch_y = np.array(x[i]), np.array(y[i])
                        s = sess.run(merged_summary, feed_dict={self.x: batch_x, self.y: batch_y})
                        train_writer.add_summary(s, epoch)

                    if epoch_loss < best_epoch_loss and save_model:
                        saver.save(sess, "../tmp/savepoints/lenet/{}/model.ckpt".format(day_time))
                        tf.train.write_graph(sess.graph.as_graph_def(), '..',
                                             'tmp/savepoints/lenet/{}/lenet.pbtxt'.format(day_time), as_text=True)

                        best_epoch_loss = epoch_loss

                    training_losses.append(epoch_loss)
                end_time = datetime.timedelta(seconds=time.perf_counter() - start_time)
                print(
                    "Finished run No. {}, \n\t train loss at epoch {}: {} | test loss last epoch: {}, \n\t Best loss {}. Training takes: {}".format(
                        run, epochs, training_losses[-1], test_losses[-1], best_epoch_loss, end_time))
                # sess.close()

                all_train_losses.append(training_losses)
                all_test_losses.append(test_losses)
                """
                plt.plot(np.arange(len(training_losses)), training_losses)
                plt.plot(np.arange(len(training_losses)), test_losses)
                plt.legend(("Train", "test"))
                plt.yscale('log')
                plt.show()
                """

            runs = []
            for i in range(len(all_test_losses)):
                plt.plot(np.arange(len(all_train_losses[i])), all_train_losses[i], '-')
                plt.plot(np.arange(len(all_test_losses[i])), all_test_losses[i], '--')
                runs.append("Taining {}".format(i))
                runs.append("Test {}".format(i))

            plt.yscale('log')
            plt.legend(tuple(runs))
            savepath = "../tmp/facial_keypoint/le_net/{}epochs_{}bs_Adam_lr{}_{}/{}/one_layer.png".format(
                self.hl_size,
                epochs,
                batch_size,
                learning_rate,
                day_time)
            plt.savefig(savepath, dpi=150)


ml_network = LeNet(path_to_data="../data/training.csv")
ml_network.train(1e-3, 125, 32, save_model=True, repeat_training_n_times=1)

import tensorflow as tf
from scripts.DataLoader import DataLoader
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt


class OneLayerNeuralNet():

    def __init__(self, path_to_data, hl_size):
        self.data_loader = DataLoader(path_to_data=path_to_data)
        self.hl_size = hl_size
        # 96px x 96px = 9216 size for input layer
        self.x = tf.placeholder(tf.float32, [None, 9216], name="x")
        self.y = tf.placeholder(tf.float32, [None, 30], name="labels")

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

    def train(self, learning_rate, epochs, batch_size, save_model=False, repeat_training_n_times=1):
        prediction = self.one_layer_network_model(self.x)
        with tf.name_scope("loss"):
            loss = tf.reduce_sum(tf.losses.mean_squared_error(labels=self.y, predictions=prediction))
            tf.summary.scalar("sse", loss)

        with tf.name_scope("train"):
            # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        hm_epochs = epochs
        day_time = str(datetime.datetime.now().time()).split('.')[0]
        with tf.Session() as sess:

            all_train_losses, all_test_losses = [], []
            for run in range(repeat_training_n_times):
                x_data = self.data_loader.images
                # Making one dimensional array from 2 dim image
                x_data = [np.ravel(x) for x in x_data]
                y_data = self.data_loader.keypoints
                x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)

                train_writer = tf.summary.FileWriter(
                    '../tmp/facial_keypoint/one_layer/{}hl/{}epochs_{}bs_Adam_lr{}/{}/{}/train'.format(self.hl_size,
                                                                                                       epochs,
                                                                                                       batch_size,
                                                                                                       learning_rate,
                                                                                                       day_time, run))
                test_writer = tf.summary.FileWriter(
                    '../tmp/facial_keypoint/one_layer/{}hl/{}epochs_{}bs_Adam_lr{}/{}/{}/test'.format(self.hl_size,
                                                                                                      epochs,
                                                                                                      batch_size,
                                                                                                      learning_rate,
                                                                                                      day_time, run))
                print("Started Training No. ", run)
                sess.run(tf.global_variables_initializer())
                train_writer.add_graph(sess.graph)

                merged_summary = tf.summary.merge_all()
                best_epoch_loss = 1000
                saver = tf.train.Saver()

                training_losses, test_losses = [], []
                start_time = time.perf_counter()
                for epoch in range(hm_epochs):
                    if epoch % 20 == 0:
                        print("Training epoch ", epoch)

                    epoch_loss = 0

                    total_batches = int(len(x_train) / batch_size)
                    x = np.array_split(x_train, total_batches)
                    y = np.array_split(y_train, total_batches)

                    for i in range(total_batches):
                        batch_x, batch_y = np.array(x[i]), np.array(y[i])
                        _, c = sess.run([optimizer, loss], feed_dict={self.x: batch_x, self.y: batch_y})
                        epoch_loss += c

                    epoch_loss /= total_batches

                    if epoch % 1 == 0:
                        l, s = sess.run([loss, merged_summary], feed_dict={self.x: x_test, self.y: y_test})
                        test_writer.add_summary(s, epoch)
                        test_losses.append(l)

                    if epoch % 1 == 0:
                        batch_x, batch_y = np.array(x[i]), np.array(y[i])
                        s = sess.run(merged_summary, feed_dict={self.x: batch_x, self.y: batch_y})
                        train_writer.add_summary(s, epoch)

                    if epoch_loss < best_epoch_loss and save_model:
                        saver.save(sess, "../tmp/savepoints/onelayer/{}/model.ckpt".format(day_time))
                        tf.train.write_graph(sess.graph.as_graph_def(), '..',
                                             'tmp/savepoints/onelayer/{}/one_layer.pbtxt'.format(day_time),
                                             as_text=True)

                        best_epoch_loss = epoch_loss

                    training_losses.append(epoch_loss)
                end_time = datetime.timedelta(seconds=time.perf_counter() - start_time)
                print(
                    "Finished run No. {}, \n\t train loss at epoch {}: {} | test loss last epoch: {}, \n\t Best loss {}. Training takes: {}".format(
                        run, epochs, training_losses[-1], test_losses[-1], best_epoch_loss, end_time))
                all_train_losses.append(training_losses)
                all_test_losses.append(test_losses)

            runs = []
            for i in range(len(all_test_losses)):
                plt.plot(np.arange(len(all_train_losses[i])), all_train_losses[i], '-')
                plt.plot(np.arange(len(all_test_losses[i])), all_test_losses[i], '--')
                runs.append("Taining {}".format(i))
                runs.append("Test {}".format(i))

            plt.yscale('log')
            plt.legend(tuple(runs))
            savepath = "../tmp/facial_keypoint/one_layer/{}hl/{}epochs_{}bs_Adam_lr{}/{}/one_layer.png".format(
                self.hl_size,
                epochs,
                batch_size,
                learning_rate,
                day_time)
            plt.savefig(savepath,dpi=150 )


olnn = OneLayerNeuralNet(path_to_data="../data/training.csv", hl_size=500)
olnn.train(1e-3, 20, 32, save_model=True, repeat_training_n_times=2)

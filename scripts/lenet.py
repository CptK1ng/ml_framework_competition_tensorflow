import tensorflow as tf
from sklearn.model_selection import train_test_split

from scripts.DataLoader import DataLoader
import numpy as np
import datetime
import cv2 as cv
from PIL import Image


class LeNet():

    def __init__(self, path_to_data, ):
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
            # TODO Angegeben war stride 1, macht aber bei maxpooling mit [2,2] wenig Sinn
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

    def train(self, learning_rate, epochs, batch_size, save_model=False, modus='inference'):
        prediction = self.le_net_model(self.x_image)

        with tf.name_scope("loss"):
            loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.y, predictions=prediction))
            tf.summary.scalar("mse", loss)

        with tf.name_scope("train"):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        summ = tf.summary.merge_all()
        best_epoch_loss = 1000
        saver = tf.train.Saver()
        time = str(datetime.datetime.now().time()).split('.')[0]

        with tf.Session() as sess:
            # Training procedure
            # Making one dimensional array from 2 dim image
            x_data = np.array([np.ravel(x) for x in self.data_loader.images])
            y_data = self.data_loader.keypoints
            x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)

            imgs_raw = self.data_loader.images
            dl1 = np.array([np.ravel(x) for x in self.data_loader.images])

            if modus is 'inference':
                saver.restore(sess, "../tmp/savepoints/lenet/17:05:10/model.ckpt")
                sess.run(tf.global_variables_initializer())

                for i in range(len(imgs_raw)):
                    batch_x = np.array(dl1[i])
                    img_e = np.expand_dims(batch_x, axis=0)

                    pred = sess.run(prediction, feed_dict={self.x: img_e})

                    new_img = np.array(self.getimage(imgs_raw[i]))
                    open_cv_image = new_img[:, :, ::-1].copy()

                    for pts in range(0, pred.shape[1], 2):
                        open_cv_image = cv.circle(open_cv_image, (int(pred[0][pts]), int(pred[0][pts + 1])), 1,
                                                  (255, 0, 0), -1)

                    open_cv_image = cv.resize(open_cv_image, (320, 320), interpolation=cv.INTER_CUBIC)
                    cv.imwrite("../data/output/{}.png".format(str(i)), open_cv_image)
                    cv.waitKey(1)

            elif modus is 'training':
                sess.run(tf.global_variables_initializer())
                writer = tf.summary.FileWriter(
                    '../tmp/facial_keypoint/le_net/{}epochs_{}bs_Adam_lr{}_{}'.format(epochs, batch_size, learning_rate,
                                                                                      time))
                writer.add_graph(sess.graph)

                for epoch in range(epochs):
                    losses = []
                    epoch_loss = 0

                    total_batches = int(len(x_train) / batch_size)
                    x = np.array_split(x_train, total_batches)
                    y = np.array_split(y_train, total_batches)

                    for i in range(total_batches):
                        batch_x, batch_y = np.array(x[i]), np.array(y[i])
                        _, c = sess.run([optimizer, loss], feed_dict={self.x: batch_x, self.y: batch_y})
                        epoch_loss += c
                        losses.append(c)

                    self.mean_loss = np.mean(np.array(losses)) / batch_size
                    tf.summary.scalar("per_image_loss", self.mean_loss)

                    if epoch % 5 == 0:
                        batch_x, batch_y = np.array(x[i]), np.array(y[i])
                        s = sess.run(summ, feed_dict={self.x: batch_x, self.y: batch_y})
                        writer.add_summary(s, epoch)

                    if epoch_loss < best_epoch_loss and save_model:
                        save_path = saver.save(sess, "../tmp/savepoints/lenet/{}/model.ckpt".format(time))
                        tf.train.write_graph(sess.graph.as_graph_def(), '..',
                                             'tmp/savepoints/lenet/{}/lenet.pbtxt'.format(time), as_text=True)

                        best_epoch_loss = epoch_loss
                        print("Model saved in path: %s" % save_path)

                    print('Epoch', epoch, 'completed out of', epochs, 'loss:', epoch_loss, 'per img loss: ',
                          self.mean_loss)


ml_network = LeNet(path_to_data="../data/training.csv")
ml_network.train(5e-3, 250, 32, save_model=True, modus="inference")

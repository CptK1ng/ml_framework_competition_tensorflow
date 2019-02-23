import tensorflow as tf
from scripts.DataLoader import DataLoader
import numpy as np
import cv2 as cv
from PIL import Image


def getimage(array):
    img = Image.new('RGB', (96, 96), "black")
    pixels = img.load()  # create the pixel map

    cot = array
    for i in range(img.size[0]):  # for every pixel:
        for j in range(img.size[1]):
            pixels[i, j] = (int(cot[i][j]), int(cot[i][j]), int(cot[i][j]))  # set the colour accordingly
    return img


def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph


if __name__ == '__main__':
    # We use our "load_graph" function
    graph = load_graph("/home/lukas/Projects/Projektarbeit_WS18/tmp/savepoints/lenet/16:01:05/frozen_model.pb")

    # access the input and output nodes
    x = graph.get_tensor_by_name('prefix/x:0')
    y = graph.get_tensor_by_name('prefix/output/add:0')

    # We launch a Session
    dl = DataLoader(path_to_data="../data/training.csv")
    dl1 = np.array([np.ravel(x) for x in dl.images])
    with tf.Session(graph=graph) as sess:

        imgs = dl1[:200]
        imgs_raw = dl.images[:200]
        counter = 0
        for img, img_raw in zip(imgs, imgs_raw):
            img_e = np.expand_dims(img, axis=0)
            y_out = sess.run(y, feed_dict={
                x: img_e  # < 45
            })

            new_img = np.array(getimage(img_raw))
            open_cv_image = new_img[:, :, ::-1].copy()

            for pts in range(0, y_out.shape[1], 2):
                open_cv_image = cv.circle(open_cv_image, (int(y_out[0][pts]), int(y_out[0][pts + 1])), 1, (255, 0, 0),
                                          -1)

            open_cv_image = cv.resize(open_cv_image, (320, 320), interpolation=cv.INTER_CUBIC)
            cv.imwrite("../data/output/{}.png".format(str(counter)), open_cv_image)
            cv.waitKey(1)
            counter +=1

def image_helper(np_array):
    mat = np.zeros((np_array.shape[0], np_array.shape[1], 3))

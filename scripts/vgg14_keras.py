from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from scripts.data_loader import DataLoader
import keras
from sklearn.model_selection import train_test_split


class TestCallback(keras.callbacks.Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))


class VGGModel():

    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.model = self.create_model()
        self.train()

    def create_model(self):
        model = Sequential()
        model.add(Conv2D(self.batch_size, (3,3), activation='relu', input_shape=(96,96,1)))
        model.add(Conv2D(self.batch_size, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(self.batch_size, (3, 3), activation='relu'))
        model.add(Conv2D(self.batch_size, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(self.batch_size, (3, 3), activation='relu'))
        model.add(Conv2D(self.batch_size, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(500, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(500, activation='relu'))
        model.add(Dropout(0.1))

        model.add(Dense(30))
        model.summary()
        return model

    def train(self):
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        dl = DataLoader(path_to_data="../data/training.csv")
        #x_data = np.array([np.ravel(x) for x in dl.images])
        y_data = dl.keypoints
        x_train, x_test, y_train, y_test = train_test_split(dl.images.reshape(-1, dl.images.shape[1],dl.images.shape[2],1), y_data, test_size=0.2)
        tb_callback = keras.callbacks.TensorBoard(log_dir='../logs', histogram_freq=0, batch_size=32, write_graph=True,
                                                  write_grads=False, write_images=True, embeddings_freq=0,
                                                  embeddings_layer_names=None, embeddings_metadata=None,
                                                  embeddings_data=None,
                                                  update_freq='epoch')

        self.model.fit(x=x_train, y=y_train, epochs=250, batch_size=32,
                  callbacks=[tb_callback, TestCallback((x_test, y_test))])

        score = self.model.evaluate(x=x_test, y=y_test, batch_size=32)
        print(score)
        test = 0

vgg = VGGModel(16)

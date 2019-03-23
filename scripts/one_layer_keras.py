from keras.models import Sequential
from keras.layers import Dense, Activation
from scripts.data_loader import DataLoader
import numpy as np
import keras
from sklearn.model_selection import train_test_split

class TestCallback(keras.callbacks.Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))

model = Sequential([
    Dense(500, input_dim=9216),
    Activation('relu'),
    Dense(30)
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

model.summary()
dl = DataLoader(path_to_data="../data/training.csv")
x_data = np.array([np.ravel(x) for x in dl.images])
y_data = dl.keypoints
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)
tb_callback = keras.callbacks.TensorBoard(log_dir='../logs', histogram_freq=0, batch_size=32, write_graph=True,
                                          write_grads=False, write_images=True, embeddings_freq=0,
                                          embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None,
                                          update_freq='epoch')

model.fit(x=x_train, y=y_train, epochs=250, batch_size=32, callbacks=[tb_callback,TestCallback((x_test, y_test))])

score = model.evaluate(x=x_test, y=y_test, batch_size=32)
print(score)
test = 0

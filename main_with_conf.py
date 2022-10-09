"""
execute code
"""
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam


def main(conf):

    # create fake data
    x_train = np.random.randn(conf["n_rows_train"], conf["n_columns"])
    y_train = np.random.choice([0, 1], size=(conf["n_rows_train"],), p=[1./2, 1./2])
    x_test = np.random.randn(conf["n_rows_train"], conf["n_columns"])
    y_test = np.random.choice([0, 1], size=(conf["n_rows_test"]), p=[1./2, 1./2])

    # define the model
    model = Sequential()
    model.add((Dense(conf["n_neurons_1"], input_shape=(conf["n_columns"],), activation=conf["activation_1"], name="layer_1")))
    model.add((Dense(conf["n_neurons_2"], activation=conf["activation_2"], name="layer_2")))
    model.add((Dense(1, activation="sigmoid", name="layer_output")))
    optimizer = Adam(learning_rate=conf["learning_rate"])
    model.compile(loss='binary_crossentropy', optimizer=optimizer)

    # train the model and make prediction
    model.fit(x_train, y_train)
    pred_test = model.predict(x_test)
    print(pred_test, y_test)


if __name__ == "__main__":

    # test 2 : configuration file
    from src.config.conf import conf
    main(conf)







"""
execute code
"""
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam


def main():

    # create fake data
    n_rows_train = 1000
    n_rows_test = 200
    n_columns = 5

    x_train = np.random.randn(n_rows_train, n_columns)
    y_train = np.random.choice([0, 1], size=(n_rows_train,), p=[1./2, 1./2])
    x_test = np.random.randn(n_rows_train, n_columns)
    y_test = np.random.choice([0, 1], size=(n_rows_test,), p=[1./2, 1./2])

    # define the model
    n_neurons_1 = 10
    activation_1 = "relu"
    n_neurons_2 = 5
    activation_2 = "tanh"
    learning_rate = 0.05
    model = Sequential()
    model.add((Dense(n_neurons_1, input_shape=(n_columns,), activation=activation_1, name="layer_1")))
    model.add((Dense(n_neurons_2, activation=activation_2, name="layer_2")))
    model.add((Dense(1, activation="sigmoid", name="layer_output")))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)

    # train the model and make prediction
    model.fit(x_train, y_train)
    pred_test = model.predict(x_test)
    print(pred_test, y_test)


if __name__ == "__main__":

    # test 1 : argparse
    main()







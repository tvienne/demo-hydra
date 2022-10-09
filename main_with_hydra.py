"""
execute code
"""
import hydra
from hydra.core.config_store import ConfigStore
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam


#cs = ConfigStore.instance()
#cs.store(name="test_config", node=)

@hydra.main(config_path="config", config_name="conf")
def main(conf):

    # create fake data
    print(conf)
    x_train = np.random.randn(conf.data.n_rows_train, conf.data.n_columns)
    y_train = np.random.choice([0, 1], size=(conf.data.n_rows_train,), p=[1./2, 1./2])
    x_test = np.random.randn(conf.data.n_rows_train, conf.data.n_columns)
    y_test = np.random.choice([0, 1], size=(conf.data.n_rows_test,), p=[1./2, 1./2])

    # define the model
    model = Sequential()
    model.add((Dense(conf.model.n_neurons_1, input_shape=(conf.data.n_columns,), activation=conf.model.activation_1, name="layer_1")))
    model.add((Dense(conf.model.n_neurons_2, activation=conf.model.activation_2, name="layer_2")))
    model.add((Dense(1, activation="sigmoid", name="layer_output")))
    optimizer = Adam(learning_rate=conf.model.learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)

    # train the model and make prediction
    model.fit(x_train, y_train)
    pred_test = model.predict(x_test)
    print(pred_test, y_test)


if __name__ == "__main__":

    # test 4 : hydra
    main()







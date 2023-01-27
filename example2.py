from keras.datasets import imdb
import numpy as np
from keras import layers, models
import matplotlib.pyplot as plt

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequences in enumerate(sequences):
        results[i, sequences] = 1.0
    return results


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype("float32")
y_test = np.asarray(test_labels).astype("float32")

model = models.Sequential()
model.add(layers.Dense(16, activation="relu", input_shape=(10000,)))
model.add(layers.Dense(16, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))

model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])

x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))

history_dict = history.history
loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]

epochs = range(1, len(history_dict["acc"]) + 1)

plt.plot(epochs, loss_values, "bo", label="Training loss")
plt.plot(epochs, val_loss_values, "b", label="Validation loss")
plt.title("Traning and validation loss")
plt.xlabel("Epoches")
plt.ylabel("Loss")
plt.legend()

plt.show()

plt.clf()

acc_values = history_dict["acc"]
val_acc_values = history_dict["val_acc"]

plt.plot(epochs, acc_values, "bo", label="Training acc")
plt.plot(epochs, val_acc_values, "b", label="Validation acc")
plt.title("Traning and validation accuracy")
plt.xlabel("Epoches")
plt.ylabel("Accuracy")
plt.legend()

plt.show()

new_model = models.Sequential()
new_model.add(layers.Dense(16, activation="relu", input_shape=(10000,)))
new_model.add(layers.Dense(16, activation="relu"))
new_model.add(layers.Dense(1, activation="sigmoid"))

new_model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])

new_model.fit(x_train, y_train, epochs=4, batch_size=512)
results = new_model.evaluate(x_test, y_test)

print(results)

print(new_model.predict(x_test))

from keras.datasets import reuters
from keras.utils.np_utils import to_categorical
from keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequences in enumerate(sequences):
        results[i, sequences] = 1.0
    return results


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)

model = models.Sequential()
model.add(layers.Dense(64, activation="relu", input_shape=(10000,)))
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(46, activation="softmax"))

model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["acc"])

x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))

loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs = range(1, len(history.history["acc"]) + 1)

plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Traning and validation loss")
plt.xlabel("Epoches")
plt.ylabel("Loss")
plt.legend()

plt.show()

plt.clf()

acc = history.history["acc"]
val_acc = history.history["val_acc"]

plt.plot(epochs, acc, "bo", label="Training acc")
plt.plot(epochs, val_acc, "b", label="Validation acc")
plt.title("Traning and validation accuracy")
plt.xlabel("Epoches")
plt.ylabel("Accuracy")
plt.legend()

plt.show()

new_model = models.Sequential()
new_model.add(layers.Dense(64, activation="relu", input_shape=(10000,)))
new_model.add(layers.Dense(64, activation="relu"))
new_model.add(layers.Dense(46, activation="softmax"))

new_model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["acc"])

new_model.fit(x_train, one_hot_train_labels, epochs=9, batch_size=512, validation_data=(x_val, y_val))
results = new_model.evaluate(x_test, one_hot_test_labels)

print(results)

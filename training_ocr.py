import tensorflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# from tensorflow.keras.datasets import mnist
from keras.datasets import mnist


(train_data, train_labels), (test_data, test_labels) = mnist.load_data()

digits_data = np.vstack([train_data, test_data])
digits_labels = np.hstack([train_labels, test_labels])


index = np.random.randint(0, digits_data.shape[0])
plt.imshow(digits_data[index], cmap="gray")
plt.title("Class: " + str(digits_labels[index]))


plt.show()

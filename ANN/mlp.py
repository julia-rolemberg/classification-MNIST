import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
import tensorflow as tf
from tensorflow import keras
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "ann"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


iris = load_iris()
X = iris.data[:,(2,3)] #petal length, petal width
y = (iris.target == 0).astype(np.int) # iris setosa?
print('\n--------------------Perceptron----------------------')
per_clf = Perceptron()
per_clf.fit(X, y)

y_pred = per_clf.predict([[2,0.5]])
print(f'y_pred of (2,0.5)--> {y_pred}')



print('\n--------------------MultiLayer Perceptron----------------------')
# Fashion MNIST
fashion_mnist = keras.datasets.fashion_mnist 
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
print(f'X_train_full shape--> {X_train_full.shape}')
print(f'X_train_full dtype--> {X_train_full.dtype}')

#scaling the input features (because we're going to use gradient descent) 
# and creating a validation set
X_valid, X_train = X_train_full[:5000]/255.0, X_train_full[5000:]/255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

# plt.imshow(X_train[0], cmap="binary")
# plt.axis('off')
# plt.show()

# adding names to the label values
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

print(f'class_names --> {class_names}')
print(f'\nclass_names[y_train[0] --> {class_names[y_train[0]]}\n')


# n_rows = 4
# n_cols = 10
# plt.figure(figsize=(n_cols * 1.2, n_rows * 1.2))
# for row in range(n_rows):
#     for col in range(n_cols):
#         index = n_cols * row + col
#         plt.subplot(n_rows, n_cols, index + 1)
#         plt.imshow(X_train[index], cmap="binary", interpolation="nearest")
#         plt.axis('off')
#         plt.title(class_names[y_train[index]], fontsize=12)
# plt.subplots_adjust(wspace=0.2, hspace=0.5)
# save_fig('fashion_mnist_plot', tight_layout=False)
# plt.show()


# Creating a model using Sequential API (NEURAL NETWORK)
# model = keras.models.Sequential() #simplest kinf of neural net in Keras; one stack of layers connected sequentially
# model.add(keras.layers.Flatten(input_shape=[28, 28])) # convert each input image into a 1D array
# model.add(keras.layers.Dense(300, activation="relu"))
# model.add(keras.layers.Dense(100, activation="relu"))
# model.add(keras.layers.Dense(10, activation="softmax"))

# ### or ####
# model = keras.models.Sequential([
#     keras.layers.Flatten(input_shape=[28, 28]),
#     keras.layers.Dense(300, activation="relu"),
#     keras.layers.Dense(100, activation="relu"),
#     keras.layers.Dense(10, activation="softmax")
# ])

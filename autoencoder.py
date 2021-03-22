import nnwithlayer as nn
import utils
from sklearn.datasets import load_iris, load_digits
from matplotlib import pyplot as plt
import numpy as np
from keras.datasets import mnist

training_set, testing_set = mnist.load_data()
x_train, y_train = training_set
x_test, y_test = testing_set
#x_train, x_test, y_train1, y_train2 = utils.divide_and_shuffle(iris.data[:1600], utils.one_hot_encode(iris.target)[:1600])

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1] * x_test.shape[2]))

print(np.max(x_train))

x_train = x_train[:]
x_test = x_test[:]

x_train = x_train/255
x_test = x_test/255

x_train_noisy = utils.noisify(x_train, 0.3)
x_test_noisy = utils.noisify(x_test, 0.3)

hdim = 64
hdim2 = 32
hdim3 = 16
lr = 0.5

layers = [nn.FCLayer(x_train.shape[1], hdim, lr), nn.FCLayer(hdim, hdim2, lr), nn.FCLayer(hdim2, hdim3, lr), nn.FCLayer(hdim3, hdim2, lr), nn.FCLayer(hdim2, hdim, lr), nn.FCLayer(hdim, x_train.shape[1], lr)]

whereswaldo = nn.nnet(x_train, x_train, layers)
whereswaldo.train(200, 20)

testnums, testindices = utils.randomselect(x_test)
fig, (toprow, midrow, bottomrow) = plt.subplots(3,5)

for i, spot in enumerate(midrow):
    im = testnums[i].reshape(28,28)
    spot.imshow(im, cmap="gray")
    spot.grid(False)
    spot.set_xticks([])
    spot.set_yticks([])

for i, spot in enumerate(toprow):
    im = x_test[testindices[i]].reshape(28,28)
    spot.imshow(im, cmap="gray")
    spot.grid(False)
    spot.set_xticks([])
    spot.set_yticks([])

for i, spot in enumerate(bottomrow):
    im = whereswaldo.predict(testnums[i]).reshape(28,-1)
    spot.imshow(im, cmap="gray")
    spot.grid(False)
    spot.set_xticks([])
    spot.set_yticks([])

plt.show()

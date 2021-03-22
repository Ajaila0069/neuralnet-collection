import numpy as np
from sklearn.datasets import load_digits, load_iris
from sklearn.model_selection import train_test_split
import pandas as pd

dig = load_iris()
onehot_target = pd.get_dummies(dig.target)
x_train, x_val, y_train, y_val = train_test_split(dig.data, onehot_target, test_size=0.1, random_state=20)

print(x_train.shape)
print(y_train.shape)

class nnet:

    def __init__(self, x, y, lr=0.5):
        self.x = x
        self.y = y
        self.lr = lr

        s1 = 16
        s2 = 24
        s3 = 32

        self.weight1 = np.random.randn(self.x.shape[1], s1)
        self.bias1 = np.zeros((1, s1))
        self.weight2 = np.random.randn(self.weight1.shape[1], s2)
        self.bias2 = np.zeros((1, s2))
        self.weight3 = np.random.randn(self.weight2.shape[1], s3)
        self.bias3 = np.zeros((1, s3))
        self.weight4 = np.random.randn(self.weight3.shape[1], self.y.shape[1])
        self.bias4 = np.zeros((1, self.y.shape[1]))

    def sigmoid(self, s):
        return 1/(1 + np.exp(-s))

    def sigmoid_derv(self, s):
        return s * (1 - s)

    def softmax(self, s):
        exps = np.exp(s - np.max(s, axis=1, keepdims=True))
        return exps/np.sum(exps, axis=1, keepdims=True)

    def cross_entropy(self, pred, real):
        n_samples = real.shape[0]
        res = pred - real
        return res/n_samples

    def error(self, pred, real):
        n_samples = real.shape[0]
        logp = - np.log(pred[np.arange(n_samples), real.argmax(axis=1)])
        loss = np.sum(logp)/n_samples
        return loss

    def forward(self, inputs, weights, biases):
        raw = np.dot(inputs, weights) + biases
        return self.sigmoid(raw)

    def final(self, inputs, weights, biases):
        raw = np.dot(inputs, weights) + biases
        return self.softmax(raw)

    def net_forward(self, data):
        l1 = self.forward(data, self.weight1, self.bias1)
        l2 = self.forward(l1, self.weight2, self.bias2)
        l3 = self.forward(l2, self.weight3, self.bias3)
        l4 = self.final(l3, self.weight4, self.bias4)

        return l1, l2, l3, l4

    def net_back(self, l4, l3, l2, l1):
        true_loss = self.error(l4, self.y)

        loss = self.cross_entropy(l4, self.y)
        l3between = np.dot(loss, self.weight4.T)
        l3raw = l3between * self.sigmoid_derv(l3)
        l2between = np.dot(l3raw, self.weight3.T)
        l2raw = l2between * self.sigmoid_derv(l2)
        l1between = np.dot(l2raw, self.weight2.T)
        l1raw = l1between * self.sigmoid_derv(l1)

        self.weight4 -= self.lr * np.dot(l3.T, loss)
        self.bias4 -= self.lr * np.sum(loss, axis=0, keepdims=True)
        self.weight3 -= self.lr * np.dot(l2.T, l3raw)
        self.bias3 -= self.lr * np.sum(l3raw, axis=0)
        self.weight2 -= self.lr * np.dot(l1.T, l2raw)
        self.bias2 -= self.lr * np.sum(l2raw, axis=0)
        self.weight1 -= self.lr * np.dot(self.x.T, l1raw)
        self.bias1 -= self.lr * np.sum(l1raw, axis=0)

        return true_loss

    def optimize(self):
        lay1, lay2, lay3, lay4 = self.net_forward(self.x)
        loss = self.net_back(lay4, lay3, lay2, lay1)
        return loss

    def train(self, iters, monitor=0):
        for i in range(iters):
            self.loss = self.optimize()
            if monitor != 0:
                if i % (iters/monitor) == 0:
                    print("Iteration ", i)
                    print("Current loss: ", self.loss)

    def think(self, data):
        _, _, _, out = self.net_forward(data)
        return out.argmax()



whereswaldo = nnet(x_train/16.0, np.array(y_train), 0.8)

def get_acc(x, y):
    acc = 0
    for xx,yy in zip(x, y):
        s = whereswaldo.think(xx)
        if s == np.argmax(yy):
            acc +=1
    return acc/len(x)*100

print("Initial accuracy : ", get_acc(x_train/16, np.array(y_train)))
print("Initial Test accuracy : ", get_acc(x_val/16, np.array(y_val)))

epochs = 5000
whereswaldo.train(epochs, 5)

print("Final loss: ", whereswaldo.loss)
print("Training accuracy : ", get_acc(x_train/16, np.array(y_train)))
print("Test accuracy : ", get_acc(x_val/16, np.array(y_val)))

import numpy as np

class rnn:

    def __init__(self, trainx, trainy, n_a):
        self.x = trainx
        self.y = trainy
        self.a = n_a

        self.uniquex = np.unique(self.x).tolist()
        self.x_len = len(self.uniquex)

        self.uniquey = np.unique(self.y).tolist()
        self.y_len = len(self.uniquey)
        self.params = self.init_params(self.a, self.x_len, self.y_len)

    def init_params(self, num_a, num_x, num_y):
        wax = np.random.randn(num_a, num_x)
        waa = np.random.randn(num_a, num_x)
        wya = np.random.randn(num_a, num_x)

        ba = np.zeros((num_a, 1))
        by = np.zeros((num_y, 1))

        params = {"wax" : wax, "waa" : waa, "wya" : wya, "ba" : ba, "by" : by}
        return params

    def sigmoid(self, x):
        return 1 / (1 - np.exp(-x))

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    

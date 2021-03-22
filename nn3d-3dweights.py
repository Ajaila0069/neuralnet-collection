import numpy as np

class nnet:

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1-x)

    def tensordot(self, a, b):
        c = np.zeros((a.shape[0], a.shape[1], b.shape[2]))
        a = a.transform((,,))

    def __init__(self, trainX, trainY):
        self.x = trainX
        self.y = trainY
        seed = 24
        np.random.seed(seed)
        self.weight1 = np.random.rand(self.x.shape[2], self.x.shape[1], 8) - 0.5
        np.random.seed(seed)
        self.weight2 = np.random.rand(self.weight1.shape[2], self.weight1.shape[1], 12) - 0.5
        np.random.seed(seed)
        self.weight3 = np.random.rand(self.weight2.shape[2], self.weight2.shape[1], 6) - 0.5
        np.random.seed(seed)
        self.weight4 = np.random.rand(self.weight3.shape[2], self.weight3.shape[1], self.y.shape[1]) - 0.5

    def think(self, inputs, weights):
        sig = self.sigmoid(np.dot(inputs, weights))
        return sig

    def final_reweight(self, inputs, outputs):
        error = 2 * (self.y - outputs)
        deriv = self.sigmoid_derivative(outputs)
        adjust = np.dot(inputs.T, error * deriv)
        return adjust

    def third_reweight(self, inputs, outputs, outputs2):
        error = 2 * (self.y - outputs)
        derivative = self.sigmoid_derivative(outputs)
        weighted_error = np.dot(error * derivative, self.weight4.T)
        derivative_c = self.sigmoid_derivative(outputs2)
        adjust = np.dot(inputs.T, weighted_error * derivative_c)
        return adjust

    def second_reweight(self, inputs, outputs, outputs2, outputs3):
        error = 2 * (self.y - outputs)
        derivative_e = self.sigmoid_derivative(outputs)
        weighted_error_1 = np.dot(error * derivative_e, self.weight4.T)
        derivative_c = self.sigmoid_derivative(outputs2)
        weighted_error_2 = np.dot(weighted_error_1 * derivative_c, self.weight3.T)
        derivative_a = self.sigmoid_derivative(outputs3)
        adjust = np.dot(inputs.T, weighted_error_2 * derivative_a)
        return adjust

    def first_reweight(self, inputs, outputs, outputs2, outputs3, outputs4):
        error = 2 * (self.y - outputs)
        derivative_g = self.sigmoid_derivative(outputs)
        weighted_error_1 = np.dot(error * derivative_g, self.weight4.T)
        derivative_e = self.sigmoid_derivative(outputs2)
        weighted_error_2 = np.dot(weighted_error_1 * derivative_e, self.weight3.T)
        derivative_c = self.sigmoid_derivative(outputs3)
        weighted_error_3 = np.dot(weighted_error_2 * derivative_c, self.weight2.T)
        derivative_a = self.sigmoid_derivative(outputs4)
        adjust = np.dot(inputs.T, weighted_error_3 * derivative_a)
        return adjust

    def train(self, computer_is_kil):
        self.it = computer_is_kil
        for i in range(computer_is_kil):
            layer1 = self.think(self.x, self.weight1)
            layer2 = self.think(layer1, self.weight2)
            layer3 = self.think(layer2, self.weight3)
            out = self.think(layer3, self.weight4)
            out_adjustment = self.final_reweight(layer3, out)
            layer3_adjustment = self.third_reweight(layer2, out, layer3)
            layer2_adjustment = self.second_reweight(layer1, out, layer3, layer2)
            layer1_adjustment = self.first_reweight(self.x, out, layer3, layer2, layer1)
            self.weight1 += layer1_adjustment
            self.weight2 += layer2_adjustment
            self.weight3 += layer3_adjustment
            self.weight4 += out_adjustment

    def process(self, data):
        return self.think(self.think(self.think(self.think(data, self.weight1), self.weight2), self.weight3), self.weight4)

    def load_weights(self, w1, w2, w3, w4):
        try:
            self.weight1 = w1
            self.weight2 = w2
            self.weight3 = w3
            self.weight4 = w4
            return 1
        except:
            return

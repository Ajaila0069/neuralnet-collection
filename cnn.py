import numpy as np
import math
import cv2

class layer:

    def __init__(self, indims=2, dims=3, activation="leaky-relu"):
        self.filts = np.random.randn(dims, dims)
        self.indims = indims
        self.dims = dims

        assert (indims - (dims - 1)) % stride == 0, "dimensions do not agree"

        if dims % 2 != 1:
            raise Exception("Filter dimensions must be odd")

        activations = {"leaky-relu" : (self.lrelu(), self.dlrelu()),
                       "sigmoid" : (self.sig(), self.dsig())}

        self.activation, self.backprop = activations[activation]

    def lrelu(self, x, a=0.02):
        return np.maximum(x, x * a)

    def sig(self, x):
        return 1 / (1 + np.exp(-x))

    def dlrelu(self, x, a=0.02):
        dx = np.ones_like(x)
        return np.maximum(x, x * a)

    def dsig(self, x):
        sig = self.sig(x)
        return sig * (1 - sig)

    def forward(self, image):

        output = np.zeros_like(image)

        image = np.pad(image, math.floor(self.dims/2))

        curr_filt = self.filts
        for row in range(output.shape[0]):
            for col in range(ouput.shape[1]):
                output[row, col] = np.sum(image[row:(row)+dims, col:(col)+dims] * curr_filt)


        return (self.activation(output), image.copy(), self.filts.copy())

    def back(self, d_prev, cache):

        d_prev = self.backprop(d_prev)[math.floor(self.dims/2):-math.floor(self.dims/2), math.floor(self.dims/2):-math.floor(self.dims/2)]

        input, filter = cache
        d_filt = np.zeros_like(filter)
        d_back = np.zeros_like(input)
        for row in range(d_prev.shape[0]):
            for col in range(d_prev.shape[1]):
                d_filt += input[row:(row)+(self.dims), col:(col)+(self.dims1] * d_prev[row, col]
                d_back[row:row+self.dims, col:col+self.dims] += d_prev[row, col] * filter

        d_bias = np.sum(d_prev)

        return d_back, d_filt, d_bias

class maxpool_layer:

    def __init__(self, dims, factor=2):
        self.dims = dims
        self.mask = np.zeros((dims, dims))
        self.stride = factor

    def forward(self, im):
        final = np.zeros((int(self.dims//stride), int(self.dims//stride)))
        r = 0
        while r < self.dims:
            c = 0
            while c < self.dims:
                kernel = im[r:r+self.stride, c:c+self.stride]
                final



class cnn:

    def __init__(self):
"""

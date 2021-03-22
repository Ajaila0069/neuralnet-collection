import numpy as np
import math
import cv2

class layer:

    def __init__(self, indims=2, filts=2, dims=3, stride=1, type="trainable", padding="same", activation="leaky-relu"):
        self.filts = np.zeros((filts, dims, dims))
        self.indims = indims
        self.stride = stride
        self.dims = dims

        assert (indims - (dims - 1)) % stride == 0, "dimensions do not agree"

        if dims % 2 != 1:
            raise Exception("Filter dimensions must be odd")

        if type == "trainable":
            for i in range(filts.shape[0]):
                filts[i] = np.random.randn(dims, dims)

        if type == "edge":
            self.filts[0] = np.array([[0,1,0],[0,1,0],[0,1,0]])
            self.filts[1] = np.array([[0,0,0],[1,1,1],[0,0,0]])

        activations = {"leaky-relu" : (self.lrelu(), self.dlrelu()),
                       "sigmoid" : (self.sig(), self.dsig())}
        self.padding = padding
        self.activation, self.backprop = activations[activation]

    def lrelu(self, x, a=0.02):
        return np.maximum(x, x * 0)

    def sig(self, x):
        return 1 / (1 + np.exp(-x))

    def dlrelu(self, x, a=0.02):
        dx = np.ones_like(x)
        return np.maximum(x, x * a)

    def dsig(self, x):
        sig = self.sig(x)
        return sig * (1 - sig)

    def forward(self, image):

        padding = (self.dims - 1) / 2

        if self.padding == "same":
            np.pad(image, padding)
            output = np.zeros(len(self.filts), image.shape[0], image.shape[1])
        if self.padding == "valid":
            output = np.zeros(len(self.filts), image.shape[0] - dims + 1, image.shape[1] - dims + 1)

        for filt in range(self.filts.shape[0]):
            curr_filt = self.filts[filt, :, :]
            curr_y = out_y = 0
            while curr_y + self.dims <= output.shape[0]:
                curr_x = out_x = 0
                while curr_x + self.dims <= output.shape[1]:
                    output[filt, out_y, out_x] = np.sum(image[filt, curr_y:curr_y+self.dims, curr_x:curr_x+self.dims] * curr_filt)
                    curr_x += self.stride
                    out_x += 1
                curr_y += self.stride
                out_x += 1

        return self.activation(np.sum(output, axis=0)), np.sum(output, axis=0), (image, self.filts)
"""
    def back(self, d_prev, cache):

        for




class cnn:

    def __init__(self):
"""

yoink = layer(type="edge")

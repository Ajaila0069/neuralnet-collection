import numpy as np
import math
import os
from matplotlib import pyplot as plt
import imageio
from keras.datasets import mnist
from tqdm import tqdm

(x_train, y_train), (_, _) = mnist.load_data()
print(x_train.shape)

class GAN:

    def __init__(self, numbers, epochs=100, batch_size=64, k=5, input_g=100, hidden_g=128, hidden_d=128, learn_rate=0.001, decay_rate=0.0001, image_size=28, display=5, create_gif=True, path='samplegifs'):
        self.numbers = numbers
        self.epochs = epochs
        self.batch = batch_size
        self.nx_g = input_g
        self.nh_g = hidden_g
        self.nh_d = hidden_d
        self.lr = learn_rate
        self.dr = decay_rate
        self.im_size = image_size
        self.disp = display
        self.create_gif = create_gif
        self.kval = k

        self.gif_path = path

        self.filenames = []

        self.w0_g = np.random.randn(self.nx_g, self.nh_g) * np.sqrt(2/self.nx_g)
        self.b0_g = np.zeros((1, self.nh_g))
        self.w1_g = np.random.randn(self.nh_g, self.im_size ** 2) * np.sqrt(2/self.nh_g)
        self.b1_g = np.zeros((1, self.im_size ** 2))

        self.w0_d = np.random.randn(self.im_size ** 2, self.nh_d) * np.sqrt(2/self.im_size ** 2)
        self.b0_d = np.zeros((1, self.nh_d))
        self.w1_d = np.random.randn(self.nh_d, 1) * np.sqrt(2/self.nh_d)
        self.b1_d = np.zeros((1, 1))

    def preprocess(self, x, y):
        x_train = []
        y_train = []

        for i in range(y.shape[0]):
            if y[i] in self.numbers:
                x_train.append(x[i])
                y_train.append(y[i])

        x_train = np.array(x_train)
        y_train = np.array(y_train)

        num_batches = x_train.shape[0] // self.batch
        x_train = x_train[: num_batches * self.batch]
        y_train = y_train[: num_batches * self.batch]

        x_train = np.reshape(x_train, (x_train.shape[0], -1))

        x_train = (x_train.astype(np.float32) - 127.5) / 127.5

        idx = np.random.permutation(len(x_train))
        x_train, y_train = x_train[idx], y_train[idx]
        print("processed data")
        return x_train, y_train, num_batches

    def lrelu(self, x, a=0.02):
        return np.maximum(x, x*a)

    def sig(self, x):
        return 1 / (1 + np.exp(-x))

    def dlrelu(self, x, a=0.02):
        dx = np.ones_like(x)
        dx[x < 0] = a
        return dx

    def dsig(self, x):
        sig = self.sig(x)
        return sig * (1 - sig)

    def dtanh(self, x):
        return 1 - (np.tanh(x) ** 2)

    def g_forward(self, z):
        self.z0_g = np.dot(z, self.w0_g) + self.b0_g
        self.a0_g = self.lrelu(self.z0_g, a=0)

        self.z1_g = np.dot(self.a0_g, self.w1_g) + self.b1_g
        self.a1_g = np.tanh(self.z1_g)

        return self.z1_g, self.a1_g

    def d_forward(self, z):
        self.z0_d = np.dot(z, self.w0_d) + self.b0_d
        self.a0_d = self.lrelu(self.z0_d)

        self.z1_d = np.dot(self.a0_d, self.w1_d) + self.b1_d
        self.a1_d = self.sig(self.z1_d)

        return self.z1_d, self.a1_d

    def d_back(self, x_real, z_real, a_real, x_fake, z_fake, a_fake):

        da1_real = -1 / (a_real + 0.00000001)

        dz1_real = da1_real * self.dsig(z_real)
        dw1_real = np.dot(self.a0_d.T, dz1_real)
        db1_real = np.sum(dz1_real, axis=0, keepdims=True)

        da0_real = np.dot(dz1_real, self.w1_d.T)
        dz0_real = da0_real * self.dlrelu(self.z0_d)
        dw0_real = np.dot(x_real.T, dz0_real)
        db0_real = np.sum(dz0_real, axis=0, keepdims=True)

        da1_fake = 1 / (1 - a_fake + 0.00000001)

        dz1_fake = da1_fake * self.dsig(z_fake)
        dw1_fake = np.dot(self.a0_d.T, dz1_fake)
        db1_fake = np.sum(dz1_fake, axis=0, keepdims=True)

        da0_fake = np.dot(dz1_fake, self.w1_d.T)
        dz0_fake = da0_fake * self.dlrelu(self.z0_d)
        dw0_fake = np.dot(x_fake.T, dz0_fake)
        db0_fake = np.sum(dz0_fake, axis=0, keepdims=True)

        dw1 = dw1_fake + dw1_real
        db1 = db1_fake + db1_real

        dw0 = dw0_fake + dw0_real
        db0 = db0_fake + db0_real

        self.w0_d -= self.lr * dw0
        self.b0_d -= self.lr * db0

        self.w1_d -= self.lr * dw1
        self.b1_d -= self.lr * db1

    def g_back(self, z, x_fake, z1_fake, a1_fake):

        da1_d = -1 / (a1_fake + 0.00000001)

        dz1_d = da1_d * self.dsig(z1_fake)
        da0_d = np.dot(dz1_d, self.w1_d.T)
        dz0_d = da0_d * self.dsig(self.z0_d)
        dx_d = np.dot(dz0_d, self.w0_d.T)

        dz1_g = dx_d * self.dtanh(self.z1_g)
        dw1_g = np.dot(self.a0_g.T, dz1_g)
        db1_g = np.sum(dz1_g, axis=0, keepdims=True)

        da0_g = np.dot(dz1_g, self.w1_g.T)
        dz0_g = da0_g * self.dlrelu(self.z0_g, a=0)
        dw0_g = np.dot(z.T, dz0_g)
        db0_g = np.sum(dz0_g, axis=0, keepdims=True)

        self.w0_g -= self.lr * dw0_g
        self.b0_g -= self.lr * db0_g

        self.w1_g -= self.lr * dw1_g
        self.b1_g -= self.lr * db1_g

    def sample_images(self, images, epoch, show):
        images = np.reshape(images, (self.batch, self.im_size, self.im_size))

        fig = plt.figure(figsize=(4, 4))

        for i in range(16):
            plt.subplot(4, 4, i + 1)
            plt.imshow(images[i] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')

        # saves generated images in the GAN_sample_images folder
        if self.create_gif:
            current_epoch_filename = os.path.join(self.gif_path, "GAN_epoch{}.png".format(epoch))
            self.filenames.append(current_epoch_filename)
            plt.savefig(current_epoch_filename)

        if show == True:
            plt.show()
        else:
            plt.close()

    def generate_gif(self):
        images = []
        for filename in self.filenames:
            images.append(imageio.imread(filename))
        imageio.mimsave("gan.gif", images)

    def train(self, x, y):

        J_Ds = []
        J_Gs = []

        x_train, _, num_batches = self.preprocess(x, y)

        for epoch in range(self.epochs):
            for i in tqdm(range(num_batches)):

                x_real = x_train[i * self.batch : (i+1) * self.batch]
                z = np.random.normal(0, 1, size=[self.batch, self.nx_g])

                z1_g, x_fake = self.g_forward(z)

                z1_d_real, a1_d_real = self.d_forward(x_real)
                z1_d_fake, a1_d_fake = self.d_forward(x_fake)

                J_D = np.mean(-np.log(a1_d_real) - np.log(1 - a1_d_fake))
                J_Ds.append(J_D)

                J_G = np.mean(-np.log(a1_d_fake))
                J_Gs.append(J_G)

                #for i in range(self.kval):
                self.d_back(x_real, z1_d_real, a1_d_real, x_fake, z1_d_fake, a1_d_fake)
                #self.d_back(x_real, z1_d_real, a1_d_real, x_fake, z1_d_fake, a1_d_fake)
                self.g_back(z, x_fake, z1_d_fake, a1_d_fake)
                #self.g_back(z, x_fake, z1_d_fake, a1_d_fake)

            if epoch % self.disp == 0:
                print(f"Epoch:{epoch:}|G loss:{J_G:.4f}|D loss:{J_D:.4f}|D(G(z))avg:{np.mean(a1_d_fake):.4f}|D(x)avg:{np.mean(a1_d_real):.4f}|LR:{self.lr:.6f}")
                self.sample_images(x_fake, epoch, show=False)
            else:
                self.sample_images(x_fake, epoch, show=False)

            self.lr *= 1 / (1 + (self.dr * epoch))

            print("epoch: {}".format(epoch))

        if self.create_gif:
            self.generate_gif()

        return J_Ds, J_Gs

numbers = [9]
whereswaldo = GAN(numbers, epochs=50)
J_Ds, J_Gs = whereswaldo.train(x_train, y_train)

plt.plot([i for i in range(len(J_Ds))], J_Ds)
plt.plot([i for i in range(len(J_Gs))], J_Gs)

plt.xlabel("Training Step")
plt.ylabel("Cost")
plt.legend(['Discriminator', 'Generator'])

plt.show()

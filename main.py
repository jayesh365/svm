import numpy as np
from matplotlib import pyplot as plt
import torch
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_blobs, make_circles
from ipywidgets import interact, fixed
np.random.seed(123456789)


def generate_samples_linear(num_samples=100):
    X = np.random.normal(0,1, (num_samples, 2))
    # create our one hot encoding
    y = np.array([ -1 if x[0] > 0 else 1 for x in X])
    return X, y

def generate_samples_non_linear(num_samples):
    # X, y = make_blobs(n_samples=num_samples, centers=4, n_features=2, random_state=1)
    X, y = make_circles(n_samples=num_samples, noise=0.06, random_state=42)
    print('X shape', X.shape, 'y shape', y.shape)
    return X, y


def kernel_trick(X):
    # here we apply the kenerel trick to non linearly seperable data
    # we map the data to a higher dimension
    # for circular data we may transfrom by taking the squared norm
    n_samples = X.shape[0]
    X_ = np.zeros((n_samples, 3)) # 3 as we want 2D -> 3D
    X_[:, :2] = X
    X_[:, 2] = np.sum(X**2, axis=1)
    return X_

def train(X, y):
    X = torch.tensor(X, dtype=torch.float32)
    y_ = np.where(y == 0, -1, y)
    y = torch.tensor(y_, dtype=torch.float32)
    # set dimenson
    dim = X.shape[1]

    # define random weights and bias
    w = torch.autograd.Variable(torch.rand(dim), requires_grad=True)
    b = torch.autograd.Variable(torch.rand(1),   requires_grad=True)

    # define learning rate
    lr = 0.001
    losses = []
    # define number of epochs
    epochs = 5000
    for _ in range(epochs):
        for i, x_i in enumerate(X):
            # calculate the squared hinge loss
            loss = max(0,1 - y[i] * (torch.dot(w, x_i) - b))**2
            # calculate gradients
            if loss != 0 :
                loss.backward()
                # update weights
                w.data -= lr * w.grad.data
                b.data -= lr * b.grad.data
                # zero the gradients
                w.grad.data.zero_()
                b.grad.data.zero_()
                losses.append(loss.data.detach().numpy())
    print(b.data)
    return w.data, b.data, losses



def predict(X, w, b):
    X = torch.tensor(X, dtype=torch.float32)
    return torch.sign(torch.dot(w, X) - b)

if __name__ == '__main__':
    # generate samples from scikit learn to create clusters with 4 centers and 2 features
    X, y = generate_samples_non_linear(500)
    # X, y = generate_samples_linear(100)

    X_ = kernel_trick(X)


    ax = plt.figure().add_subplot(projection='3d')
    ax.scatter3D(X_[:, 0], X_[:, 1], X_[:, 2], c=y)

    # plt.show()

    output = train(X_, y)
    
    w, b = output[0], output[1]
    loss = output[2]

    slope = -w[0]/w[1]
    intercept = -b/w[1]
    # print(slope.data, intercept.data)
    # print(np.floor(min(X[:, 0])), np.ceil(max(X[:, 1])))
    min_x = X_[:, 0].min() - 1
    max_x = X_[:, 0].max() + 1

    min_y = X_[:, 1].min() - 1
    max_y = X_[:, 1].max() + 1
    # x = np.arange(min_x, max_x, 0.1)
    # print(x)
    # x = np.arange(-0.05,0.05, 0.001)


    # classes = [-1, 1]
    # values = y
    # colors = ListedColormap(['y','m'])
    # plt.plot(x, slope * x + b)
    # plt.scatter(X[:, 0], X[:, 1], c=values, cmap=colors)

    # test = torch.tensor([0.5, 0.5], dtype=torch.float32)
    # test1 = torch.tensor([-0.5, 0.5], dtype=torch.float32)
    # test2 = torch.tensor([-1.0, -0.5], dtype=torch.float32)
    # prediction = predict(test, w, b)
    # prediction1 = predict(test1, w, b)
    # prediction2 = predict(test2, w, b)
    # print('p1 ', prediction.data.detach().numpy())
    # print('p2 ', prediction1.data.detach().numpy())
    # print('p3', prediction2.data.detach().numpy())
    # plt.scatter(0.5, 0.5, c='r')
    # plt.scatter(-0.5, -0.5, c='g')
    # plt.scatter(-1.0, -0.5, c='b')
    
    # plt.show()
    # plt.plot(loss, c='red')
    # plt.show()

    xx, yy = np.meshgrid(np.arange(min_x, max_x, 0.1),
                        np.arange(min_y, max_y, 0.1))
    
    zz = (-w.data[0] * xx - w.data[1] * yy + b.data) / w.data[2]

    ax.plot_surface(xx, yy, zz, alpha=0.5)

    plt.show()





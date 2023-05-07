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
    epochs = 500
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
    return w.data, b.data, losses


def decision_plane(X, w, b):

    # apply kernel trick
    X = kernel_trick(X)

    # get the min and max values of the data
    min_x = X[:, 0].min() - 1
    max_x = X[:, 0].max() + 1

    min_y = X[:, 1].min() - 1
    max_y = X[:, 1].max() + 1

    # create a meshgrid of points with the above min and max values
    xx, yy = np.meshgrid(np.arange(min_x, max_x, 0.1),
                        np.arange(min_y, max_y, 0.1))

    # calculate the decision boundary
    zz = (-w[0] * xx - w[1] * yy + b) / w[2]

    return xx ,yy, zz


def margin(X, w, b):
    ''' returns the margin '''

    # calculate the distance from the decision plane to the closest vectors
    # calcualte the margin
    w_normalized = w / np.linalg.norm(w)
    distance = np.abs(np.dot(X, w_normalized.detach().numpy()) - b.detach().numpy())
    margin = 1/np.max(distance)
    return margin


def _support_vectors(X, w, b):
    ''' returns the support vectors '''

    marg = margin(X, w, b)
    predictions = predict(X, w, b)
    support_vectors_inds = []
    for ind, decision in enumerate(predictions):
        # get the vectors within the margin of the decision plane
        if abs(decision) <= marg:
            support_vectors_inds.append(ind)
    return support_vectors_inds

def predict(X, w, b):
    X = torch.tensor(X, dtype=torch.float32)
    return np.dot(X, w) - b.detach().numpy()

if __name__ == '__main__':
    # generate samples from scikit learn to create clusters with 4 centers and 2 features
    X, y = generate_samples_non_linear(500)
    # X, y = generate_samples_linear(100)

    X_ = kernel_trick(X)
    # X = kernel_trick(X)


    ax = plt.figure().add_subplot(projection='3d')
    # ax.scatter3D(X_[:, 0], X_[:, 1], X_[:, 2], c=y, cmap='autumn')

    # plt.show()

    output = train(X_, y)
    sv_i = _support_vectors(X_, output[0], output[1])
    s_v = X_[sv_i]

    xx, yy, zz = decision_plane(X, output[0], output[1])
    ax.plot_surface(xx, yy, zz, alpha=0.2)
    ax.scatter3D(s_v[:, 0], s_v[:, 1], s_v[:, 2],s= 2.5,cmap='winter', c=y[sv_i])
    plt.show()
    # w, b = output[0], output[1]
    # loss = output[2]

    # slope = -w[0]/w[1]
    # intercept = -b/w[1]
    # print(slope.data, intercept.data)
    # print(np.floor(min(X[:, 0])), np.ceil(max(X[:, 1])))
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

    # ax.plot_surface(xx, yy, zz, alpha=0.5)

    # plt.show()





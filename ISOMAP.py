import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import random
from scipy.io import loadmat
from Matrix_D import Matrix_D
from PCA import PCA


class Isomap:
    """
    reproduce the ISOMAP algorithm results in the original paper
    """
    def __init__(self, k=2, n=100, distance='euclidean'):
        self.k = k
        # at least have n neighbors
        self.n = n
        self.distance = distance
        
    # calculate euclidean distance of
    # 2 vectors or (1 vector and many vectors)
    # with vector broadcast feature
    def euc_distance(self, x, y):
        return np.sqrt(np.sum(np.square(x - y), axis=-1))
    
    # calculate manhattan distance of
    # 2 vectors or (1 vector and many vectors)
    # with vector broadcast feature
    def manhattan_distance(self, x, y):
        return np.sum(np.abs(x - y), axis=-1)
    
    def dis_matrix(self, x, distance):
        if distance == 'euclidean':
            dis_func = self.euc_distance
        elif distance == 'manhattan':
            dis_func = self.manhattan_distance
        else:
            raise RuntimeError('Please use available distance function.')
        dim, num = x.shape[0], x.shape[1]
        d = np.zeros((num, num))
        for i in range(num):
            d[i] = dis_func(x[:, i], x.T)
        return d
    
    def find_epsilon(self, dis_matrix):
        num = dis_matrix.shape[0]
        anchor = 0
        for i in range(num):
            anchor = max(sorted(dis_matrix[i])[self.n], anchor)
        return anchor
    
    def adj_matrix(self, x):
        d = self.dis_matrix(x, self.distance)
        epsilon = self.find_epsilon(d)
        d[d > epsilon] = 0
        plt.matshow(d)
        return d
    
    def shortest_path(self, adj_matrix):
        # set unexisting edges weight as inf
        adj_matrix[adj_matrix == 0] = np.inf
        return Matrix_D(adj_matrix)
    
    def reduced_representation(self, D):
        num = D.shape[0]
        H = np.identity(num) - np.ones((num, num)) / num
        C = (-1/2) * H.dot(D**2).dot(H)
        # compute eigenvalues and eigenvectors
        val, vec = np.linalg.eigh(C)
        # find the first k principal directions
        idx_largestTwo_eigval = np.argsort(val)[-self.k:]
        eigvals = []
        W = np.zeros((num, self.k))
        for i in range(self.k):
            idx = self.k - i - 1
            eig_val = val[idx_largestTwo_eigval[idx]]
            eig_vec = vec[:, idx_largestTwo_eigval[idx]]
            eigvals.append(eig_val)
            W[:, i] = eig_vec
        eigvals = np.array(eigvals)
        Z = W.dot(np.diag(eigvals**(-1/2)))
        return Z
    
    def forward(self, data):
        adj = self.adj_matrix(data)
        D = self.shortest_path(adj)
        return self.reduced_representation(D)

# scatter plot
def plot_faces_distribution(Z, path):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(Z[:, 0], Z[:, 1], alpha = 0.6)
    random.seed(12)
    random_plots = [random.randint(0, Z.shape[0]) for _ in range(10)]
    for i in random_plots:
        ab = AnnotationBbox(OffsetImage(np.reshape(images[:, i], (64, 64)), cmap='gray'), (Z[i, 0], Z[i, 1]))
        ax.add_artist(ab)
    # ax.figure.savefig('result/PCA_reduced_representation.png')
    fig.tight_layout()
    plt.savefig(path)


if __name__ == '__main__':
    # loading data
    images = loadmat('./data/isomap.mat')['images']
    # print(images.shape)
    # (4096, 698)

    # plot one image in dataset
    # plt.imshow(np.reshape(images[:, 0], (64, 64)))

    model1 = Isomap(k=2, distance='euclidean')
    # model2 = Isomap(k=2, distance='manhattan')

    Z1 = model1.forward(images)
    # print(Z1.shape)
    # (698, 2)

    plot_faces_distribution(Z1, 'result/Isomap.png')

    # PCA
    pca_model = PCA(k=2, normalize=False)
    reduced_matrix = pca_model.forward(images)
    plot_faces_distribution(reduced_matrix.T, 'result/pca_faces.png')

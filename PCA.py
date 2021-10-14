import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class PCA:
    """
    reduce data (d, m) dimension to (k, m)
    """
    def __init__(self, k=2, normalize=True):
        # reduced to k dimensions
        self.k = k
        self.normalize = normalize
    
    def normalize_data(self, data):   
        row_mean = np.mean(data, axis=1)
        row_std = np.std(data, axis=1)
        data = (data - row_mean.reshape(-1, 1)) / row_std.reshape(-1,1)
        return data
    
    def compute_mean_cov(self, data):
        # compute mean
        mean = data.sum(axis=1) / data.shape[1]
        # compute covarivance data
        cov_matrix = np.matmul(data-(mean.reshape(-1, 1)), (data-(mean.reshape(-1, 1))).T) / data.shape[1]
        return mean, cov_matrix
    
    def compute_eig(self, cov_matrix):
        num = cov_matrix.shape[0]
        # compute eigenvalues and eigenvectors
        val, vec = np.linalg.eigh(cov_matrix)
        # find the first k principal directions
        idx_largestTwo_eigval = np.argsort(val)[-self.k:]
        self.eigvals = []
        self.W = np.zeros((num, self.k))
        for i in range(self.k):
            idx = self.k - i - 1
            eig_val = val[idx_largestTwo_eigval[idx]]
            eig_vec = vec[:, idx_largestTwo_eigval[idx]]
            self.eigvals.append(eig_val)
            self.W[:, idx] = eig_vec
        return self.eigvals, self.W
    
    def forward(self, data):
        if self.normalize:
            data = self.normalize_data(data)
        mean, cov_matrix = self.compute_mean_cov(data)
        eigvals, eigvecs = self.compute_eig(cov_matrix)
        # compute reduced matrix
        reduced_matrix = np.zeros((self.k, data.shape[1]))
        for i in range(reduced_matrix.shape[1]):
            for j in range(reduced_matrix.shape[0]):
                eigvec = eigvecs[:, j].reshape(-1, 1)
                sample = data[:, i].reshape(-1, 1)
                reduced_matrix[j][i] = eigvec.T.dot((sample - mean.reshape(-1, 1))) / np.sqrt(eigvals[j])
        return reduced_matrix

if __name__ == '__main__':
    # load data
    csv_data = pd.read_csv('./data/food-consumption.csv')
    # print(csv_data.shape)
    # (16, 21)

    # convert to numpy.arrays
    # convert data type to be dxm
    data = csv_data.iloc[:, 1:].values.T
    # print(data.shape)
    # (20, 16)

    model = PCA(k=2)
    reduced_matrix = model.forward(data)
    pd1 = model.W[:, 0]
    pd2 = model.W[:, 1]

    # combine reduced data with country name
    new_matrix = pd.DataFrame(reduced_matrix.T)
    new_data = pd.concat([csv_data.iloc[:, 0], new_matrix], axis=1)

    # stem plots for first principal directions
    plt.figure(figsize=(25, 10))
    plt.stem(pd1, use_line_collection=True)
    plt.xticks(range(pd1.shape[0]), labels=csv_data.columns.values.tolist()[1:])
    plt.savefig('result/principal_direction.png')

    # scatter plot with country name
    plot = new_data.plot.scatter(x=0, y=1, alpha=0.5)
    for i, label in enumerate(new_data.Country):
        plot.annotate(label, (new_data.iloc[i, 1], new_data.iloc[i, 2]))
    plot.figure.savefig('result/PCA_reduced_representation.png')

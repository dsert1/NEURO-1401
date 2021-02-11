# @authors: Deniz B. Sert, Shawn _____
# with inspiration from the Internet
# @version: February 8, 2021

import numpy as np

class RBF_Network(object):

    def __init__(self, hidden_shape, sigma=1.0):
        '''Radial Bases Function (RBF) Network
        # ARGUMENTS:
            input_shape: dimension of input data
            e.g scalar functions should have input_dimension = 1
            hidden_shape: number of hidden rbfs, or number of centers
        '''
        self.hidden_shape = hidden_shape
        self.sigma = sigma
        self.centers = None
        self.weights = None


    def kernel_function(self, center, data_point):
        # return np.exp(-np.linalg.norm(center-data_point)**2 / self.sigma)
        return np.exp(-self.sigma * np.linalg.norm(center - data_point) ** 2)

    def calculate_interpolation_matrix(self, X):
        '''calculates interpolation matrix using a kernel_function
        # ARGUMENTS:
            X: training data
        # INPUT SHAPE
            (num_data_samples, input_shape)

        # RETURNS
            G: Interpolation matrix
        '''

        G = np.zeros((len(X), self.hidden_shape))
        outer_count = 0
        inner_count = 0

        print(len(X))

        for data_point_ix, data_point in enumerate(X):
            outer_count += 1
            if outer_count % 500000 == 0: print('OUTER COUNT: ', outer_count)
            for center_arg, center in enumerate(self.centers):
                inner_count += 1
                if inner_count % 500000 == 0: print('INNER COUNT: ', inner_count)
                G[data_point_ix, center_arg] = self.kernel_function(center, data_point)
        return G

    def select_centers(self, X):
        random_args = np.random.choice(len(X), self.hidden_shape)
        centers = X[random_args]
        return centers

    def fit(self, X, Y):
        '''fits weights using linear regression
        # ARGUMENTS:
            X: training samples
            Y: targets

        # INPUT SHAPE
            X: (num_data_samples, input_shape)
            Y: (num_data_samples, input_shape)
        '''
        # print(X)
        self.centers = self.select_centers(X)
        G = self.calculate_interpolation_matrix(X)
        self.weights = np.dot(np.linalg.pinv(G), Y) # .pinv finds the Moore-Penrose inverse, or the pseudoinverse of a matrix.
        # the pseudomatrix is the closest approximation of a solution for a system with no solutions

    def predict(self, X):
        '''
        # ARGUMENTS:
            X: test data

        # INPUT SHAPE
            (num_test_samples, input_shape)
        '''
        G = self.calculate_interpolation_matrix(X)
        predictions = np.dot(G, self.weights)
        return predictions
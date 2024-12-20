import numpy as np
from scipy.linalg import expm

class parent_matrix:
    def __init__(self, f, H):
        """
        Parameters:
        f (numpy.ndarray): 1-D array
        H (numpy.ndarray): 2-D array
        """
        if H.ndim != 2 or H.shape[0] != H.shape[-1]:
            raise ValueError(f"H must be a 2-D square numpy array.")
        
        # Validate f and H sizes
        if H.shape[0] != f.shape[0]:
            raise ValueError(f"f and H must be the same size. Got f: {f.shape[0]} and H: {H.shape[0]}")
        
        # row_sums = np.sum(H, axis=1) - np.diag(H)

        # if not np.allclose(np.diag(H), row_sums):
        #     raise ValueError("H does not satisfy the condition H_jj = sum_{i!=j} H_ij.")

        
        self.size = H.shape[0]
        self.f = f
        self.H = H
        self.matrix = self._construct_matrix()
        self.eigenvalues, self.eigenvectors = np.linalg.eig(self.matrix)

    def _construct_matrix(self):
        """
        Construct the growth rate matrix

        Returns:
        numpy.ndarray: The constructed matrix.
        """
        matrix = self.H.copy()
        for i in range(self.size):
            matrix[i, i] = self.f[i] - self.H[i, i]
        return matrix

    def exponent_at_time(self, t):
        """
        Calculate the exponent of the growth rate matrix at time t.

        Parameters:
        t (float): The time at which to calculate the matrix exponent.

        Returns:
        numpy.ndarray: The matrix exponential of the growth rate matrix at time t.
        """
        exp = self.eigenvectors @ np.diag(np.exp(self.eigenvalues * t)) @ np.linalg.inv(self.eigenvectors)
        return exp

    def show_matrix(self):
        return self.matrix

class stochastic_growth_rate(parent_matrix):
    def __init__(self, f, H):
        """
        Parameters:
        f (numpy.ndarray): 1-D array
        H (numpy.ndarray): 2-D array
        """

        super().__init__(f, H)

class responsive_growth_rate(parent_matrix):
    def __init__(self, f, H_m, row_ind):
        """
        Parameters:
        f (numpy.ndarray): 1-D array
        H_m (float): Value for the elements in the specified row
        row_ind (int): Index of the row to be initialized with H_m
        """
        len_f = f.shape[0]
        H = np.zeros((len_f, len_f))

        # Set all elements in the specified row to H_m
        H[row_ind, :] = H_m

        # Set the diagonal element in the specified row to f
        H[row_ind, row_ind] = f[row_ind]

        # Set other diagonal elements to f - H_m
        for i in range(len_f):
            if i != row_ind:
                H[i, i] = f[i] - H_m

        super().__init__(f, H)

class ProbabilityMatrix:
    def __init__(self, k, matrix=None):
        self.k = k
        if matrix is None:
            # Default k by k probability matrix with equal probabilities and zero diagonals
            self.matrix = np.full((k, k), 1 / (k - 1))
            np.fill_diagonal(self.matrix, 0)
        else:
            self.matrix = matrix
            self._validate_matrix()

    def _validate_matrix(self):
        if self.matrix.shape != (self.k, self.k):
            raise ValueError(f"The matrix must be of size {self.k} x {self.k}.")
        if not np.all(self.matrix >= 0):
            raise ValueError("All elements must be non-negative.")
        if not np.allclose(self.matrix.sum(axis=1), 1):
            raise ValueError("Each row must sum to 1.")
        if not np.allclose(np.diag(self.matrix), 0):
            raise ValueError("All diagonal elements must be zero.")

    def set_matrix(self, matrix):
        self.matrix = matrix
        self._validate_matrix()

    def get_matrix(self):
        return self.matrix

    def transition(self, state):
        return np.dot(state, self.matrix)
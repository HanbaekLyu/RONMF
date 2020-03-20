# (setq python-shell-interpreter "./venv/bin/python")


# import tensorflow as tf
import numpy as np
import progressbar
# import imageio
import matplotlib.pyplot as plt
from numpy import linalg as LA
from time import time

from sklearn.decomposition import SparseCoder





DEBUG = False


class Online_RNMF():

    def __init__(self,
                 X,
                 n_components=100,
                 iterations=500,
                 iter_sparsecoding_approx=10,
                 batch_size=20,
                 ini_dict=None,
                 ini_A=None,
                 ini_B=None,
                 history=0,
                 alpha=None,
                 beta=None):
        '''
        Algorithm for Online Robust NMF
        X: data matrix
        Goal: X \approx S + WH
        S : Noise
        W : Dictionary
        H : Code
        n_components (int): number of columns in dictionary matrix W where each column represents on topic/feature
        iter (int): number of iterations where each iteration is a call to step(...)
        batch_size (int): number random of columns of X that will be sampled during each iteration
        '''
        self.X = X
        self.n_components = n_components
        self.batch_size = batch_size
        self.iterations = iterations
        self.iter_sparsecoding_approx = iter_sparsecoding_approx
        self.initial_dict = ini_dict
        self.initial_A = ini_A
        self.initial_B = ini_B
        self.history = history
        self.alpha = alpha
        self.beta = beta
        self.code = np.zeros(shape=(n_components, X.shape[1]))

    def sparse_code_admm(self, X, W):
        '''
        Given data matrix X and dictionary matrix W, find 
        code matrix H and noise matrix S such that
        H, S = argmin ||X - WH - S||_{F}^2 + \alpha ||H||_{1}  + \beta ||S||_{1}
        Uses ADMM for hard constraint X - WH - S = 0

        args:
            X (numpy array): data matrix with dimensions: features (d) x samples (n)
            W (numpy array): dictionary matrix with dimensions: features (d) x topics (r)

        returns:
            H (numpy array): code matrix with dimensions: topics (r) x samples(n)
            S (numpy array): noise matrix with dimensions: features (d) x samples (n)
        '''

        if DEBUG:
            print('sparse_code')
            print('X.shape:', X.shape)
            print('W.shape:', W.shape, '\n')

        # initialize the SparseCoder with W as its dictionary
        # H_new = LASSO with W as dictionary
        # S_new = LASSO with id (d x d) as dictionary
        # Y_new = Y + (W H_new + S_new - S) : Dual variable

        ### Initialization
        d, n = X.shape
        r = self.n_components
        H = np.random.rand(r, n)
        S = np.zeros(shape=(d, n))
        Y = np.zeros(shape=(d, n))

        for step in np.arange(self.iter_sparsecoding_approx):
            ### update H
            if self.alpha == None:
                coder = SparseCoder(dictionary=W.T, transform_n_nonzero_coefs=None,
                                    transform_alpha=2, transform_algorithm='lasso_lars', positive_code=True)
            else:
                coder = SparseCoder(dictionary=W.T, transform_n_nonzero_coefs=None,
                                    transform_alpha=self.alpha, transform_algorithm='lasso_lars', positive_code=True)
            H = coder.transform(np.transpose(X-S-Y))
            H = H.T

            ### update S
            '''
            if self.beta == None:
                coder = SparseCoder(dictionary=np.identity(d), transform_n_nonzero_coefs=None,
                                    transform_alpha=2, transform_algorithm='lasso_lars', positive_code=True)
            else:
                coder = SparseCoder(dictionary=np.identity(d), transform_n_nonzero_coefs=None,
                                    transform_alpha=self.beta, transform_algorithm='lasso_lars', positive_code=True)
            '''
            if self.beta is None:
                S = self.soft_thresholding(X - W @ H - Y, 2)
            else:
                S = self.soft_thresholding(X - W @ H - Y, self.beta)

            ### update Y
            Y = Y + (W @ H + S - X)

        # transpose H before returning to undo the preceding transpose on X
        return H, S

    def sparse_code(self, X, W):
        '''
        Given data matrix X and dictionary matrix W, find
        code matrix H and noise matrix S such that
        H, S = argmin ||X - WH - S||_{F}^2 + \alpha ||H||_{1}  + \beta ||S||_{1}
        Uses proximal gradient

        G = [H \\ S']
        V = [W, b I] (so that VG = WH + bS')

        Then solve

        min_{G,V} |X-VG|_{F} + \alpha |G|_{1} =
        = min_{H,S'} |X - HW - bS'|_{F}^2 + \alpha |H|_{1} + \alpha |S'|_{1}
        = min_{H,S} |X - HW - S|_{F}^2 + \alpha |H|_{1} + (\alpha/b)|S|_{1}

        using constrained LASSO

        args:
            X (numpy array): data matrix with dimensions: features (d) x samples (n)
            W (numpy array): dictionary matrix with dimensions: features (d) x topics (r)

        returns:
            H (numpy array): code matrix with dimensions: topics (r) x samples(n)
            S (numpy array): noise matrix with dimensions: features (d) x samples (n)
        '''

        if DEBUG:
            print('sparse_code')
            print('X.shape:', X.shape)
            print('W.shape:', W.shape, '\n')

        # initialize the SparseCoder with W as its dictionary
        # H_new = LASSO with W as dictionary
        # S_new = LASSO with id (d x d) as dictionary
        # Y_new = Y + (W H_new + S_new - S) : Dual variable

        ### Initialization
        d, n = X.shape
        r = self.n_components

        ### set reguiarization parameters
        if self.alpha == None:
            a = 2
        else:
            a = self.alpha
        if self.beta == None:
            b = 2
        else:
            b = self.beta

        ### Augmented dictionary matrix for proximal gradient
        V = np.hstack((W, b*np.identity(d)))

        ### Proximal sparse coding by constrained LASSO
        coder = SparseCoder(dictionary=V.T, transform_n_nonzero_coefs=None,
                            transform_alpha=a, transform_algorithm='lasso_lars', positive_code=True)
        G = coder.transform(X.T)
        G = G.T
        # transpose G before returning to undo the preceding transpose on X

        ### Read off H and S from V
        H = G[0:r, :]
        S = b*G[r:, :]

        return H, S

    def soft_thresholding(self, y, threshold):
        y[np.where((-threshold <= y) & (y <= threshold))] = 0
        y[np.where(y > threshold)] -= threshold
        y[np.where(y < -threshold)] += threshold
        return y

    def update_dict(self, W, A, B):
        '''
        Updates dictionary matrix W using new aggregate matrices A and B

        args:
            W (numpy array): dictionary matrix with dimensions: data_dim (d) x topics (r)
            A (numpy array): aggregate matrix with dimensions: topics (r) x topics(r)
            B (numpy array): aggregate matrix with dimensions: topics (r) x data_dim (d)

        returns:
            W1 (numpy array): updated dictionary matrix with dimensions: features (d) x topics (r)
        '''
        # extract matrix dimensions from W
        # and initializes the copy W1 that is updated in subsequent for loop
        d, r = np.shape(W)
        W1 = W.copy()

        #****
        for j in np.arange(r):
                # W1[:,j] = W1[:,j] - (1/W1[j,j])*(np.dot(W1, A[:,j]) - B.T[:,j])
                W1[:,j] = W1[:,j] - (1/(A[j,j]+1) )*(np.dot(W1, A[:,j]) - B.T[:,j])
                W1[:,j] = np.maximum(W1[:,j], np.zeros(shape=(d, )))
                W1[:,j] = (1/np.maximum(1, LA.norm(W1[:,j])))*W1[:,j]
        
        return W1


    def step(self, X, A, B, W, t):
        '''
        Performs a single iteration of the online NMF algorithm from
        Han's Markov paper. 
        Note: H (numpy array): code matrix with dimensions: topics (r) x samples(n)

        args:
            X (numpy array): data matrix with dimensions: data_dim (d) x samples (n)
            A (numpy array): aggregate matrix with dimensions: topics (r) x topics(r)
            B (numpy array): aggregate matrix with dimensions: topics (r) x data_dim (d)
            W (numpy array): dictionary matrix with dimensions: data_dim (d) x topics (r)
            t (int): current iteration of the online algorithm
        
        returns:
            Updated versions of H, A, B, and W after one iteration of the online NMF
            algorithm (H1, A1, B1, and W1 respectively)
        '''
        d, n = np.shape(X)
        d, r = np.shape(W)
        
        # Compute H1 by sparse coding X using dictionary W
        H1, S1 = self.sparse_code(X, W)

        if DEBUG:
            print(H1.shape)

        # Update aggregate matrices A and B
        A1 = (1/t)*((t-1)*A + np.dot(H1, H1.T))
        B1 = (1/t)*((t-1)*B + np.dot(H1, (X-S1).T))
        # Update dictionary matrix
        W1 = self.update_dict(W, A, B)
        self.history = t+1
        # print('history=', self.history)
        return H1, S1, A1, B1, W1

    def train_dict(self):
        '''
        Learns a dictionary matrix W with n_components number of columns based 
        on a fixed data matrix X
        
        args:
            X (numpy array): data matrix with dimensions: data_dim (d) x samples (n)


        return:
            W (numpy array): dictionary matrix with dimensions: data_dim (d) x topics (r)
        '''
        # extract matrix dimensions from X
        d, n = np.shape(self.X)
        r = self.n_components
        code = self.code

        if self.initial_dict is None:
            # initialize dictionary matrix W with random values
            # and initialize aggregate matrices A, B with zeros
            W = np.random.rand(d, r)
            A = np.zeros((r, r))
            B = np.zeros((r, d))
            t0 = self.history
        else:
            W = self.initial_dict
            A = self.initial_A
            B = self.initial_B
            t0 = self.history

        for i in np.arange(1, self.iterations):
            # randomly choose batch_size number of columns to sample
            idx = np.random.randint(n, size=self.batch_size)

            # initializing the "batch" of X, which are the subset
            # of columns from X that were randomly chosen above
            X_batch = self.X[:, idx]
            
            # iteratively update W using batches of X, along with
            # iteratively updated values of A and B
            H, S, A, B, W = self.step(X_batch, A, B, W, t0+i)
            code[:, idx] += H
            # print('dictionary=', W)
            # print('code=', H)
            # plt.matshow(H)
        # print('iteration %i out of %i' % (i, self.iterations))
        return W, A, B, code


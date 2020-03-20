from ornmf import Online_RNMF
import itertools
import numpy as np
from PIL import Image
from skimage.transform import downscale_local_mean
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from tensorly import unfold as tl_unfold
from sklearn.decomposition import SparseCoder

from time import time

import matplotlib.pyplot as plt

DEBUG = False


class Image_Reconstructor_ornmf():
    def __init__(self,
                 path,
                 n_components=100,
                 iterations=200,
                 iter_sparsecoding_approx=10,
                 sub_iterations=20,
                 num_patches=1000,
                 batch_size=20,
                 downscale_factor=2,
                 patch_size=7,
                 patches_file='',
                 noise_rate=0.1,
                 patches = None,
                 is_matrix=False,
                 is_stack=False,
                 is_color=True,
                 beta = None):
        '''
        batch_size = number of patches used for training dictionaries per ONMF iteration
        sources: array of filenames to make patches out of
        patches_array_filename: numpy array file which contains already read-in images
        '''
        self.path = path
        self.n_components = n_components
        self.iterations = iterations
        self.iter_sparsecoding_approx = iter_sparsecoding_approx
        self.sub_iterations = sub_iterations
        self.num_patches = num_patches
        self.batch_size = batch_size
        self.downscale_factor = downscale_factor
        self.patch_size = patch_size
        self.patches = patches
        self.patches_file = patches_file
        self.is_matrix = is_matrix
        self.is_stack = is_stack  # if True, input data is a 3d array
        self.is_color = is_color
        self.W = np.zeros(shape=(patch_size, n_components))
        self.code = np.zeros(shape=(n_components, iterations*batch_size))
        self.beta = beta  ### noise regularization parameter
        self.noise_rate = noise_rate

        '''
        # read in patches
        if patches_file:
            self.patches = np.load(patches_file)
        elif not is_stack:
            self.patches = self.read_patches(patch_size=patch_size, iterations=iterations,
                                             batch_size=batch_size, is_matrix=is_matrix)
            print(self.patches.shape)
        else:
            self.patches = self.read_patches_stack(patch_size=patch_size, iterations=iterations,
                                                   batch_size=batch_size)
            b = np.asarray(self.patches)
            b = np.moveaxis(b, 0, -1)
            self.patches = b
            print(self.patches.shape)
        '''
        # read in image as array
        self.data = self.read_img_as_array(self.path)
        if self.noise_rate > 0:
            self.data = self.add_noise2img()

    def read_img_as_array(self, path):
        '''
        Read input image as a narray
        '''

        if self.is_matrix:
            img = np.load(path)
            data = (img + 1) / 2  # it was +-1 matrix; now it is 0-1 matrix
        else:
            img = Image.open(path)
            if self.is_color:
                img = img.convert('RGB')
                #  = data.reshape(-1, data.shape[1])  # (row, col, 3) --> (3row, col)
            else:
                img = img.convert('L')
            data = np.asarray(img) / 255
            print('img.shape', data.shape)
            # normalize pixel values (range 0-1)
        print('data.shape', data.shape)
        return data

    def add_noise2img(self):
        A = self.data
        p = self.noise_rate
        B = np.random.rand(A.shape[0], A.shape[1])
        A[np.where(B < p)] = (5)*(1/p)*B[np.where(B < p)]  # each entry chosen with prob. p independently
        # Unif[0,1] conditioned on [0,p] is Unif[0,p]. This is the noise matrix.
        # noise ~ Uniform[0,2]
        return A

    def display_img(self):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6),
                                subplot_kw={'xticks': [], 'yticks': []})
        ax.imshow(self.data, cmap="gray", interpolation='nearest')
        p = 100*self.noise_rate
        plt.tight_layout()
        # plt.suptitle('Image with %.0f %% of corruption' % p, fontsize=12)
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig('Data/escher/11.jpg', bbox_inches='tight', pad_inches=0)
        plt.show()

    def read_patches(self, patch_size=7, iterations=500, batch_size=20, is_matrix=False):
        '''
        reads in patches from images specified by self.sources
        '''
        patches = []
        for filename in self.sources:
            print("reading {}".format(filename))
            img_patches = self.image_to_patches(filename, iterations=iterations, batch_size=batch_size,
                                                patch_size=patch_size, is_matrix=is_matrix)
            patches = patches + img_patches.T.tolist()
        return np.array(patches).T

    def read_patches_stack(self, patch_size=7, iterations=500, batch_size=20):
        '''
        reads in patches from stack of matrices specified by self.sources
        '''
        stack_patches = []
        for filename in self.sources:
            print("reading {}".format(filename))
            stack_patches = self.stack_to_patches(filename, patch_size=patch_size, iterations=iterations,
                                                  batch_size=batch_size)
        return np.array(stack_patches)

    def image_to_patches(self, path, patch_size=7, iterations=500, batch_size=20, color=False, downscale_factor=2,
                         is_matrix=False, is_recons=False):
        '''
        #*****

        args:
            path (string): Path and filename of input image
            patch_size (int): Pixel dimension of square patches taken of image
            color (boolean): Specifies conversion of image to RGB (True) or grayscale (False).
                Default value = false. When color = True, images in gray colorspace will still appear
                gray, but will thereafter be represented in RGB colorspace using three channels.
            downscale_factor: Specifies the extent to which the image will be downscaled. Greater values
                will result in more downscaling but faster speed. For no downscaling, use downscale_factor=1.
        returns: #***

        '''
        #open image and convert it to either RGB (three channel) or grayscale (one channel)
        if is_matrix:
            img = np.load(path)
            data = (img + 1) / 2  # it was +-1 matrix; now it is 0-1 matrix
        else:
            img = Image.open(path)
            if color:
                img = img.convert('RGB')
                img = tl_unfold(img, mode=2).T
                print('img.shape_color_after_unfolding', img.shape)
            else:
                img = img.convert('L')
            # normalize pixel values (range 0-1)
            data = np.asarray(img) / 255

        '''
        img = np.load(path)
        data = (img + 1) / 2  # it was +-1 matrix; now it is 0-1 matrix
        '''
        if DEBUG:
            print(np.asarray(img))

        # downscale image for speed
        if downscale_factor > 1:
            data = downscale_local_mean(data, (downscale_factor, downscale_factor))

        # show_array(data)
        if not is_recons:
            patches = extract_patches_2d(data, (patch_size, patch_size), max_patches=iterations*batch_size)
        else:  # for reconstruction we need to use all patches not to mess up with the ordering
            patches = extract_patches_2d(data, (patch_size, patch_size))
        # we do not need more than iterations*batch_size patches for training dictionaries
        # convert 7x7 squares to 1d (49,) vectors
        patches_flat = patches.reshape(len(patches), -1).T
        # (Num patches)*(7*7) array
        # .reshape(len(patches),-1) makes it two dimensional -- (Num patches)*(whatever that's correct)
        # take transpose to make it 49 * (Num patches)
        print(patches_flat.shape)
        return patches_flat

    def extract_random_patches(self):
        '''
        Extract 'num_patches' many random patches of given size
        color -- 3 patch_size * patch_size (color pixels are flattened)
        b/w -- patch_size * patch_size
        '''
        x = self.data.shape
        k = self.patch_size

        if self.is_color:
            X = np.zeros(shape=(3*(k ** 2), 1))
            for i in np.arange(self.num_patches):
                a = np.random.choice(x[0] - k)  # y coordinate of the top left corner of the random patch
                b = np.random.choice(x[1] - k)  # x coordinate of the top left corner of the random patch
                Y = self.data[a:a+k, b:b+k, :]
                Y = Y.reshape((-1, 1))  # size 3k**2 by 1
                # print('Y.shape', Y.shape)
                if i == 0:
                    X = Y
                else:
                    X = np.append(X, Y, axis=1)  # x is class ndarray
        else:  # b/w image
            X = np.zeros(shape=(k ** 2, 1, 1))
            for i in np.arange(self.num_patches):
                a = np.random.choice(x[0] - k)  # x coordinate of the top left corner of the random patch
                b = np.random.choice(x[1] - k)  # y coordinate of the top left corner of the random patch
                Y = self.data[a:a + k, b:b + k]
                Y = Y.reshape(k ** 2, 1)  # size k**2 by 1
                # print('Y.shape', Y.shape)
                if i == 0:
                    X = Y
                else:
                    X = np.append(X, Y, axis=1)  # X is class ndarray
        return X  # X.shape = (3k**2, num_patches) for color, (k**2, 1) for b/w

    def stack_to_patches(self, path, patch_size=7, iterations=500, batch_size=20, downscale_factor=2):
        # converts a stack of data matrices into 3d array of patch matrices
        X = np.load(path)
        data = (X + 1) / 2
        m = X.shape[0]  # number of matrices stacked up
        stack_patches_flat = []
        for i in np.arange(0, m):
            img = data[i, :, :]
            # downscale image for speed
            if downscale_factor > 1:
                img = downscale_local_mean(img, (downscale_factor, downscale_factor))

            # show_array(data)
            patches = extract_patches_2d(img, (patch_size, patch_size), max_patches=iterations * batch_size)
            # we do not need more than iterations*batch_size patches for training dictionaries
            patches_flat = patches.reshape(len(patches), -1).T  # -1 means unspecified
            stack_patches_flat.append(patches_flat)

        #  b = np.asarray(stack_patches_flat)
        #  np.moveaxis(b, 0, -1)
        #  stack_patches_flat = np.stack((stack_patches_flat, patches_flat), axis=0)
        return stack_patches_flat

    def save_patches(self, filename):
        '''
        Save patches array to filename. in future set patches_file in constructor
        '''
        np.save(filename, self.patches)

    def display_dictionary(self, W):
        k = self.patch_size
        rows = np.floor(np.sqrt(self.n_components)).astype(int)
        cols = np.ceil(np.sqrt(self.n_components)).astype(int)
        fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(6, 6),
                                subplot_kw={'xticks': [], 'yticks': []})
        for ax, i in zip(axs.flat, range(100)):
            if not self.is_color:
                ax.imshow(W.T[i].reshape(k, k), cmap="gray", interpolation='nearest')
            else:
                patch = W.T[i].reshape(k, k, 3)
                ax.imshow(patch/np.max(patch))

        plt.tight_layout()
        plt.suptitle('Dictionary learned from patches of size %d' % k, fontsize=16)
        plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
        plt.show()

    def get_downscaled_dims(self, path, downscale_factor=None, is_matrix=False):
        # need to put is_matrix arg at the end to avoid error (don't know why)
        if downscale_factor is None:
            downscale_factor = self.downscale_factor

        if not is_matrix:
            img = Image.open(path)
            img = np.asarray(img.convert('L'))
        else:
            img = np.load(path)

        data = downscale_local_mean(img, (downscale_factor, downscale_factor))
        return data.shape[0], data.shape[1]

    def train_dict(self, is_stack=False):
        print('training dictionaries from patches...')
        '''
        Trains dictionary based on patches.
        '''
        W = self.W
        At = []
        Bt = []
        code = self.code
        for t in np.arange(self.iterations):
            X = self.extract_random_patches()
            if t == 0:
                self.rnmf = Online_RNMF(X,
                                        n_components=self.n_components,
                                        iterations=self.sub_iterations,
                                        iter_sparsecoding_approx=self.iter_sparsecoding_approx,
                                        batch_size=self.batch_size,
                                        ini_dict=None,
                                        ini_A=None,
                                        ini_B=None,
                                        history=0,
                                        alpha=None,
                                        beta=self.beta)
                W, At, Bt, H = self.rnmf.train_dict()
            else:
                self.rnmf = Online_RNMF(X,
                                        n_components=self.n_components,
                                        iterations=self.sub_iterations,
                                        iter_sparsecoding_approx=self.iter_sparsecoding_approx,
                                        batch_size=self.batch_size,
                                        ini_dict=W,
                                        ini_A=At,
                                        ini_B=Bt,
                                        history=self.rnmf.history,
                                        alpha=None,
                                        beta=self.beta)
                # out of "sample_size" columns in the data matrix, sample "batch_size" randomly and train the dictionary
                # for "iterations" iterations
                W, At, Bt, H = self.rnmf.train_dict()
                # code += H
            print('Current iteration %i out of %i' % (t, self.iterations))
        self.W = W
        print('dict_shape:', self.W.shape)
        print('code_shape:', self.code.shape)
        np.save('Image_dictionary/dict_learned_escher_noise' + str(self.beta), self.W)
        np.save('Image_dictionary/code_learned_escher_noise' + str(self.beta), self.code)

        '''
        code = self.code
        if not is_stack:
            self.nmf = Online_NMF(self.patches, self.n_components, self.iterations, self.batch_size)
            self.W, self.A, self.B, H = self.nmf.train_dict()
        else:
            self.nmf = Online_NMF_stack(self.patches, self.n_components, self.iterations, self.batch_size)
            self.W, self.A, self.B, self.code = self.nmf.train_dict()
        print('dict_shape:', self.W.shape)
        print('code_shape:', self.code.shape)
        np.save('Image_dictionary/dict_learned_21', self.W)
        np.save('Image_dictionary/code_learned_21', self.code)
        '''

    def show_array(self, arr):
        plt.figure()
        plt.imshow(arr, cmap="gray")
        plt.show()

    def reconstruct_image_old(self, path, downscale_factor=None, patch_size=10, is_matrix=False):
        print('reconstructing given image...')
        if downscale_factor is None:
            downscale_factor = self.downscale_factor

        t0 = time()
        dims = self.get_downscaled_dims(path, downscale_factor, is_matrix=is_matrix)
        patches = self.image_to_patches(path, patch_size=patch_size, downscale_factor=downscale_factor,
                                        is_matrix=is_matrix, is_recons=True)
        self.rnmf = Online_RNMF(patches, self.n_components, self.iterations, self.batch_size)
        code = self.rnmf.sparse_code(patches, self.W.T)
        print('Reconstructed in %.2f seconds' % (time() - t0))
        patches_recons = np.dot(self.W, code).T
        patches_recons = patches_recons.reshape(patches_recons.shape[0], patch_size, patch_size)
        img_recons = reconstruct_from_patches_2d(patches_recons, (dims[0], dims[1]))
        self.show_array(img_recons)
        return code

    def reconstruct_image_denoised(self, path, recons_resolution=1, beta=3):
        print('reconstructing given network...')
        '''
        Note: For WAN data, the algorithm reconstructs the normalized WAN matrix A/np.max(A). 
        Scale the reconstructed matrix B by np.max(A) and compare with the original network. 
        '''
        A = self.read_img_as_array(path)  # A.shape = (row, col, 3)
        A_matrix = A.reshape(-1, A.shape[1])  # (row, col, 3) --> (3row, col)
        [m, n] = A_matrix.shape
        A_recons = np.zeros(shape=A.shape)
        A_noise = np.zeros(shape=A.shape)
        A_overlap_count = np.zeros(shape=(A.shape[0], A.shape[1]))
        k = self.patch_size
        t0 = time()
        c = 0
        num_rows = np.floor((A_recons.shape[0]-k)/recons_resolution).astype(int)
        num_cols = np.floor((A_recons.shape[1]-k)/recons_resolution).astype(int)

        for i in np.arange(0, A_recons.shape[0]-k, recons_resolution):
            for j in np.arange(0, A_recons.shape[1]-k, recons_resolution):
                patch = A[i:i + k, j:j + k]
                patch = patch.reshape((-1, 1))
                # print('patch.shape', patch.shape)
                rnmf = Online_RNMF(patch, beta=beta)
                code, noise = rnmf.sparse_code(patch, self.W)
                print('code', np.sum(np.sum(code)))
                print('noise', np.sum(np.sum(noise)))
                # alpha = L1 regularization parameter. alpha=2 makes all codes zero (why?)
                patch_recons = np.dot(self.W, code).T
                patch_recons = patch_recons.reshape(k, k)
                noise_recons = noise.reshape(k, k)

                # now paint the reconstruction canvas
                for x in itertools.product(np.arange(k), repeat=2):
                    c = A_overlap_count[i+x[0], j+x[1]]
                    A_recons[i+x[0], j+x[1]] = (c*A_recons[i+x[0], j+x[1]] + patch_recons[x[0], x[1]])/(c+1)
                    A_noise[i + x[0], j + x[1]] = (c * A_noise[i + x[0], j + x[1]] + noise_recons[x[0], x[1]]) / (
                                c + 1)
                    A_overlap_count[i+x[0], j+x[1]] += 1

                # progress status
                print('reconstructing (%i, %i)th patch out of (%i, %i)' % (i/recons_resolution, j/recons_resolution, num_rows, num_cols))
        print('Reconstructed in %.2f seconds' % (time() - t0))
        print('A_recons.shape', A_recons.shape)
        print('A_noise.shape', A_noise.shape)
        np.save('Image_dictionary/img_recons_denoised', A_recons)
        np.save('Image_dictionary/img_recons_noise', A_noise)
        plt.matshow(A_recons)
        return A_recons

    def reconstruct_image_color(self, path, recons_resolution=1):
        print('reconstructing given network...')
        '''
        Note: For WAN data, the algorithm reconstructs the normalized WAN matrix A/np.max(A). 
        Scale the reconstructed matrix B by np.max(A) and compare with the original network. 
        '''
        A = self.read_img_as_array(path)  # A.shape = (row, col, 3)
        A_matrix = A.reshape(-1, A.shape[1])  # (row, col, 3) --> (3row, col)
        [m, n] = A_matrix.shape
        A_recons = np.zeros(shape=A.shape)
        A_overlap_count = np.zeros(shape=(A.shape[0], A.shape[1]))
        k = self.patch_size
        t0 = time()
        c = 0
        num_rows = np.floor((A_recons.shape[0]-k)/recons_resolution).astype(int)
        num_cols = np.floor((A_recons.shape[1]-k)/recons_resolution).astype(int)

        for i in np.arange(0, A_recons.shape[0]-k, recons_resolution):
            for j in np.arange(0, A_recons.shape[1]-k, recons_resolution):
                patch = A[i:i + k, j:j + k, :]
                patch = patch.reshape((-1, 1))
                # print('patch.shape', patch.shape)
                coder = SparseCoder(dictionary=self.W.T, transform_n_nonzero_coefs=None,
                                    transform_alpha=1, transform_algorithm='lasso_lars', positive_code=True)
                # alpha = L1 regularization parameter. alpha=2 makes all codes zero (why?)
                code = coder.transform(patch.T)
                patch_recons = np.dot(self.W, code.T).T
                patch_recons = patch_recons.reshape(k, k, 3)

                # now paint the reconstruction canvas
                for x in itertools.product(np.arange(k), repeat=2):
                    c = A_overlap_count[i+x[0], j+x[1]]
                    A_recons[i+x[0], j+x[1], :] = (c*A_recons[i+x[0], j+x[1], :] + patch_recons[x[0], x[1], :])/(c+1)
                    A_overlap_count[i+x[0], j+x[1]] += 1

                # progress status
                print('reconstructing (%i, %i)th patch out of (%i, %i)' % (i/recons_resolution, j/recons_resolution, num_rows, num_cols))
        print('Reconstructed in %.2f seconds' % (time() - t0))
        print('A_recons.shape', A_recons.shape)
        np.save('Image_dictionary/img_recons_denoised', A_recons)
        plt.imshow(A_recons)
        return A_recons

def main():

    patch_size = 10
    sources = ["Data/escher/" + str(n) + ".jpg" for n in np.arange(15,16)]
    for path in sources:
        reconstructor = Image_Reconstructor_ornmf(path=path,
                                                  n_components=100,
                                                  iterations=200,
                                                  iter_sparsecoding_approx=4,
                                                  sub_iterations=3,
                                                  patch_size=patch_size,
                                                  batch_size=100,
                                                  num_patches=100,
                                                  downscale_factor=3,
                                                  is_matrix=False,
                                                  is_color=False,
                                                  beta=0.1,
                                                  noise_rate=0)
        # reconstructor.save_patches("escher_patches.npy")
        reconstructor.display_img()
        reconstructor.train_dict()
        # reconstructor.W = np.load('Image_dictionary/dict_learned_escher_noise4.npy')
        dict = reconstructor.W  # trained dictionary
        # print(dict.shape)
        reconstructor.display_dictionary(dict)
        reconstructor.reconstruct_image_denoised(path="Data/escher/15.jpg",
                                                 recons_resolution=4,
                                                 beta=0.1)
        # reconstructor.reconstruct_image_denoised(path=path, recons_resolution=4)

if __name__ == '__main__':
    main()


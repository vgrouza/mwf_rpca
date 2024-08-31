"""
Implementation of Song et al (2020) Blind Source Separation with robust Principal Component Analysis for 
Myelin Water Imaging from T2* Multi-Echo Gradient-Recalled Echo Data. 

Written by Vladimir Grouza 2022
Quantitative Microstructure Imaging Lab
McConnell Brain Imaging Centre, Montreal Neurological Institute and Hospital
Department of Neurology and Neurosurgery, McGill University, Montreal, QC
www.rudkolab.com
"""


import numpy as np
import time
import nimfa
import scipy.linalg.interpolative as sli


def _compute_temporal_modulation(te_vec_in, temp_mod_fact_shape):
    te_3d = np.ones(shape=temp_mod_fact_shape, order='F') * te_vec_in[np.newaxis, np.newaxis, :]
    temp_modulation_factor = (np.exp(1e3 * te_3d))
    return temp_modulation_factor


def _hankelize(input_matrix, hankel_length):
    # input: NVOX x NE >> output: NH x hankel_length

    # compute sizes
    N1, N2 = input_matrix.shape
    # Number of rows in the Hankelized Matrix
    NH = N1 * (N2 - hankel_length + 1)

    # preallocate memory
    hankelized_matrix = np.zeros(shape=[NH, hankel_length], order='F')

    # perform hankelization
    for index in range(0, N2 - hankel_length + 1):
        hankelized_matrix[(index * N1): (index + 1) * N1, :] = input_matrix[:, index:(index + hankel_length)]
    return hankelized_matrix


def _dehankelize(input_matrix, dehankel_length):
    # input: NH x NHANK >> output: NVOX x NE

    # compute sizes
    N1, N2 = input_matrix.shape
    NP = int(N1 / dehankel_length)
    NL = dehankel_length - 1 + N2

    # preallocate memory
    dehankelized_matrix = np.zeros(shape=[NP, NL], order='F')
    counts = np.zeros(shape=[NP, NL], order='F')

    # perform dehankelization
    for index in range(0, dehankel_length):
        dehankelized_matrix[:, index:(index + N2)] = dehankelized_matrix[:, index: (index + N2)] \
                                                     + input_matrix[index * NP:(index + 1) * NP, :]
        counts[:, index:(index + N2)] = counts[:, index:(index + N2)] + 1
    dehankelized_matrix = np.true_divide(dehankelized_matrix, counts)
    return dehankelized_matrix


def _vectorize(input_matrix, new_shape):
    # compartmentalize the reshape function
    output_matrix = np.reshape(input_matrix, newshape=new_shape, order='F')
    return output_matrix


def _devectorize(input_matrix, new_shape):
    output_matrix = _vectorize(input_matrix, new_shape)
    return output_matrix


def _llr_patches(nx, ny, nz, window):
    xpos_list = list()
    ypos_list = list()
    zpos_list = list()

    for iz in range(0, int(np.floor(nz / window[2]))):
        zpos = int(window[2] * iz)
        for iy in range(0, int(np.floor(ny / window[1]))):
            ypos = int(window[1] * iy)
            for ix in range(0, int(np.floor(nx / window[0]))):
                xpos = int(window[0] * ix)
                # populate the list
                zpos_list.append(zpos)
                ypos_list.append(ypos)
                xpos_list.append(xpos)

    return xpos_list, ypos_list, zpos_list


def _extract_patch(input_tensor, xpos, ypos, zpos, window):
    # takes in a 4D tensor and returns just what's in the patch
    return input_tensor[xpos:xpos + window[0], ypos:ypos + window[1], zpos:zpos + window[2]]


def _soft_thresh(input_tensor, lambda1):
    """Soft-thresholding of array X."""
    res = abs(input_tensor) - lambda1
    np.maximum(res, 0.0, out=res)
    res *= np.sign(input_tensor)
    return res


def _check_convergence(x_tensor, x_tensor_prev, conv_thresh):
    x_update = np.linalg.norm(x_tensor.flatten() - x_tensor_prev.flatten()) / \
               np.linalg.norm(x_tensor_prev.flatten()) * 100
    print(x_update)
    if x_update <= conv_thresh:
        return True
    else:
        return False


class RobustPCA:

    def __init__(self):
        # set up input mGRE image, mask, and echo_times
        self.echo_times = None
        self.mgre_data = None
        self.binary_mask = None

        # image shape constants
        self.NX, self.NY, self.NZ, self.NECHOES = [0, 0, 0, 0]
        self.NHANK = int(0)
        self.NBASES = int(0)
        self.NVOX = int(0)
        self.NWINDOW = int(0)

        # regularization constants
        self.DELTA1 = 25
        self.DELTA2 = 25
        self.DELTA3 = 50

        # instantiate empty X, U, and Z tensors for rPCA
        self.X = None
        self.U = None
        self.Z = None

    def initialize(self):

        # [modulate] output: (4D) NX x NY x NZ x NE << input: (4D) NX x NY x NZ x NE times NX x NY x NZ x NE
        temp_mod_fact = _compute_temporal_modulation(self.echo_times, self.mgre_data.shape)
        input_data = np.multiply(self.mgre_data, temp_mod_fact)

        # [vectorize] output: (2D) NVOX x NE << input: (4D) NX x NY x NZ x NE
        vectorized_matrix = _vectorize(input_data, [self.NVOX, self.NECHOES])

        # [hankelize] input: output: (2D) NH x NE/2 << (2D) NVOX x NE
        hankelized_matrix = _hankelize(vectorized_matrix, self.NHANK)

        # [nndsvd] output (2D) W: NH x NBASES; H: NBASES x NE/2 << input: (2D) NH x NE/2
        nmf_model = nimfa.Nmf(hankelized_matrix, rank=self.NBASES, seed='nndsvd', update='euclidean', max_iter=15)
        nmf_fit = nmf_model()

        # [outer product]: output: 2D NH x NE/2 << input: (2D) W[:,0] (NH x 1) outer (2D) H[0,:] (1 x NE/2)
        outer_product = np.outer(nmf_fit.fit.W[:, 0], nmf_fit.fit.H[0, :])

        # [dehankelize] output: (2D) NVOX x NE << input: (2D) NH x NE/2
        dehankelized_matrix = _dehankelize(outer_product, self.NECHOES - self.NHANK + 1)

        # [devectorize] output: (4D) NX x NY x NZ x NE << input: (2D) NVOX x NE
        devectorized_matrix = _devectorize(dehankelized_matrix, self.mgre_data.shape)

        # [demodulate] output: (4D) NX x NY x NZ x NE << input: (4D) NX x NY x NZ x NE divide NX x NY x NZ x NE
        L1 = np.true_divide(devectorized_matrix, temp_mod_fact)
        L2 = self.mgre_data - L1

        # Set up Tensors with correct sizes [NX, NY, NZ, NE, 3] and populate with initial estimates of L1 and L2
        self.X = np.zeros(shape=[self.NX, self.NY, self.NZ, self.NECHOES, 3], order='F')
        self.U = np.copy(self.X)
        self.Z = np.copy(self.X)

        self.X[:, :, :, :, 0] = L1
        self.X[:, :, :, :, 1] = L2

    def fit(self, mgre_data, binary_mask, echo_times):

        # image, mask, and echo times
        self.mgre_data = mgre_data
        self.binary_mask = binary_mask
        self.echo_times = echo_times

        # image shape constants
        self.NX, self.NY, self.NZ, self.NECHOES = mgre_data.shape
        self.NHANK = int(self.NECHOES / 2)
        self.NBASES = 3
        self.NVOX = int(self.NX * self.NY * self.NZ)
        self.NWINDOW = 8

        # do the initial NMF with NNDSVD
        self.initialize()

        # begin iterating X, U, and Z
        k = 0
        converged = False
        while not converged:
            print('Iteration Number', k)
            # specificy window size depending on iteration number
            if k < 1:
                window = [self.NX, self.NY, self.NZ]
            else:
                # option to use randomly sized patches implemented as well
                # window = [self.NWINDOW, self.NWINDOW, self.NWINDOW]
                window = [np.random.randint(low=2, high=self.NWINDOW), np.random.randint(low=2, high=self.NWINDOW), 2]

            xpos, ypos, zpos = _llr_patches(self.NX, self.NY, self.NZ, window)

            print('Updating U1 and U2')
            start_time = time.time()
            x_tensor = np.copy(self.X)
            u_tensor = np.copy(self.U)
            z_tensor = np.copy(self.Z)

            ##TODO: would be really nice to use multiprocessing to parallelize these loops, but it looks tricky
            # update U1 (Equation 15a)
            for rp in range(0, len(xpos)):
                # extract local patch for X and Z
                curr_patch = _extract_patch(x_tensor[:, :, :, :, 0], xpos[rp], ypos[rp], zpos[rp], window) \
                             + self.DELTA1 ** (-1) \
                             * _extract_patch(z_tensor[:, :, :, :, 0], xpos[rp], ypos[rp], zpos[rp], window)

                # [vectorize] output: (2D) NVOX x NE << input: (4D) NX x NY x NZ x NE
                curr_patch_vec = _vectorize(curr_patch, [np.prod(window), self.NECHOES])

                # [hankelize] input: output: (2D) NH x NE/2 << (2D) NVOX x NE
                curr_patch_hank = _hankelize(curr_patch_vec, self.NHANK)

                # [SVD with hard thresholding]
                U, s, V = sli.svd(curr_patch_hank, 1e-12)
                curr_patch_svt = np.inner(U[:, 0:1], np.inner(np.diag(s[0:1]), V[:, 0:1]).T)

                # [dehankelize] output: (2D) NVOX x NE << input: (2D) NH x NE/2
                curr_patch_dhank = _dehankelize(curr_patch_svt, self.NECHOES - self.NHANK + 1)

                # [devectorize] output: (4D) NX x NY x NZ x NE << input: (2D) NVOX x NE
                curr_patch_dvec = _devectorize(curr_patch_dhank, [window[0], window[1], window[2], self.NECHOES])

                # populate the correct patch
                u_tensor[xpos[rp]:xpos[rp] + window[0],
                         ypos[rp]:ypos[rp] + window[1],
                         zpos[rp]:zpos[rp] + window[2], :, 0] = curr_patch_dvec

            # update U2 (Equation 15b)
            for rp in range(0, len(xpos)):
                # extract local patch for X and Z
                curr_patch = _extract_patch(x_tensor[:, :, :, :, 1], xpos[rp], ypos[rp], zpos[rp], window) \
                             + self.DELTA2 ** (-1) \
                             * _extract_patch(z_tensor[:, :, :, :, 1], xpos[rp], ypos[rp], zpos[rp], window)

                # [vectorize] output: (2D) NVOX x NE << input: (4D) NX x NY x NZ x NE
                curr_patch_vec = _vectorize(curr_patch, [np.prod(window), self.NECHOES])

                # [hankelize] input: output: (2D) NH x NE/2 << (2D) NVOX x NE
                curr_patch_hank = _hankelize(curr_patch_vec, self.NHANK)

                # [SVD with hard thresholding]
                U, s, V = sli.svd(curr_patch_hank, 1e-12)
                curr_patch_svt = np.inner(U[:, 0:2], np.inner(np.diag(s[0:2]), V[:, 0:2]).T)

                # [dehankelize] output: (2D) NVOX x NE << input: (2D) NH x NE/2
                curr_patch_dhank = _dehankelize(curr_patch_svt, self.NECHOES - self.NHANK + 1)

                # [devectorize] output: (4D) NX x NY x NZ x NE << input: (2D) NVOX x NE
                curr_patch_dvec = _devectorize(curr_patch_dhank, [window[0], window[1], window[2], self.NECHOES])

                # populate the correct patch
                u_tensor[xpos[rp]:xpos[rp] + window[0],
                         ypos[rp]:ypos[rp] + window[1],
                         zpos[rp]:zpos[rp] + window[2], :, 1] = curr_patch_dvec

            print("--- %s seconds ---" % np.round((time.time() - start_time), decimals=3))

            # update U3 (Equation 15c)
            print('Updating U3')
            start_time = time.time()
            soft_thresholding = 0.001
            u_tensor[:, :, :, :, 2] = _soft_thresh(
                (x_tensor[:, :, :, :, 2] + z_tensor[:, :, :, :, 2] * self.DELTA3 ** (-1)), soft_thresholding)
            print("--- %s seconds ---" % np.round((time.time() - start_time), decimals=3))

            # update L1 (Equation 16a)
            print('Updating X and Z')
            start_time = time.time()
            x_tensor_prev = np.copy(x_tensor) # deep copy x_tensor at previous iteration
            x_tensor[:, :, :, :, 0] = (1 + self.DELTA1) ** (-1) * (self.mgre_data -
                                                                   x_tensor_prev[:, :, :, :, 1] -
                                                                   x_tensor_prev[:, :, :, :, 2] +
                                                                   u_tensor[:, :, :, :, 0] * self.DELTA1 -
                                                                   z_tensor[:, :, :, :, 0])

            # update L2 (Equation 16b)
            x_tensor[:, :, :, :, 1] = (1 + self.DELTA2) ** (-1) * (self.mgre_data -
                                                                   x_tensor[:, :, :, :, 0] -
                                                                   x_tensor_prev[:, :, :, :, 2] +
                                                                   u_tensor[:, :, :, :, 1] * self.DELTA2 -
                                                                   z_tensor[:, :, :, :, 1])

            # update S (Equation 16c)
            x_tensor[:, :, :, :, 2] = (1 + self.DELTA3) ** (-1) * (self.mgre_data -
                                                                   x_tensor[:, :, :, :, 0] -
                                                                   x_tensor[:, :, :, :, 1] +
                                                                   u_tensor[:, :, :, :, 2] * self.DELTA3 -
                                                                   z_tensor[:, :, :, :, 2])

            # update Langangian multipliers (Equation 17)
            z_tensor += self.DELTA1 * (x_tensor - u_tensor)
            print("--- %s seconds ---" % np.round((time.time() - start_time), decimals=3))

            self.X = x_tensor
            self.U = u_tensor
            self.Z = z_tensor

            # update iteration number
            k += 1

            # update convergence status
            convergence_threshold = 0.2
            converged = _check_convergence(self.X, x_tensor_prev, convergence_threshold)

        # once convergence has been achieved, clean up X and exit
        self.X[self.X < 0] = 0.0
        self.X[np.isnan(self.X)] = 0.0
        self.X[np.isinf(self.X)] = 0.0
        self.X = np.real(self.X)
        print('Convergence Achieved. rPCA Fit Complete.')

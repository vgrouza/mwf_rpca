# Generic Libraries
import os
import numpy as np
import time

# Scientific Libraries
from scipy import fft
from skimage import metrics

def recon_k2image(k_space_in):
    # TODO: add comments
    image_space_out = fft.ifftshift(fft.ifftn(fft.ifftshift(k_space_in, axes=[0, 1, 2]), axes=[0, 1, 2]), axes=[0, 1, 2])
    return image_space_out


def recon_image2k(image_space_in):
    # TODO: add comments
    k_space_out = fft.fftshift(fft.fftn(fft.fftshift(image_space_in, axes=[0, 1, 2]), axes=[0, 1, 2]), axes=[0, 1, 2])
    return k_space_out


def apply_phase_correction(image_space_in):
    """
    This function implements the correction of k-space echo misalignment as described by Lu et al; MRM 2008
    The basic idea is to determine an optimal slope alpha that can be used in a phase ramp applied to image space
    in order to mitigate misalignment of magnitude in k-space. This correction permits downstream corrections
    for delta-B0 inhomogeneity.
    :param image_space_in: a complex image-domain 4D numpy array [NX, NY, NZ, necho], with  NX
    as the readout direction.
    :return: image_space_in: same as in the input, but with even echoes corrected with a linear phase ramp.
    """
    start_time = time.time()
    print('Correcting odd/even echo k-space misalignment...')

    # Use kx-NY-NZ hybrid domain for odd echos (using only 1st and 3rd echo is sufficient)
    img_spc_odd = image_space_in[:, :, :, [0, 2]]
    img_spc_odd = fft.fftshift(fft.fftn(fft.fftshift(img_spc_odd, axes=0), axes=[0]), axes=0)

    # average projection across NY and NZ dimensions
    img_spc_odd = np.mean(np.mean(img_spc_odd, 1), 1)

    # specify range of slopes (alpha) to test and array to store correlation coefficient (rho)
    # according to "Correction of k -Space Echo Misalignment" in Lu, 2008. MRM 60: 198-209
    alpha = np.arange(-0.2, 0.2, 0.005)
    rho_list = np.zeros(len(alpha))

    # set up phase factor basis
    phase_factor = np.repeat(np.arange(-image_space_in.shape[0], 0, 1)[:, np.newaxis], image_space_in.shape[1], axis=1)
    phase_factor = np.repeat(phase_factor[:, :, np.newaxis], image_space_in.shape[2], axis=2)

    for i in range(len(alpha)):
        # multiply 2nd echo by phase factor, introducing shift in k-space
        img_spc_even = image_space_in[:, :, :, 1] * np.exp(1j * alpha[i] * phase_factor)

        # Use kx-NY-NZ hybrid domain
        img_spc_even = fft.fftshift(fft.fftn(fft.fftshift(img_spc_even, axes=0), axes=[0]), axes=0)

        # average projection across NY and NZ dimensions as for odd echoes
        img_spc_even = np.mean(np.mean(img_spc_even, 1), 1)

        # compute cross correlation of (1st and 2nd) echoes * (3rd and 2nd) echoes
        # according to Eq. 3 in Lu, 2008. MRM 60: 198-209
        rho_list[i] = np.dot(abs(np.dot(img_spc_odd[:, 0].conj().T, img_spc_even)) /
                             np.dot(abs(img_spc_odd[:, 0]), abs(img_spc_even)),
                             abs(np.dot(img_spc_odd[:, 1].conj().T, img_spc_even)) /
                             np.dot(abs(img_spc_odd[:, 1]), abs(img_spc_even)))

    # expand the phase factor basis from 3D into 4D to apply along echo dimension
    phase_factor = np.repeat(phase_factor[:, :, :, np.newaxis], int(image_space_in.shape[3] / 2), axis=3)

    # apply phase correction factor to even echoes only, using the optimal slope alpha
    image_space_in[:, :, :, 1::2] = image_space_in[:, :, :, 1::2] * \
                                    np.exp(1j * alpha[np.argmax(rho_list)] * phase_factor)

    print("--- %s seconds ---" % np.round((time.time() - start_time), decimals=3))
    print('Optimal slope derived from CC was alpha = ' + str(alpha[np.argmax(alpha)]))
    # return the phase-corrected image space
    return image_space_in


def apply_misregistration_correction(image_space_in):
    """ This function implements even/odd echo mistregistration in image space in a method similar to
    Lu et al; MRM 2008. A phase ramp is applied in k-space along the readout dimension and
    mutual information between the first and second echos are computed in image space. Once an optimal
    phase ramp is computed, it is applied to all even echos, solving even/odd echo misregistration of the
    magnitude images.
    :param image_space_in: a complex image-domain 4D numpy array [NX, NY, NZ, necho], with  NX
    as the readout direction.
    :return: same as in the input, but with even echoes corrected with a linear phase ramp.
    """

    start_time = time.time()
    print('Correcting odd/even echo image-space misregistration...')

    def misreg_corr(k_space_in, beta):
        """
        Correct the misregistration in image space by applying a phase slope along the readout direction in k space

        Args:
        slope: the phase slope along readout direction
        k: 4D k-space data (last dimension is the number of echoes)

        Return:
        k_corr: the corrected k-space data
        """
        # Correction factor along readout direction (x)
        x_range = np.arange(1, k_space_in.shape[0] + 1)
        phase_corr_factor = np.exp(1j * x_range * beta)

        # extend the correction factor to all other dimensions
        phase_corr_matrix = np.repeat(phase_corr_factor[:, np.newaxis], k_space_in.shape[1], axis=1)
        phase_corr_matrix = np.repeat(phase_corr_matrix[:, :, np.newaxis], k_space_in.shape[2], axis=2)
        phase_corr_matrix = np.repeat(phase_corr_matrix[:, :, :, np.newaxis], k_space_in.shape[3] / 2, axis=3)

        # apply correction to even echoes
        k_space_corr_out = np.copy(k_space_in)
        k_space_corr_out[:, :, :, 1::2] = k_space_corr_out[:, :, :, 1::2] * phase_corr_matrix
        return k_space_corr_out

    def image_reg_metrics(k_space_in):
        """
        Use mutual information (MI) to measure the goodness of the registration between even and odd echoes in image space.
        """
        image_space_in = recon_k2image(k_space_in)
        mi = metrics.normalized_mutual_information(abs(image_space_in[:, :, :, 0]), abs(image_space_in[:, :, :, 1]))
        return mi

    # initialize the lists of slopes and MIs
    beta_list = np.arange(-0.03, 0.03, 0.0005)
    mi_list = np.zeros(beta_list.shape[0])

    # iterate through each slope and calculate its corresponding MI
    for index, beta in enumerate(beta_list):
        k_space_corr = misreg_corr(recon_image2k(image_space_in[:, :, :, (0, 1)]), beta)
        mi_list[index] = image_reg_metrics(k_space_corr)

    np.savetxt("mi_list.csv", mi_list)
    # obtain the optimal slope (maximal MI) and make the correction
    k_space_corr = misreg_corr(recon_image2k(image_space_in), beta_list[np.argmax(mi_list)])
    image_space_corr = recon_k2image(k_space_corr)

    print("--- %s seconds ---" % np.round((time.time() - start_time), decimals=3))
    print('Optimal slope derived from MI was beta = ' + str(beta_list[np.argmax(mi_list)]))
    print('Highest MI attained was: ' + str(np.max(mi_list)))
    return image_space_corr
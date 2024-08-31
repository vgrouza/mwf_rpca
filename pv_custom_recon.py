import os
import numpy as np
import itertools
import warnings
import scipy.signal as signal
import brkraw as bru
import nibabel as nib


def recon_mgre(fid_path: str, matrix_dims: tuple):
    """
    A custom reconstruction script for all mGRE acquistions.

    Parameters
    ----------
    fid_path : str
        A string or path-like object which specifies the path to the scan id folder.
    matrix_dims: tuple
        A 4-tuple of integers describing the acquired matrix dimensions as (Kx, Ky, Kz, NECHO). Can be accessed through
        brkraw as pv_data_set.get_matrix_size(<MGE Scan ID>, 1)

    Returns
    -------
    image_space : np.ndarray
        The reconstructed complex image space volume as (Nx, Ny, Nz, NECHO).
    """
    # check that file exists
    if not os.path.exists(os.path.join(fid_path, 'fid')):
        raise FileNotFoundError(f"No fid file found at the specified path: {fid_path}")
    else:
        # load the raw free induction decay (FID) data
        with open(os.path.join(str(fid_path), 'fid'), 'rb') as fid:
            fid_data = np.fromfile(fid, np.int32)
        fid.close()
        fid_data[np.isinf(fid_data) | np.isnan(fid_data)] = 0.0

    # reshape, truncate, and permute to match PVM_EncMatrix
    KX, KY, KZ, NECHO = matrix_dims
    k_space = fid_data[0::2].astype(float) + 1j * fid_data[1::2].astype(float)
    KX_RAW = int(len(k_space) / (KY * KZ * NECHO))
    k_space = np.reshape(k_space, (KX_RAW, NECHO, KY, KZ), order='F')
    k_space = np.transpose(k_space, (0, 2, 3, 1))[0:KX, :, :, :]

    # flip even echoes along readout direction (KX)
    k_space[:, :, :, 1::2] = np.flip(k_space[:, :, :, 1::2], 0)

    # reconstruct image space with IFFT
    return np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(k_space, axes=[0, 1, 2]), axes=[0, 1, 2]), axes=[0, 1, 2])


def recon_mese(fid_path: str, matrix_dims: tuple, zero_pad: bool = False, apodization_factor: float = 0.0):
    """
    A custom reconstruction script for all MESE acquistions.

    Parameters
    ----------
    fid_path : str
        A string or path-like object which specifies the path to the scan id folder.
    matrix_dims: tuple
        A 4-tuple of integers describing the acquired matrix dimensions as (Kx, Ky, Kz, NECHO). Can be accessed through
        brkraw as pv_data_set.get_matrix_size(<MGE Scan ID>, 1)
    zero_pad : bool
        Toggle zero-padding to reconstruct at native resolution (False) or zero-pad to match mGRE acquisition (True).
    apodization_factor : float
        Apodization factor for the Tukey window applied to each k-space dimension (Kx, Ky, Kz).

    Returns
    -------
    image_space : np.ndarray
        The reconstructed complex image space volume as (Nx, Ny, Nz, NECHO).
    """
    # check that file exists
    if not os.path.exists(os.path.join(fid_path, 'fid')):
        raise FileNotFoundError(f"No fid file found at the specified path: {fid_path}")
    else:
        # load the raw free induction decay (FID) data
        with open(os.path.join(str(fid_path), 'fid'), 'rb') as fid:
            fid_data = np.fromfile(fid, np.int32)
        fid.close()
        fid_data[np.isinf(fid_data) | np.isnan(fid_data)] = 0.0

    # reshape, truncate, and permute to match PVM_EncMatrix
    KX, KY, KZ, NECHO = matrix_dims
    k_space = fid_data[0::2].astype(float) + 1j * fid_data[1::2].astype(float)
    KX_RAW = int(len(k_space) / (KY * KZ * NECHO))
    k_space = np.reshape(k_space, (KX_RAW, NECHO, KY, KZ), order='F')
    k_space = np.transpose(k_space, (0, 2, 3, 1))[0:KX, :, :, :]

    # apply Tukey window apodization: generate a 1D window and broadcast it to 4D
    wind_fun_x = (signal.windows.tukey(k_space.shape[0], apodization_factor)).reshape((-1, 1, 1, 1), order='F')
    wind_fun_y = (signal.windows.tukey(k_space.shape[1], apodization_factor)).reshape((1, -1, 1, 1), order='F')
    wind_fun_z = (signal.windows.tukey(k_space.shape[2], apodization_factor)).reshape((1, 1, -1, 1), order='F')
    k_space *= wind_fun_x * wind_fun_y * wind_fun_z

    # zero pad if to match mGRE aquisition resolution, if specified.
    if zero_pad:
        k_space = np.pad(k_space,
                         (((int(np.floor(KX / 4)), int(np.ceil(KX / 4))),
                           (int(np.floor(KY / 4)), int(np.ceil(KY / 4))),
                           (int(np.floor(KZ / 4)), int(np.ceil(KZ / 4))),
                           (0, 0))),
                         mode='constant',
                         constant_values=0)

    # reconstruct image space with IFFT
    return np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(k_space, axes=[0, 1, 2]), axes=[0, 1, 2]), axes=[0, 1, 2])


def recon_dwgrase(fid_path: str, nvol: int, zero_pad: bool = False, apodization_factor: float = 0.0):
    """
    A custom reconstruction script for all dwGRASE acquistions. Since all acquisition parameters are known a-priori, the
    script features several hard-coded constants to aid in data reading and reconstruction. Translated from MATLAB by
    Vladimir Grouza (2024) based on the script by Dan Wu (2011), converted from bruker_3dgrase_reader5z.pro (DLI script)
    by Jiangyang Zhang. For more info about the dwGRASE sequence, please consult the following literature:

    [1] Aggarwal, M., Mori, S., Shimogori, T., Blackshaw, S. and Zhang, J. (2010), Three-dimensional diffusion tensor
    microimaging for anatomical characterization of the mouse brain. Magn. Reson. Med., 64: 249-261.
    https://doi.org/10.1002/mrm.22426

    [2] Wu, D., Xu J., McMahon, M.T., van Zijl, P.C.M., Mori, S., Northington, F.J., Zhang, J. (2013), In vivo high-
    resolution diffusion tensor imaging of the mouse brain. NeuroImage, 83: 18-26.
    https://doi.org/10.1016/j.neuroimage.2013.06.012.

    Parameters
    ----------
    fid_path : str
        A string or path-like object which specifies the path to the scan id folder.
    nvol : int
        The number of 3D volumes in this acquisition. Should be identical to number of b-values in the dw acquisition.
    zero_pad : bool
        Toggle zero-padding to reconstruct at native resolution (False) or zero-pad to match mGRE acquisition (True).
    apodization_factor : float
        Apodization factor for the Tukey window applied to each k-space dimension (Kx, Ky, Kz).

    Returns
    -------
    image_space : np.ndarray
        The reconstructed image space volume as (Nx, Ny, Nz, NVOL).
    """
    if not isinstance(zero_pad, bool):
        raise ValueError("Zero padding toggle must be a boolean.")
    if apodization_factor < 0.0 or apodization_factor > 1.0:
        raise ValueError("Apodization factor must be in [0.0, 1.0].")

    # hard coded constants related to the acquisition
    NX, NY, NZ = 90, 64, 16     # 16 is the number of gradient echoes
    RAREFACTOR = 4              # 4 is the number of spin echoes
    NCH = 1                     # number of channels
    NAVFLAG = 1                 # navigator flag
    NVOL = nvol                 # number of volumes

    # hard coded constants for specifying buffer size to read
    NX0 = int(NX * 9)
    NX0_TOT = int(np.ceil(NX0 * NCH / 128) * 128)
    NY0 = int(RAREFACTOR + 2 * NAVFLAG)
    NY1 = int(NY // RAREFACTOR)
    NZ0 = 5
    COUNT = int(RAREFACTOR + 2 * NAVFLAG) * (NY // RAREFACTOR) * NZ * NVOL
    NAVG = 2

    # other variables related to signal processing operations
    APZ_FACTOR = apodization_factor * np.ones(3)
    SMOOTHING_WINDOW = 11
    smoothing_kernel = np.ones(SMOOTHING_WINDOW) / SMOOTHING_WINDOW
    # -----------------------------------------------------------------------------------------------------------------#

    # helper functions organized in a modular manner
    def _read_fid_data(fid_path: str):
        # -- module for reading fid data from binary file
        # check that file exists
        if not os.path.exists(os.path.join(fid_path, 'fid')):
            raise FileNotFoundError(f"No fid file found at the specified path: {fid_path}")
        else:
            # load the raw free induction decay (FID) data
            with open(os.path.join(fid_path, 'fid'), 'rb') as fid:
                fid_data = np.fromfile(fid, dtype=np.int32)
            fid.close()
            fid_data[np.isinf(fid_data) | np.isnan(fid_data)] = 0

        # extract the part of fid data required for recon
        fid_data = (fid_data[0:2 * NX0_TOT * COUNT]).reshape((2, NX0_TOT, NY0, NY1, NZ, NVOL), order='F')
        fid_data = (fid_data[:, 0:NX0 * NCH, :, :, :, :]).reshape((2, NX0, NCH, NY0, NY1, NZ, NVOL), order='F')
        fid_data = np.tile(fid_data, (1, 1, 2, 1, 1, 1, 1))
        return fid_data


    def _get_phe_table():
        # -- module for constructing the phase encoding table
        phe_segment = 1.0 / RAREFACTOR
        phe_step = NY // RAREFACTOR
        phe_table = np.zeros(shape=NY)
        for phe in range(phe_step):
            phe_table[phe * RAREFACTOR] = -1.0 * (phe_step - phe * 2.0) / NY
            for rare in range(1, RAREFACTOR):
                if phe_table[phe * RAREFACTOR] >= 0:
                    phe_table[phe * RAREFACTOR + rare] = phe_table[phe * RAREFACTOR] + rare * phe_segment
                else:
                    phe_table[phe * RAREFACTOR + rare] = phe_table[phe * RAREFACTOR] - rare * phe_segment
        phe_table = (np.round((phe_table * (NY * 0.5)) + (NY * 0.5))).astype(np.int32)
        return phe_table

    # -- main function operations
    fid_data = _read_fid_data(fid_path)
    phe_table = _get_phe_table()

    # preallocate memory for the reconstructed magnitude image
    if zero_pad:
        image_space = np.zeros(shape=(int(NX * 1.5), int(NY * 1.5), int(NZ * NZ0 * 1.5), NVOL))
    else:
        image_space = np.zeros(shape=(int(NX), int(NY), int(NZ * NZ0), NVOL))

    # loop over individual volumes in dataset
    for curr_vol in range(NVOL):
        # combine real and imaginary channels into complex data for the current volume
        fid_data_curr_vol_cplx = fid_data[0, :9 * NX, :, :, :, :, curr_vol] + \
                                 1j * fid_data[1, :9 * NX, :, :, :, :, curr_vol]

        # preallocate memory to populate the rearranged and shifted current volume
        fid_data_curr_vol = np.zeros(shape=(NX, NAVG, NY0, NY1, NZ0, NZ), dtype=complex)

        # separate the gradient and spin echoes and rearrange them in the fid_data-space
        for pos in range(NZ0):
            start_pos = 2 * pos * NX
            fid_data_curr_vol[:, :, :, :, pos, :] = fid_data_curr_vol_cplx[start_pos:start_pos + NX, :, :, :, :]

        # -------------------------------------------------------------------------------------------------------------#
        # -- module for correcting shifts along fid_data in the readout direction
        for curr_avg in range(NAVG):
            odd_idx = np.zeros(shape=NZ0, dtype=int)
            even_idx = np.zeros(shape=NZ0, dtype=int)

            # check scan shift due to unbalanced readout gradients
            for shift_idx in range(NZ0):
                odd_idx[shift_idx] = np.argmax(
                    np.abs(fid_data_curr_vol[:, curr_avg, RAREFACTOR, NY1 // 2, shift_idx, NZ // 2]))
                even_idx[shift_idx] = np.argmax(
                    np.abs(fid_data_curr_vol[:, curr_avg, RAREFACTOR + 1, NY1 // 2, shift_idx, NZ // 2]))

            # obtain the shifts
            odd_idx = NX // 2 - odd_idx
            shift_1 = np.max(np.abs(odd_idx))
            even_idx = NX // 2 - even_idx
            shift_2 = np.max(np.abs(even_idx))
            shift = np.max((shift_1, shift_2))

            # apply the shifts echo-by-echo
            for curr_echo in range(NY0):
                temp = fid_data_curr_vol_cplx[:, curr_avg, curr_echo, :, :]
                # even echoes
                if curr_echo % 2 == 0:
                    for curr_shift in range(NZ0):
                        temp_shifted = np.roll(temp, odd_idx[curr_shift], axis=0)
                        start_pos = 2 * curr_shift * NX
                        fid_data_curr_vol[:, curr_avg, curr_echo, :, curr_shift, :] = temp_shifted[
                                                                                      start_pos:start_pos + NX, :, :]
                # odd echoes
                else:
                    for curr_shift in range(NZ0):
                        temp_shifted = np.roll(temp, even_idx[curr_shift], axis=0)
                        start_pos = 2 * curr_shift * NX
                        fid_data_curr_vol[:, curr_avg, curr_echo, :, curr_shift, :] = temp_shifted[
                                                                                      start_pos:start_pos + NX, :, :]

            # apply the final shifts
            fid_data_curr_vol[:shift, curr_avg, :, :, :, :] = 0.0
            fid_data_curr_vol[-shift:, curr_avg, :, :, :, :] = 0.0
        # -- end loop for correcting shifts
        # -------------------------------------------------------------------------------------------------------------#

        # -------------------------------------------------------------------------------------------------------------#
        # -- module for signal processing along the X direction
        # apodize, then zero pad, then go to the X, ky, kz hybrid domain
        wind_fun_x = (signal.windows.tukey(fid_data_curr_vol.shape[0], APZ_FACTOR[2])).reshape((-1, 1, 1, 1, 1, 1),
                                                                                               order='F')
        fid_data_curr_vol *= wind_fun_x
        if zero_pad:
            fid_data_curr_vol = np.pad(fid_data_curr_vol, ((int(np.floor(NX / 4)), int(np.ceil(NX / 4))),
                                                           (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)),
                                       mode='constant', constant_values=0)
        fid_data_curr_vol = np.fft.ifftshift(np.fft.ifft(fid_data_curr_vol, axis=0), axes=0)
        # -------------------------------------------------------------------------------------------------------------#

        # -------------------------------------------------------------------------------------------------------------#
        # -- module for correcting fid_data phase offsets in the slice-select (z) direction
        for curr_avg in range(NAVG):

            # compute phase offsets
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                phase_1 = fid_data_curr_vol[:, curr_avg, RAREFACTOR, :, :, :]
                phase_1 /= np.abs(phase_1)
                phase_1[np.isinf(phase_1) | np.isnan(phase_1)] = 0.0

                phase_2 = fid_data_curr_vol[:, curr_avg, RAREFACTOR + 1, :, :, :]
                phase_2 /= np.abs(phase_2)
                phase_2[np.isinf(phase_2) | np.isnan(phase_2)] = 0.0

            # compute phase at centre of fid_data
            phase_central = np.conj(phase_1[:, NY1 // 2, NZ0 // 2, NZ // 2])

            # apply phase offsets
            for step_y in range(NY1):
                for step_z1 in range(NZ):
                    for step_z2 in range(NZ0):
                        # multiply by conjugate
                        phase_1[:, step_y, step_z2, step_z1] *= phase_central
                        phase_2[:, step_y, step_z2, step_z1] *= phase_central

                        # smooth because an artificial line may appear in the image along the frequency-encode direction
                        phase_1[:, step_y, step_z2, step_z1] = signal.convolve(phase_1[:, step_y, step_z2, step_z1],
                                                                               smoothing_kernel,
                                                                               mode='same')
                        phase_2[:, step_y, step_z2, step_z1] = signal.convolve(phase_2[:, step_y, step_z2, step_z1],
                                                                               smoothing_kernel,
                                                                               mode='same')
                        # apply phase shift echo-by-echo
                        for curr_echo in range(RAREFACTOR):
                            # even echoes - phase 1
                            if curr_echo % 2 == 0:
                                fid_data_curr_vol[:, curr_avg, curr_echo, step_y, step_z2, step_z1] *= \
                                    np.conj(phase_1[:, step_y, step_z2, step_z1])
                            # odd echoes - phase 2
                            else:
                                fid_data_curr_vol[:, curr_avg, curr_echo, step_y, step_z2, step_z1] *= \
                                    np.conj(phase_2[:, step_y, step_z2, step_z1])
        # -- end loop for phase offsets
        # -------------------------------------------------------------------------------------------------------------#

        # -------------------------------------------------------------------------------------------------------------#
        # -- module for correctly populating the fid_data into k-space based on the phase-encode table
        # preallocate memory for k-space of the current volume
        if zero_pad:
            k_space_curr_vol = np.zeros(shape=(int(NX * 1.5), NAVG, NY, NZ0 * NZ), dtype=complex)
        else:
            k_space_curr_vol = np.zeros(shape=(int(NX), NAVG, NY, NZ0 * NZ), dtype=complex)

        # populate k-space with the aid of the phase-encode table
        for step_y in range(NY1):
            for phe in range(RAREFACTOR):
                for step_z2 in range(NZ0):
                    start_pos = step_z2 * NZ
                    end_pos = (step_z2 + 1) * NZ
                    k_space_curr_vol[:, :, phe_table[phe + step_y * RAREFACTOR], start_pos:end_pos] = fid_data_curr_vol[
                                                                                                      :, :, phe, step_y,
                                                                                                      step_z2, :]
        # -- end loop for populating k-space using phe table
        # -------------------------------------------------------------------------------------------------------------#

        # -------------------------------------------------------------------------------------------------------------#
        # -- module for final window apodization, zero-padding, and fourier reconstruction
        # final check for NaNs and infs
        k_space_curr_vol[np.isnan(k_space_curr_vol) | np.isinf(k_space_curr_vol)] = 0.0

        # apodization along NY and NZ
        wind_fun_y = (signal.windows.tukey(k_space_curr_vol.shape[2], APZ_FACTOR[1])).reshape((1, 1, -1, 1),
                                                                                              order='F')
        wind_fun_z = (signal.windows.tukey(k_space_curr_vol.shape[3], APZ_FACTOR[2])).reshape((1, 1, 1, -1),
                                                                                              order='F')
        k_space_curr_vol = k_space_curr_vol * wind_fun_y * wind_fun_z

        # zero padding along NY and NZ
        if zero_pad:
            k_space_curr_vol = np.pad(k_space_curr_vol,
                                      ((0, 0), (0, 0), (int(np.floor(NY / 4)), int(np.ceil(NY / 4))),
                                       (int(np.floor(NZ * NZ0 / 4)), int(np.ceil(NZ * NZ0 / 4)))),
                                      mode='constant', constant_values=0)

        # FFT along NY and NZ
        image_space_curr_vol = np.fft.ifftshift(np.fft.ifftn(k_space_curr_vol, axes=[2, 3]), axes=[2, 3])

        # combine averages along NAVG and populate the final magnitude image space
        image_space_curr_vol = np.sqrt(
            np.abs(image_space_curr_vol[:, 0, :, :]) ** 2 + np.abs(image_space_curr_vol[:, 1, :, :]) ** 2)
        image_space[:, :, :, curr_vol] = image_space_curr_vol
    # -- end loop over individual volumes

    # return the reconstructed image magnitude
    return image_space


class Reconstructor:
    """
    A custom reconstruction suite for the data acquired ex vivo from Mbp-KO mouse strains. This class leverages the
    methods implemented in the brkraw package (https://brkraw.github.io/) to facilitate reconstruction. For more
    information about the Mbp-KO mouse strains, please consult the following literature:
    [1]  Bagheri, H., Friedman, H., Hadwen, A., Jarweh, C., Cooper, E., Oprea, L., Guerrier, C., Khadra, A., Collin, A.,
    Cohen-Adad, J., Young, A., Victoriano, G. M., Swire, M., Jarjour, A., Bechler, M. E., Pryce, R. S., Chaurand, P.,
    Cougnaud, L., Vuckovic, D., ... Peterson, A. C. (2024). Myelin basic protein mRNA levels affect myelin sheath
    dimensions, architecture, plasticity, and density of resident glial cells. Glia, 1–22.
    https://doi.org/10.1002/glia.24589
    """

    def __init__(self, input_study_path: str):
        """
        Parameters
        ----------
        input_study_path : str
            A string or path-like object which points to a Pavarision 5.1 dataset collected from a single subject in a
            single session. If a folder with multiple subjects or sessions is provided, unexpected behavior may occur.

        A typical Paravision 5.1 dataset usually looks something like the following in a Linux terminal:
        $ cd <input_study_path>
        $ tree -L 1
        .
        ├── 14
        ├── 16
        ├── 18
        ├── 20
        ├── 22
        ├── 25
        ├── 26
        ├── AdjResult
        ├── AdjStatePerStudy
        └── subject
        """
        self.study_path = input_study_path
        self.pv_data_set = bru.load(self.study_path)
        if not self.pv_data_set.is_pvdataset:
            raise ValueError(f"Folder at path {self.study_path} is not a ParaVision dataset.")
        self.study_scan_dict = self._populate_scan_dict()
        print(f"\033[36m{'Loaded Study: ' + self.pv_data_set.pvobj.subj_id}\033[0m")
        self._print_study_header()

    def __repr__(self):
        return f"Custom PV5.1 Reconstructor Object (study_path='{self.study_path}')"

    # -- helper functions
    def _populate_scan_dict(self):
        # cycle through all scan IDs and populate a dictionary
        study_scan_dict = {}
        for scan_id in range(self.pv_data_set.num_scans):
            curr_scan = self.pv_data_set.pvobj.avail_scan_id[scan_id]
            curr_method = self.pv_data_set.get_method(curr_scan).parameters['Method']
            if curr_method == 'dWGRASE5bz5e_dwu':
                curr_method = 'dwGRASE'
            if curr_method not in study_scan_dict:
                study_scan_dict[curr_method] = [curr_scan]
            else:
                study_scan_dict[curr_method].append(curr_scan)
        return study_scan_dict

    def _print_study_header(self):
        # determine the maximum length of the values string
        max_values_length = max(len(", ".join(map(str, values))) for values in self.study_scan_dict.values())

        # print header
        print(f"{'Protocol':<{max_values_length + 5}} {'Scan IDs':>{max_values_length}}")

        # print each key-value pair in the specified format
        for key, values in self.study_scan_dict.items():
            values_str = ', '.join(map(str, values))
            print(f"{key:<{max_values_length + 5}} {values_str:>{max_values_length}}")
        return None

    @staticmethod
    def _populate_json_dict():
        return {
            "Manufacturer": "Bruker",
            "ManufacturersModelName": "Pharmascan",
            "MagneticFieldStrength": 7.0,
            "SoftwareVersion": "ParaVision 5.1",
            "InstitutionName": "McConnell Brain Imaging Centre",
            "InstitutionAddress": "3801 rue University, Montreal, QC, Canada",
            "InstitutionalDepartmentName": "https://www.rudkolab.com/",
            "Species": "Mouse",
            "Age(Days)": 30
        }

    # -- functions for calling protocol-specific recon
    def recon_mgre(self):
        """
        A wrapper for recon_mgre() in pv_custom_recon.py. Reconstructs all mGRE data in self.study_path and
        concatenates them into single magnitude and phase NIfTIs, echo times, and json sidecar files.

        Parameters
        ----------

        Returns
        -------
        image_space_mag : nibabel.Nifti1Image
            The reconstructed mGRE image space magnitude volumes as (Nx, Ny, Nz, NECHO).
        image_space_phs : nibabel.Nifti1Image
            The reconstructed mGRE image space phase volumes as (Nx, Ny, Nz, NECHO).
        json_sidecar : dict
            A dictionary object which can be saved with json.dump().
        echo_times : numpy.ndarray
            Corresponding echo times (NECHO, 1).
        """
        if len(self.study_scan_dict['MGE']) == 0:
            raise FileNotFoundError(f"The study '{self.study_path}' has no mGRE acquisitions.")

        curr_scan_id = self.study_scan_dict['MGE'][0]
        print(f"\033[36m{'--> Reconstructing scan ID ' + str(curr_scan_id) + '... '}\033[0m", end="")

        # get the path to the mGRE folder in the study
        curr_mgre_path = os.path.join(self.study_path, str(curr_scan_id))

        # get the matrix dimensions from the pv_data_set object
        KY, KX, KZ, NECHO = self.pv_data_set.get_matrix_size(curr_scan_id, 1)

        # get the data matrix for complex image space
        image_space = recon_mgre(fid_path=curr_mgre_path, matrix_dims=(KX, KY, KZ, NECHO))

        # apply phase and slice offsets from acqp
        spatial_resol = 0.1 * np.ones(3)
        acqp = self.pv_data_set.get_acqp(curr_scan_id).parameters
        offsets = {
            'ACQ_phase2_offset': (0, spatial_resol[0]),
            'ACQ_phase1_offset': (1, spatial_resol[1]),
            'ACQ_slice_offset':  (2, -spatial_resol[2])
        }
        for key, (axis, resol) in offsets.items():
            image_space = np.roll(image_space, int(acqp[key] / resol), axis)

        # permute dimensions to align with true sample orientation within magnet bore
        image_space = np.transpose(image_space, (1, 0, 2, 3))

        # separate the magnitude and phase
        image_space_mag = np.abs(image_space)
        image_space_phs = np.angle(image_space)

        # construct a Nifti1 object using the native affine
        affine = self.pv_data_set.get_affine(curr_scan_id, 1)
        affine[np.isclose(affine, 0.0)] = 0.0
        affine[:3, :3] = np.where(affine[:3, :3] != 0, np.sign(affine[:3, :3]) * spatial_resol[0], affine[:3, :3])
        image_space_mag = nib.Nifti1Image(dataobj=image_space_mag.astype(np.float32), affine=affine)
        image_space_phs = nib.Nifti1Image(dataobj=image_space_phs.astype(np.float32), affine=affine)
        image_space_mag.header.set_xyzt_units(xyz='mm')
        image_space_phs.header.set_xyzt_units(xyz='mm')

        # construct a JSON sidecar to store metadata
        json_sidecar = self._populate_json_dict()
        params = self.pv_data_set.get_method(curr_scan_id).parameters
        params_dict = {
            'Method': 'MGE',
            'EchoTime': params['PVM_EchoTime'] * 1e-3,
            'EchoSpacing': params['EchoSpacing'] * 1e-3,
            'NumberOfEchoes': params['PVM_NEchoImages'],
            'RepetitionTime': params['PVM_RepetitionTime'] * 1e-3,
            'NumberOfAverages': params['PVM_NAverages'],
            'FlipAngle': params['PVM_ExcPulseAngle'],
            'B0Direction': [0, 1, 0],
            'BandWidth': 100000
        }
        json_sidecar.update(params_dict)
        echo_times = np.round(params['EffectiveTE']) * 1e-3
        print(f"\033[36m{'[DONE]'}\033[0m", '\n', end="")
        return image_space_mag, image_space_phs, json_sidecar, echo_times

    def recon_mese(self, zero_pad: bool = False, apodization_factor: float = 0.0):
        """
        A wrapper for recon_mese() in pv_custom_recon.py. Reconstructs all MESE data in self.study_path and
        concatenates them into single magnitude NIfTIs, echo times, and json sidecar files.

        Parameters
        ----------
        zero_pad : bool
            Toggle zero-padding to reconstruct at native resolution (False) or zero-pad to match native mGRE acquisition
            resolution (True).
        apodization_factor : float
            Apodization factor [0.0, 1.0] for the Tukey window applied to each k-space dimension (Kx, Ky, Kz).

        Returns
        -------
        image_space_mag : nibabel.Nifti1Image
            The reconstructed MESE image space magnitude volumes as (Nx, Ny, Nz, NECHO).
        json_sidecar : dict
            A dictionary object which can be saved with json.dump().
        echo_times : numpy.ndarray
            Corresponding echo times (NECHO, 1).
        """
        if len(self.study_scan_dict['MSME']) == 0:
            raise FileNotFoundError(f"The study '{self.study_path}' has no MESE acquisitions.")

        curr_scan_id = self.study_scan_dict['MSME'][0]
        print(f"\033[36m{'--> Reconstructing scan ID ' + str(curr_scan_id) + '... '}\033[0m", end="")

        # get the path to the MESE folder in the study
        curr_mese_path = os.path.join(self.study_path, str(curr_scan_id))

        # get the matrix dimensions from the pv_data_set object
        KY, KX, KZ, NECHO = self.pv_data_set.get_matrix_size(curr_scan_id, 1)

        # get the data matrix for magnitude and phase
        image_space = recon_mese(fid_path=curr_mese_path,
                                 matrix_dims=(KX, KY, KZ, NECHO),
                                 zero_pad=zero_pad,
                                 apodization_factor=apodization_factor)

        # apply phase and slice offsets from acqp parameters
        spatial_resol = 0.1 * np.ones(3) if zero_pad else 0.15 * np.ones(3)
        acqp = self.pv_data_set.get_acqp(curr_scan_id).parameters
        offsets = {
            'ACQ_phase2_offset': (0, spatial_resol[0]),
            'ACQ_phase1_offset': (1, spatial_resol[1]),
            'ACQ_slice_offset': (2, -spatial_resol[2])
        }
        for key, (axis, resol) in offsets.items():
            image_space = np.roll(image_space, int(acqp[key] / resol), axis)

        # permute dimensions to align with true sample orientation within magnet bore
        image_space = np.transpose(image_space, (1, 0, 2, 3))
        image_space_mag = np.abs(image_space)

        # construct a Nifti1 object using the native affine
        affine = self.pv_data_set.get_affine(curr_scan_id, 1)
        affine[np.isclose(affine, 0.0)] = 0.0
        affine[:3, :3] = np.where(affine[:3, :3] != 0, np.sign(affine[:3, :3]) * spatial_resol[0], affine[:3, :3])
        image_space_mag = nib.Nifti1Image(dataobj=image_space_mag.astype(np.float32), affine=affine)
        image_space_mag.header.set_xyzt_units(xyz='mm')

        # construct a JSON sidecar to store metadata
        json_sidecar = self._populate_json_dict()
        params = self.pv_data_set.get_method(curr_scan_id).parameters
        params_dict = {
            'Method': 'MESE',
            'EchoTime': params['PVM_EchoTime'] * 1e-3,
            'EchoSpacing': params['PVM_EchoTime'] * 1e-3,
            'NumberOfEchoes': params['PVM_NEchoImages'],
            'RepetitionTime': params['PVM_RepetitionTime'] * 1e-3,
            'NumberOfAverages': params['PVM_NAverages'],
            'FlipAngle': params['RfcFlipAngle'],
            'B0Direction': [0, 1, 0],
            'TukeyApodizationFactor': apodization_factor,
            'ZeroPadding': str(zero_pad)
        }
        json_sidecar.update(params_dict)
        echo_times = np.array((params['EffectiveTE'])) * 1e-3
        print(f"\033[36m{'[DONE]'}\033[0m", '\n', end="")
        return image_space_mag, json_sidecar, echo_times

    def recon_dwi(self, zero_pad: bool = False, apodization_factor: float = 0.0):
        """
        A wrapper for recon_dwgrase() in pv_custom_recon.py. Reconstructs all dwGRASE data in self.study_path and
        concatenates them into single NIfTI, b-val, b-vec, and json sidecar files.

        Parameters
        ----------
        zero_pad : bool
            Toggle zero-padding to reconstruct at native resolution (False) or zero-pad to match native mGRE acquisition
            resolution (True).
        apodization_factor : float
            Apodization factor [0.0, 1.0] for the Tukey window applied to each k-space dimension (Kx, Ky, Kz).

        Returns
        -------
        dw_images : nibabel.Nifti1Image
            The reconstructed dwGRASE image space volumes as (Nx, Ny, Nz, NVOL).
        json_sidecar : dict
            A dictionary object which can be saved with json.dump().
        b_values : numpy.ndarray
            Corresponding b-values (NVOL, 1).
        b_vectors : numpy.ndarray
            Corresponding unit b-vectors (NVOL, 3).
        """
        if len(self.study_scan_dict['dwGRASE']) == 0:
            raise FileNotFoundError(f"The study '{self.study_path}' has no dwGRASE acquisitions.")

        # threshold for "zero" diffusion weighting
        B0_THRESH = 100

        # lists to populate
        b_values = []
        b_vectors = []
        dw_images = []

        # iterate over the dwGRASE scan ids and collect b-values, b-vectors, and the reconstructed images
        for curr_dw_scan in range(len(self.study_scan_dict['dwGRASE'])):
            curr_scan_id = self.study_scan_dict['dwGRASE'][curr_dw_scan]
            curr_dw_path = os.path.join(self.study_path, str(curr_scan_id))
            print(f"\033[36m{'--> Reconstructing scan ID ' + str(curr_scan_id) + '... '}\033[0m", end="")

            # get the b-values
            curr_b_values = self.pv_data_set.get_method(curr_scan_id).parameters['bvalues']
            curr_b_values = list(map(int, curr_b_values))
            curr_b_values = [value if value >= B0_THRESH else 0 for value in curr_b_values]
            b_values.append(curr_b_values)

            # get the b-vectors
            curr_b_vectors = self.pv_data_set.get_method(curr_scan_id).parameters['PVM_DwDir']

            # insert zeros into b-vector matrix where b-values are zero
            curr_b_vectors_updated = []
            b_vector_index = 0
            for value in curr_b_values:
                if value == 0:
                    # insert a zero triplet for zero b-value
                    curr_b_vectors_updated.append([0.0, 0.0, 0.0])
                else:
                    # insert the next b-vector from the original list
                    curr_b_vectors_updated.append(curr_b_vectors[b_vector_index])
                    b_vector_index += 1
            b_vectors.append(curr_b_vectors_updated)

            # get the data matrix
            image_space = recon_dwgrase(fid_path=curr_dw_path,
                                        nvol=len(curr_b_values),
                                        zero_pad=zero_pad,
                                        apodization_factor=apodization_factor)
            dw_images.append(image_space)
            print(f"\033[36m{'[DONE]'}\033[0m", '\n', end="")
        # -- end loop over all dw scans

        # sort the b-values, b-vectors, and image matrix according to the b-values
        b_values = np.array(list(itertools.chain(*b_values)))
        b_value_index = np.argsort(b_values)
        b_values = b_values[b_value_index]
        b_vectors = np.array(list(itertools.chain(*b_vectors)))
        b_vectors = b_vectors[b_value_index]
        dw_images = np.concatenate(dw_images, axis=3)
        dw_images = dw_images[:, :, :, b_value_index]

        # apply phase and slice offsets from acqp parameters
        spatial_resol = 0.1 * np.ones(3) if zero_pad else 0.15 * np.ones(3)
        acqp = self.pv_data_set.get_acqp(self.study_scan_dict['dwGRASE'][-1])
        offsets = {
            'ACQ_phase2_offset': (0, spatial_resol[0]),
            'ACQ_phase1_offset': (1, spatial_resol[1]),
            'ACQ_slice_offset': (-2, spatial_resol[2])
        }
        for key, (axis, resol) in offsets.items():
            dw_images = np.roll(dw_images, int(acqp.parameters[key] / resol), axis)

        # permute dimensions to align with true sample orientation within magnet bore
        dw_images = np.flip(np.transpose(dw_images, (1, 0, 2, 3)), axis=2)
        b_vectors[:, [0, 1, 2]] = b_vectors[:, [2, 0, 1]]

        # construct a Nifti1 object using the native affine
        affine = self.pv_data_set.get_affine(self.study_scan_dict['dwGRASE'][-1], 1)
        affine[np.isclose(affine, 0.0)] = 0.0
        affine[:3, :3] = np.where(affine[:3, :3] != 0, np.sign(affine[:3, :3]) * spatial_resol[0], affine[:3, :3])
        dw_images = nib.Nifti1Image(dataobj=dw_images.astype(np.float32), affine=affine)
        dw_images.header.set_xyzt_units(xyz='mm')

        # construct a JSON sidecar to store metadata
        json_sidecar = self._populate_json_dict()
        params = self.pv_data_set.get_method(self.study_scan_dict['dwGRASE'][-1]).parameters
        params_dict = {
            'Method': 'dwGRASE',
            'EchoTime': params['PVM_EchoTime'] * 1e-3,
            'RepetitionTime': params['PVM_RepetitionTime'] * 1e-3,
            'SmallDelta': params['DiffusionGradientDuration'] * 1e-3,
            'BigDelta': params['DiffusionGradientSeparation'] * 1e-3,
            'B0Direction': [0, 1, 0],
            'TukeyApodizationFactor': apodization_factor,
            'ZeroPadding': str(zero_pad)
        }
        json_sidecar.update(params_dict)
        return dw_images, json_sidecar, b_values, b_vectors

    def recon_epi(self):

        return None
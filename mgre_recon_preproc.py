import os
import json
import numpy as np
import nibabel as nib


# Add the directory containing Reconstructor class to the Python path
import sys
project_dir = '/export01/data/vgrouza/mbpko'
sys.path.append(project_dir)
from src import pv_custom_recon
from src import bipolar_corrections
from src import mwf_rpca

# create path variables for input and output data
raw_data_dir = os.path.join(project_dir, 'data/mri/raw')
int_data_dir = os.path.join(project_dir, 'data/mri/interim')


def recon_mgres(input_study_path: str):
    """
    A simple function to reconstruct the mGRE images in the MbpKO dataset.
    """
    # set up paths and create directories, if needed
    input_study_full_path = os.path.join(raw_data_dir, input_study_path)
    output_study_full_path = os.path.join(int_data_dir, input_study_path, 'mgre')
    os.makedirs(output_study_full_path, exist_ok=True)

    # instantiate the reconstructor object and carry out reconstruction
    print('Reconstructing ' + input_study_path + '...')
    recon_obj = pv_custom_recon.Reconstructor(input_study_full_path)
    image_space_mag, image_space_phs, json_sidecar, echo_times = recon_obj.recon_mgre()

    # save the images and echo times
    nib.save(img=image_space_mag,
             filename=os.path.join(output_study_full_path, input_study_path + '_mgre-mag.nii.gz'))
    nib.save(img=image_space_phs,
             filename=os.path.join(output_study_full_path, input_study_path + '_mgre-phs.nii.gz'))
    np.savetxt(os.path.join(output_study_full_path, input_study_path + '_echo-times.txt'), echo_times)

    # update the sidecar and dump
    sidecar_updates = {
        'echo_times': input_study_path + '_echo-times.txt',
    }
    json_sidecar.update(sidecar_updates)
    with open(os.path.join(output_study_full_path, input_study_path + '_mgre-mag.json'), 'w') as json_file:
        json.dump(json_sidecar, json_file, indent=4)
    with open(os.path.join(output_study_full_path, input_study_path + '_mgre-phs.json'), 'w') as json_file:
        json.dump(json_sidecar, json_file, indent=4)
    return None


def recon_meses(input_study_path: str, zero_pad: bool = True, apodization_factor: float = 0.25):
    """
    A simple function to reconstruct the mGRE images in the MbpKO dataset.
    """
    # set up paths and create directories, if needed
    input_study_full_path = os.path.join(raw_data_dir, input_study_path)
    output_study_full_path = os.path.join(int_data_dir, input_study_path, 'mese')
    os.makedirs(output_study_full_path, exist_ok=True)

    # instantiate the reconstructor object and carry out reconstruction
    print('Reconstructing ' + input_study_path + '...')
    recon_obj = pv_custom_recon.Reconstructor(input_study_full_path)
    image_space, json_sidecar, echo_times = recon_obj.recon_mese(zero_pad=zero_pad,
                                                                 apodization_factor=apodization_factor)

    # save the images and echo times
    nib.save(img=image_space,
             filename=os.path.join(output_study_full_path, input_study_path + '_mese.nii.gz'))
    np.savetxt(os.path.join(output_study_full_path, input_study_path + '_echo-times.txt'), echo_times)

    # update the sidecar and dump
    sidecar_updates = {
        'echo_times': input_study_path + '_echo-times.txt',
    }
    json_sidecar.update(sidecar_updates)
    with open(os.path.join(output_study_full_path, input_study_path + '_mese.json'), 'w') as json_file:
        json.dump(json_sidecar, json_file, indent=4)
    return None


def preproc_mgres(input_study_path: str):
    """
    A simple function to preprocess mGRE images for downstream model fitting.
    """
    # set up paths
    output_study_full_path = os.path.join(int_data_dir, input_study_path, 'dwi')

    # load the reconstruted image and the b_values
    print('Preprocessing ' + input_study_path + '...', end="")
    image_space_mag = nib.load(os.path.join(output_study_full_path, input_study_path + '_mgre-mag.nii.gz'))
    image_space_phs = nib.load(os.path.join(output_study_full_path, input_study_path + '_mgre-phs.nii.gz'))
    image_space_complex = image_space_mag.get_fdata() * np.exp(1j * image_space_phs.get_fdata())

    # do bipolar gradient k-space magnitude correction
    image_space_complex = bipolar_corrections.apply_phase_correction(image_space_complex)

    # do bipolar gradient k-space phase correction
    image_space_complex = bipolar_corrections.apply_misregistration_correction(image_space_complex)

    # separate back into magnitude and phase
    image_space_mag = nib.Nifti1Image(dataobj=np.abs(image_space_comlex),
                                      affine=image_space_mag.affine,
                                      header=image_space_mag.header)
    image_space_phs = nib.Nifti1Image(dataobj=np.angle(image_space_comlex),
                                      affine=image_space_phs.affine,
                                      header=image_space_phs.header)
    # generate a binary mask
    binary_mask = median_otsu(np.sqrt(np.sum(abs(image_space) ** 2, axis=3)), median_radius=1, numpass=2)[1]

    # do denoising, then Gibbs unringing, then intra-echo and through-echo phase unwrapping

    # save niftis

    # update JSON sidecars and dump

    # compute MWF through rPCA
    # mask away all background
    image_space_mag = image_space_mag * np.repeat(binary_mask[:, :, :, np.newaxis], 24, axis=3)

    # instantiate the TE vector
    te_vec = np.arange(start=2, stop=50, step=2) * 1e-3

    # summon the rPCA
    rpca = mwf_rpca.RobustPCA()
    rpca.fit(mgre_data=mgre_data, binary_mask=mask_img, echo_times=te_vec)
    x_tensor = rpca.X

    # explicitly define components
    L1_to_save = x_tensor[:, :, :, :, 0]
    L2_to_save = x_tensor[:, :, :, :, 1]
    S_to_save = x_tensor[:, :, :, :, 2]

    # compute MWF
    np.seterr(divide='ignore', invalid='ignore')
    mwf_to_save = np.true_divide(L2_to_save[:, :, :, 0], (L2_to_save[:, :, :, 0] + L1_to_save[:, :, :, 0]))
    mwf_to_save = mwf_to_save * binary_erosion(mask_img.get_fdata())
    
    print('[DONE]')
    pass



if __name__ == '__main__':
    os.chdir(raw_data_dir)
    study_dirs = sorted([name for name in os.listdir(os.getcwd()) if os.path.isdir(os.path.join(os.getcwd(), name))])
    for curr_study_path in study_dirs:
        recon_mgres(input_study_path=curr_study_path)
        recon_meses(input_study_path=curr_study_path, zero_pad=True, apodization_factor=0.35)
        #preproc_dwis(curr_study_path)
    os.chdir(project_dir)



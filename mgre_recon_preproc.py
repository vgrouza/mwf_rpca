import os
import json
import numpy as np
import nibabel as nib
import subprocess

from dipy.denoise.gibbs import gibbs_removal
from dipy.segment.mask import median_otsu
from dipy.denoise.localpca import mppca

# Add the directory containing Reconstructor class to the Python path
import sys
project_dir = '/export01/data/vgrouza/mbpko'
sys.path.append(project_dir)
from src import pv_custom_recon
from src import bipolar_corrections

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


def preproc_mgres(input_study_path: str):
    """
    A simple function to preprocess mGRE images for downstream model fitting.
    """
    # set up paths
    output_study_full_path = os.path.join(int_data_dir, input_study_path, 'mgre')

    # load the reconstruted image and the b_values
    print('Preprocessing ' + input_study_path + '...', end="")
    image_space_mag = nib.load(os.path.join(output_study_full_path, input_study_path + '_mgre-mag.nii.gz'))
    image_space_phs = nib.load(os.path.join(output_study_full_path, input_study_path + '_mgre-phs.nii.gz'))
    image_space_complex = image_space_mag.get_fdata() * np.exp(1j * image_space_phs.get_fdata())
    affine = image_space_mag.affine
    header = image_space_mag.header

    # do bipolar gradient k-space magnitude correction
    image_space_complex, alpha_max = bipolar_corrections.apply_phase_correction(image_space_complex,
                                                                                readout_axis=1)

    # do bipolar gradient k-space phase correction
    image_space_complex, beta_max = bipolar_corrections.apply_misregistration_correction(image_space_complex,
                                                                                         readout_axis=1)

    # separate back into magnitude and phase
    image_space_mag = np.abs(image_space_complex)
    image_space_phs = np.angle(image_space_complex)

    # do Gibbs removal on magnitude
    image_space_mag = gibbs_removal(image_space_mag, slice_axis=2, num_processes=16)
    image_space_mag = gibbs_removal(image_space_mag, slice_axis=1, num_processes=16)

    # generate a binary mask
    binary_mask = median_otsu(np.sqrt(np.sum(image_space_mag ** 2, axis=3)), median_radius=1, numpass=2)[1]

    '''
    # unwrap phase - spatial
    for curr_echo in range(image_space_phs.shape[-1]):
        image_space_phs[..., curr_echo] = unwrap3d(np.asfortranarray(image_space_phs[..., curr_echo], 'float32'),
                                                   np.asfortranarray(binary_mask, 'bool'))
        image_space_phs[..., curr_echo] = unwrap3d(np.asfortranarray(image_space_phs[..., curr_echo], 'float32'),
                                                   np.asfortranarray(binary_mask, 'bool'))

    # unwrap phase - temporal
    NX, NY, NZ = binary_mask.shape
    binary_mask = np.reshape(binary_mask, NX * NY * NZ, order='F')
    image_space_phs = np.reshape(image_space_phs, (NX * NY * NZ, image_space_phs.shape[-1]), order='F')
    for curr_vox in range(len(binary_mask)):
        if binary_mask[curr_vox]:
            image_space_phs[curr_vox, :] = np.unwrap(image_space_phs[curr_vox, :], axis=-1)
    binary_mask = np.reshape(binary_mask, (NX, NY, NZ), order='F')
    image_space_phs = np.reshape(image_space_phs, (NX, NY, NZ, image_space_phs.shape[-1]), order='F')
    '''
    # save NIfTIs
    image_space_mag = nib.Nifti1Image(dataobj=image_space_mag *
                                              np.repeat(binary_mask[:, :, :, np.newaxis], image_space_mag.shape[-1],
                                                        axis=3),
                                      affine=affine,
                                      header=header)
    image_space_phs = nib.Nifti1Image(dataobj=image_space_phs,
                                      affine=affine,
                                      header=header)
    binary_mask = nib.Nifti1Image(dataobj=binary_mask.astype(np.int32),
                                  affine=affine)
    binary_mask.header.set_xyzt_units(xyz='mm')

    nib.save(img=image_space_mag,
             filename=os.path.join(output_study_full_path, input_study_path + '_mgre-mag-preproc.nii.gz'))
    nib.save(img=image_space_phs,
             filename=os.path.join(output_study_full_path, input_study_path + '_mgre-phs-preproc.nii.gz'))
    nib.save(img=binary_mask,
             filename=os.path.join(output_study_full_path, input_study_path + '_mgre-binary-mask.nii.gz'))

    # update JSON sidecars and dump
    # --> magnitude
    with open(os.path.join(output_study_full_path, input_study_path + '_mgre-mag.json'), 'r') as fid:
        json_sidecar_mag = json.load(fid)
    sidecar_updates_mag = {
        'bipolar_alpha_max': alpha_max,
        'bipolar_beta_max': beta_max,
        'GibbsUnringing': 'True',
        'BinaryMask': input_study_path + '_mgre-binary-mask.nii.gz',
    }
    json_sidecar_mag.update(sidecar_updates_mag)
    with open(os.path.join(output_study_full_path, input_study_path + '_mgre-mag-preproc.json'), 'w') as json_file:
        json.dump(json_sidecar_mag, json_file, indent=4)

    # --> phase
    with open(os.path.join(output_study_full_path, input_study_path + '_mgre-phs.json'), 'r') as fid:
        json_sidecar_phs = json.load(fid)
    sidecar_updates_phs = {
        'bipolar_alpha_max': alpha_max,
        'bipolar_beta_max': beta_max,
        'BinaryMask': input_study_path + '_mgre-binary-mask.nii.gz',
    }
    json_sidecar_phs.update(sidecar_updates_phs)
    with open(os.path.join(output_study_full_path, input_study_path + '_mgre-phs-preproc.json'), 'w') as json_file:
        json.dump(json_sidecar_phs, json_file, indent=4)

    print('[DONE]')
    return None


def romeo_phaseunwrap_api(input_study_path: str):
    """
    A simple function to finish the preprocessing of mGRE images by calling ROMEO phase unwrapping.
    [1] Dymerska B, Eckstein K, Bachrata B, et al. Phase unwrapping with a rapid opensource minimum spanning tree
    algorithm. Magn Reson Med. 2021; 85: 2294–2308. https://doi.org/10.1002/mrm.28563
    """
    print('Unwrapping ' + input_study_path + '...', end="")
    # set up paths
    output_study_full_path = os.path.join(int_data_dir, input_study_path, 'mgre')

    # set up specific paths to images
    phs_img_path = os.path.join(output_study_full_path, input_study_path + "_mgre-phs-preproc.nii.gz")
    mag_img_path = os.path.join(output_study_full_path, input_study_path + "_mgre-mag-preproc.nii.gz")
    binary_mask_path = os.path.join(output_study_full_path, input_study_path + "_mgre-binary-mask.nii.gz")
    unwrapped_img_path = os.path.join(output_study_full_path, input_study_path + "_mgre-phs-unwrapped.nii.gz")

    # get the echo times
    echo_times = np.loadtxt(os.path.join(output_study_full_path, input_study_path + '_echo-times.txt')) * 1000
    echo_times_str = "[" + ",".join(f"{et:.6f}" for et in echo_times) + "]"

    # set up the Julia command to ROMEO
    julia_cmd = f"""
    julia -e \"
    using ROMEO
    using NIfTI

    # Load .nii files using niread with double quotes for paths
    phs_img = niread(\\\"{phs_img_path}\\\");
    mag_img = niread(\\\"{mag_img_path}\\\");
    binary_mask = niread(\\\"{binary_mask_path}\\\");

    # Convert the binary mask's data to Boolean
    binary_mask_bool = binary_mask .!= 0;

    # Echo times as an array
    TEs = {echo_times_str};

    # Unwrap phase images using ROMEO
    unwrapped_img = unwrap(phs_img; mag=mag_img, mask=binary_mask_bool, TEs=TEs, individual=true, maxseeds=1, merge_regions=true, temporal_uncertain_unwrapping=false, correctglobal=true, wrap_addition=π);

    # Create a new NIfTI object with the unwrapped data and original header and affine
    unwrapped_nii = NIVolume(unwrapped_img);

    # Save the unwrapped image with the same affine and header
    niwrite(\\\"{unwrapped_img_path}\\\", unwrapped_nii);
    \"
    """
    # Execute the command using subprocess.run
    result = subprocess.run(julia_cmd, shell=True, text=True, capture_output=True)

    # write correct affines and headers using nibabel - this is a little awkward, but I don't know anything about Julia
    phs_img_original = nib.load(phs_img_path)
    phs_img_unwrapped = nib.load(unwrapped_img_path).get_fdata()

    # unwrap phase - temporal
    binary_mask = nib.load(binary_mask_path).get_fdata()
    NX, NY, NZ = binary_mask.shape
    binary_mask = np.reshape(binary_mask, NX * NY * NZ, order='F')
    phs_img_unwrapped = np.reshape(phs_img_unwrapped, (NX * NY * NZ, phs_img_unwrapped.shape[-1]), order='F')
    for curr_vox in range(len(binary_mask)):
        if binary_mask[curr_vox]:
            phs_img_unwrapped[curr_vox, :] = np.unwrap(phs_img_unwrapped[curr_vox, :], axis=-1)
    binary_mask = np.reshape(binary_mask, (NX, NY, NZ), order='F')
    phs_img_unwrapped = np.reshape(phs_img_unwrapped, (NX, NY, NZ, phs_img_unwrapped.shape[-1]), order='F')

    nib.save(img=nib.Nifti1Image(dataobj=phs_img_unwrapped * np.repeat(binary_mask[:, :, :, np.newaxis], 24, axis=3),
                                 affine=phs_img_original.affine,
                                 header=phs_img_original.header),
             filename=unwrapped_img_path)
    print('[DONE]')
    return None


#unwrapped_img = unwrap(phs_img; mag=mag_img, mask=binary_mask_bool, TEs=TEs);


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
    image_space_mag, image_space_phs, json_sidecar, echo_times = recon_obj.recon_mese(zero_pad=zero_pad,
                                                                 apodization_factor=apodization_factor)

    # save the images and echo times
    nib.save(img=image_space_mag,
             filename=os.path.join(output_study_full_path, input_study_path + '_mese-mag.nii.gz'))
    nib.save(img=image_space_phs,
             filename=os.path.join(output_study_full_path, input_study_path + '_mese-phs.nii.gz'))
    np.savetxt(os.path.join(output_study_full_path, input_study_path + '_echo-times.txt'), echo_times)

    # update the sidecar and dump
    sidecar_updates = {
        'echo_times': input_study_path + '_echo-times.txt',
    }
    json_sidecar.update(sidecar_updates)
    with open(os.path.join(output_study_full_path, input_study_path + '_mese-mag.json'), 'w') as json_file:
        json.dump(json_sidecar, json_file, indent=4)
    with open(os.path.join(output_study_full_path, input_study_path + '_mese-phs.json'), 'w') as json_file:
        json.dump(json_sidecar, json_file, indent=4)
    print('[DONE]')
    return None


def preproc_meses(input_study_path: str):
    """
    A simple function to preprocess mGRE images for downstream model fitting.
    """
    # set up paths
    output_study_full_path = os.path.join(int_data_dir, input_study_path, 'mese')

    # load the reconstruted image and the b_values
    print('Preprocessing ' + input_study_path + '...', end="")
    image_space_mag = nib.load(os.path.join(output_study_full_path, input_study_path + '_mese-mag.nii.gz'))
    affine = image_space_mag.affine
    header = image_space_mag.header
    image_space_mag = image_space_mag.get_fdata()
    image_space_mag[np.isnan(image_space_mag) | np.isinf(image_space_mag)] = 0.0

    # generate a binary mask
    binary_mask = median_otsu(np.sqrt(np.sum(image_space_mag ** 2, axis=3)), median_radius=1, numpass=2)[1]

    # de denoising
    image_space_mag = mppca(image_space_mag, mask=binary_mask, patch_radius=2)

    # do Gibbs removal on magnitude image
    image_space_mag = gibbs_removal(image_space_mag, slice_axis=2, num_processes=16)
    image_space_mag = gibbs_removal(image_space_mag, slice_axis=1, num_processes=16)

    # final check for NaNs and INFs
    image_space_mag[np.isnan(image_space_mag) | np.isinf(image_space_mag)] = 0.0

    # save NIfTIs
    image_space_mag = nib.Nifti1Image(dataobj=image_space_mag * np.repeat(binary_mask[:, :, :, np.newaxis], image_space_mag.shape[-1], axis=3),
                                      affine=affine,
                                      header=header)

    binary_mask = nib.Nifti1Image(dataobj=binary_mask.astype(np.int32),
                                  affine=affine)
    binary_mask.header.set_xyzt_units(xyz='mm')

    nib.save(img=image_space_mag,
             filename=os.path.join(output_study_full_path, input_study_path + '_mese-mag-preproc.nii.gz'))
    nib.save(img=binary_mask,
             filename=os.path.join(output_study_full_path, input_study_path + '_mese-binary-mask.nii.gz'))

    # update JSON sidecars and dump
    # --> magnitude
    with open(os.path.join(output_study_full_path, input_study_path + '_mese.json'), 'r') as fid:
        json_sidecar_mag = json.load(fid)
    sidecar_updates_mag = {
        'Denoising': 'mppca',
        'GibbsUnringing': 'True',
        'BinaryMask': input_study_path + '_mese-binary-mask.nii.gz',
    }
    json_sidecar_mag.update(sidecar_updates_mag)
    with open(os.path.join(output_study_full_path, input_study_path + '_mese-mag-preproc.json'), 'w') as json_file:
        json.dump(json_sidecar_mag, json_file, indent=4)
    print('[DONE]')
    return None


if __name__ == '__main__':
    os.chdir(raw_data_dir)
    study_dirs = sorted([name for name in os.listdir(os.getcwd()) if os.path.isdir(os.path.join(os.getcwd(), name))])
    for curr_study_path in study_dirs:
        #recon_mgres(input_study_path=curr_study_path)
        #preproc_mgres(curr_study_path)
        #romeo_phaseunwrap_api(curr_study_path)
        recon_meses(input_study_path=curr_study_path, zero_pad=True, apodization_factor=0.35)
        preproc_meses(input_study_path=curr_study_path)
    os.chdir(project_dir)




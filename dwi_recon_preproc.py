import os
import numpy as np
import nibabel as nib
import json

from dipy.denoise.patch2self import patch2self
from dipy.denoise.gibbs import gibbs_removal
from dipy.segment.mask import median_otsu


# Add the directory containing Reconstructor class to the Python path
import sys
project_dir = '/export01/data/vgrouza/mbpko'
sys.path.append(project_dir)
from src import pv_custom_recon

# create path variables for input and output data
raw_data_dir = os.path.join(project_dir, 'data/mri/raw')
int_data_dir = os.path.join(project_dir, 'data/mri/interim')


def recon_dwis(input_study_path: str, zero_pad: bool = True, apodization_factor: float = 0.25):
    """
    A simple function to reconstruct the DWI images in the MbpKO dataset.
    Note: subject 11 - volume 72 and subject 16 - volume 41 carry corrupt data, likely due to a gradient heating issue.
    These are manually "popped" out of the NIfTI and corresponding b-vals/b-vecs of the dataset after reconstruction.
    """
    # set up paths and create directories, if needed
    input_study_full_path = os.path.join(raw_data_dir, input_study_path)
    output_study_full_path = os.path.join(int_data_dir, input_study_path, 'dwi')
    os.makedirs(output_study_full_path, exist_ok=True)

    # instantiate the reconstructor object and carry out reconstruction
    print('Reconstructing ' + input_study_path + '...')
    recon_obj = pv_custom_recon.Reconstructor(input_study_full_path)
    dw_images, json_sidecar, b_values, b_vectors, = recon_obj.recon_dwi(zero_pad=zero_pad,
                                                                        apodization_factor=apodization_factor)
    bad_volumes = {
        'sub-11_strain-1xko': 72,
        'sub-16_strain-3xko': 41
    }
    if input_study_path in bad_volumes:
        image_matrix = dw_images.get_fdata()
        BAD_VOLUME = bad_volumes[input_study_path]
        image_matrix = np.delete(image_matrix, BAD_VOLUME, axis=3)
        b_vectors = np.delete(b_vectors, BAD_VOLUME, axis=0)
        b_values = np.delete(b_values, BAD_VOLUME, axis=0)
        dw_images = nib.Nifti1Image(dataobj=image_matrix.astype(np.float32),
                                    affine=dw_images.affine,
                                    header=dw_images.header)

    # save the images, b-values, and b_vectors
    nib.save(img=dw_images,
             filename=os.path.join(output_study_full_path, input_study_path + '_dwi.nii.gz'))
    np.savetxt(os.path.join(output_study_full_path, input_study_path + '_dwi.bvals'), b_values)
    np.savetxt(os.path.join(output_study_full_path, input_study_path + '_dwi.bvecs'), b_vectors)

    # update the sidecar and dump
    sidecar_updates = {
        'bvals': input_study_path + '_dwi.bvals',
        'bvecs': input_study_path + '_dwi.bvecs'
    }
    json_sidecar.update(sidecar_updates)
    with open(os.path.join(output_study_full_path, input_study_path + '_dwi.json'), 'w') as json_file:
        json.dump(json_sidecar, json_file, indent=4)
    return None


def preproc_dwis(input_study_path: str):
    """
    A simple function to preprocess DWIs for further model fitting. Note that since a dwGRASE sequence was employed, no
    EPI-related distortion correction (topup, eddy, etc) is required.

    [1] Fadnavis, J. Batson, E. Garyfallidis, Patch2Self: Denoising Diffusion MRI with Self-supervised Learning,
    Advances in Neural Information Processing Systems 33 (2020).
    https://doi.org/10.48550/arXiv.2011.01355
    [2] Kellner E, Dhital B, Kiselev VG, Reisert M. Gibbs-ringing artifact removal based on local subvoxel-shifts.
    Magn Reson Med. 2016 Nov;76(5):1574-1581. doi: https://doi.org/10.1002/mrm.26054
    """
    # set up paths
    output_study_full_path = os.path.join(int_data_dir, input_study_path, 'dwi')

    # load the reconstruted image and the b_values
    print('Preprocessing ' + input_study_path + '...', end="")
    dw_images = nib.load(os.path.join(output_study_full_path, input_study_path + '_dwi.nii.gz'))
    b_values = np.loadtxt(os.path.join(output_study_full_path, input_study_path + '_dwi.bvals'))
    with open(os.path.join(output_study_full_path, input_study_path + '_dwi.json'), 'r') as fid:
        json_sidecar = json.load(fid)

    # denoise with patch2self
    dw_images_denoised = patch2self(dw_images.get_fdata(),
                                    b_values,
                                    model='ols',
                                    shift_intensity=True,
                                    clip_negative_vals=False,
                                    b0_threshold=50)

    # correct for Gibbs ringing
    dw_images_denoised = gibbs_removal(dw_images_denoised, slice_axis=2, num_processes=16)
    dw_images_denoised = gibbs_removal(dw_images_denoised, slice_axis=1, num_processes=16)

    # generate a binary mask
    dw_images_denoised[np.isnan(dw_images_denoised) | np.isinf(dw_images_denoised)] = 0.0
    binary_mask = median_otsu(np.sqrt(np.sum(abs(dw_images_denoised[:, :, :, b_values == 0]) ** 2, axis=3)),
                              median_radius=1,
                              numpass=2)[1]

    # apply the binary mask to dwis
    dw_images_denoised *= binary_mask[..., np.newaxis] * np.ones(dw_images_denoised.shape[-1])

    # normalize to mean b=0 image intensity
    b0_mean_intensity = np.mean(np.mean(dw_images_denoised[:, :, :, b_values == 0], axis=3)[binary_mask])
    dw_images_denoised /= b0_mean_intensity

    # construct NIfTI objects and save
    dw_images_denoised = nib.Nifti1Image(dataobj=dw_images_denoised.astype(np.float32),
                                         affine=dw_images.affine,
                                         header=dw_images.header)
    b0_average_image = nib.Nifti1Image(dataobj=np.mean(dw_images_denoised.get_fdata()[:, :, :, b_values == 0], axis=3).astype(np.float32),
                                       affine=dw_images.affine)
    b0_average_image.header.set_xyzt_units(xyz='mm')
    binary_mask = nib.Nifti1Image(dataobj=binary_mask.astype(np.int32),
                                  affine=dw_images.affine)
    binary_mask.header.set_xyzt_units(xyz='mm')

    nib.save(img=dw_images_denoised,
             filename=os.path.join(output_study_full_path, input_study_path + '_dwi-preproc.nii.gz'))
    nib.save(img=b0_average_image,
             filename=os.path.join(output_study_full_path, input_study_path + '_dwi-b0-average.nii.gz'))
    nib.save(img=binary_mask,
             filename=os.path.join(output_study_full_path, input_study_path + '_dwi-binary-mask.nii.gz'))

    # update the sidecar and dump
    sidecar_updates = {
        'Denoising': 'patch2self',
        'GibbsUnringing': 'True',
        'BinaryMask': input_study_path + '_dwi-binary-mask.nii.gz',
    }
    json_sidecar.update(sidecar_updates)
    with open(os.path.join(output_study_full_path, input_study_path + '_dwi-preproc.json'), 'w') as json_file:
        json.dump(json_sidecar, json_file, indent=4)
    print('[DONE]')
    return None


if __name__ == '__main__':
    os.chdir(raw_data_dir)
    study_dirs = sorted([name for name in os.listdir(os.getcwd()) if os.path.isdir(os.path.join(os.getcwd(), name))])
    for curr_study_path in study_dirs:
        recon_dwis(input_study_path=curr_study_path, zero_pad=True, apodization_factor=0.35)
        preproc_dwis(curr_study_path)
    os.chdir(project_dir)




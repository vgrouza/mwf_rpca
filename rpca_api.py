import os
import numpy as np
import nibabel as nib
from scipy.ndimage.morphology import binary_erosion


# create path variables
project_dir = '/export01/data/vgrouza/mbpko'
int_data_dir = os.path.join(project_dir, 'data/mri/interim')

# Add the directory containing Reconstructor class to the Python path
import sys
project_dir = '/export01/data/vgrouza/mbpko'
sys.path.append(project_dir)
from src import mwf_rpca


def compute_iewf(input_study_path: str):
    # set up paths and create directories, if needed; set the correct template for NNLS
    input_study_full_path = os.path.join(int_data_dir, input_study_path, 'mgre')
    output_study_full_path = os.path.join(int_data_dir, input_study_path, 'mgre/rpca')

    L1 = nib.load(os.path.join(output_study_full_path, input_study_path + '_rpca-L1.nii.gz'))
    L2 = nib.load(os.path.join(output_study_full_path, input_study_path + '_rpca-L2.nii.gz'))
    mask = nib.load(os.path.join(input_study_full_path, input_study_path + '_mgre-binary-mask.nii.gz'))

    iewf_to_save = np.true_divide(L1.get_fdata()[..., 0], (L2.get_fdata()[..., 0] + L1.get_fdata()[..., 0])) * mask.get_fdata()

    nib.save(img=nib.Nifti1Image(dataobj=iewf_to_save, affine=mask.affine, header=mask.header),
             filename=os.path.join(output_study_full_path, input_study_path + '_rpca-iewf.nii.gz'))
    return None


def rpca_mwf(input_study_path: str):
    # set up paths and create directories, if needed; set the correct template for NNLS
    input_study_full_path = os.path.join(int_data_dir, input_study_path, 'mgre')
    output_study_full_path = os.path.join(int_data_dir, input_study_path, 'mgre/rpca')
    os.makedirs(output_study_full_path, exist_ok=True)

    # instantiate the paths and load the images
    mgre_img_path = os.path.join(input_study_full_path, input_study_path + '_mgre-mag-preproc.nii.gz')
    mask_img_path = os.path.join(input_study_full_path, input_study_path + '_mgre-binary-mask.nii.gz')
    mgre_img = nib.load(mgre_img_path)
    mask_img = nib.load(mask_img_path)

    # mask away all background
    mgre_data = mgre_img.get_fdata() * np.repeat(mask_img.get_fdata()[:, :, :, np.newaxis], 24, axis=3)

    # instantiate the TE vector
    te_vec = np.arange(start=2, stop=50, step=2) * 1e-3

    # summon the rPCA
    rpca = mwf_rpca.RobustPCA()
    rpca.fit(mgre_data=mgre_data, binary_mask=mask_img.get_fdata(), echo_times=te_vec)
    x_tensor = rpca.X

    # explicitly define components
    L1_to_save = x_tensor[:, :, :, :, 0]
    L2_to_save = x_tensor[:, :, :, :, 1]
    S_to_save = x_tensor[:, :, :, :, 2]

    # compute MWF
    np.seterr(divide='ignore', invalid='ignore')
    mwf_to_save = np.true_divide(L2_to_save[..., 0], (L2_to_save[..., 0] + L1_to_save[..., 0]))
    mwf_to_save = mwf_to_save * binary_erosion(mask_img.get_fdata())

    # save niftis
    nib.save(img=nib.Nifti1Image(dataobj=mwf_to_save, affine=mask_img.affine, header=mask_img.header),
             filename=os.path.join(output_study_full_path, input_study_path + '_rpca-mwf.nii.gz'))
    nib.save(img=nib.Nifti1Image(dataobj=L1_to_save, affine=mgre_img.affine, header=mgre_img.header),
             filename=os.path.join(output_study_full_path, input_study_path + '_rpca-L1.nii.gz'))
    nib.save(img=nib.Nifti1Image(dataobj=L2_to_save, affine=mgre_img.affine, header=mgre_img.header),
             filename=os.path.join(output_study_full_path, input_study_path + '_rpca-L2.nii.gz'))
    nib.save(img=nib.Nifti1Image(dataobj=S_to_save, affine=mgre_img.affine, header=mgre_img.header),
             filename=os.path.join(output_study_full_path, input_study_path + '_rpca-sparse.nii.gz'))



if __name__ == '__main__':
    os.chdir(int_data_dir)
    study_dirs = sorted([name for name in os.listdir(os.getcwd()) if os.path.isdir(os.path.join(os.getcwd(), name))])
    for curr_study_path in study_dirs:
        print(curr_study_path)
        #rpca_mwf(input_study_path=curr_study_path)
        compute_iewf(input_study_path=curr_study_path)
    os.chdir(project_dir)
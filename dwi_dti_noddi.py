import os
import numpy as np
import nibabel as nib
import json

from dipy.core.gradients import gradient_table
import dipy.reconst.dti as dti
import amico

# create path variables
project_dir = '/export01/data/vgrouza/mbpko'
int_data_dir = os.path.join(project_dir, 'data/mri/interim')

def fit_dti(input_study_path):
    """
    A simple function for fitting the diffusion tensor model using code practically identical to the DiPy tutorials.
    Please consult the following webpage:
    https://workshop.dipy.org/documentation/1.6.0./examples_built/reconst_dti/

    [1] Garyfallidis E, Brett M, Amirbekian B, Rokem A, van der Walt S, Descoteaux M, Nimmo-Smith I; Dipy Contributors.
    Dipy, a library for the analysis of diffusion MRI data. Front Neuroinform. 2014 Feb 21;8:8.
    doi: https://doi.org/10.3389/fninf.2014.00008
    """

    # set up paths and create directories, if needed
    input_study_full_path = os.path.join(int_data_dir, input_study_path, 'dwi')
    output_study_full_path = os.path.join(int_data_dir, input_study_path, 'dwi/dti')
    os.makedirs(output_study_full_path, exist_ok=True)

    # load the preprocessed images and metadata
    dw_images = nib.load(os.path.join(input_study_full_path, input_study_path + '_dwi-preproc.nii.gz'))
    binary_mask = nib.load(os.path.join(input_study_full_path, input_study_path + '_dwi-binary-mask.nii.gz'))
    b_vals = np.loadtxt(os.path.join(input_study_full_path, input_study_path + '_dwi.bvals'))
    b_vecs = np.loadtxt(os.path.join(input_study_full_path, input_study_path + '_dwi.bvecs'))

    with open(os.path.join(input_study_full_path, input_study_path + '_dwi-preproc.json'), 'r') as fid:
        json_sidecar = json.load(fid)

    # instantiate the gradient table object and fit the diffusion tensor
    grad_tab = gradient_table(b_vals, b_vecs,
                              big_delta=json_sidecar['BigDelta'],
                              small_delta=json_sidecar['SmallDelta'])
    dti_model = dti.TensorModel(grad_tab)
    dti_fit = dti_model.fit(dw_images.get_fdata(), mask=binary_mask.get_fdata().astype(bool))

    # get diffusion tensor parameter maps
    dti_FA = dti_fit.fa  # fractional anistropy
    dti_MD = dti_fit.md  # mean diffusivity
    dti_AD = dti_fit.ad  # axial diffusivity
    dti_RD = dti_fit.rd  # radial diffusivity
    dti_FA[np.isnan(dti_FA) | np.isinf(dti_FA)] = 0.0
    dti_FA_RGB = (255 * dti.color_fa(dti_FA, dti_fit.evecs)).astype(np.uint8)
    dti_evec1 = dti_fit.evecs[:, :, :, :, 0]

    # compute the angle between the B0 field and the principal diffusion direction (first eigenvector)
    b0_vec = np.array(json_sidecar['B0Direction'])

    #TODO: rescale this calculation so it's +/- 90 degrees
    dot_product = np.clip(np.einsum('ijkl,l->ijk', dti_evec1, b0_vec), -1.0, 1.0)
    b0_angle_map = np.degrees(np.arccos(dot_product))
    b0_angle_map = np.where(b0_angle_map > 90, 180 - b0_angle_map, b0_angle_map)

    # make nifti objects and save - could put this in a helper function but whatever
    fa_image = nib.Nifti1Image(dataobj=dti_FA.astype(np.float32), affine=dw_images.affine)
    fa_image.header.set_xyzt_units(xyz='mm')
    nib.save(img=fa_image, filename=os.path.join(output_study_full_path, input_study_path + '_dti-fa.nii.gz'))

    md_image = nib.Nifti1Image(dataobj=dti_MD.astype(np.float32), affine=dw_images.affine)
    md_image.header.set_xyzt_units(xyz='mm')
    nib.save(img=md_image, filename=os.path.join(output_study_full_path, input_study_path + '_dti-md.nii.gz'))

    ad_image = nib.Nifti1Image(dataobj=dti_AD.astype(np.float32), affine=dw_images.affine)
    ad_image.header.set_xyzt_units(xyz='mm')
    nib.save(img=ad_image, filename=os.path.join(output_study_full_path, input_study_path + '_dti-ad.nii.gz'))

    rd_image = nib.Nifti1Image(dataobj=dti_RD.astype(np.float32), affine=dw_images.affine)
    rd_image.header.set_xyzt_units(xyz='mm')
    nib.save(img=rd_image, filename=os.path.join(output_study_full_path, input_study_path + '_dti-rd.nii.gz'))

    fa_rgb_image = nib.Nifti1Image(dataobj=dti_FA_RGB, affine=dw_images.affine)
    fa_rgb_image.header.set_xyzt_units(xyz='mm')
    fa_rgb_image.header.set_intent('vector')
    nib.save(img=fa_rgb_image, filename=os.path.join(output_study_full_path, input_study_path + '_dti-fa-rgb.nii.gz'))

    b0_angle_image = nib.Nifti1Image(dataobj=b0_angle_map.astype(np.float32) * binary_mask.get_fdata().astype(bool),
                                     affine=dw_images.affine)
    b0_angle_image.header.set_xyzt_units(xyz='mm')
    nib.save(img=b0_angle_image, filename=os.path.join(output_study_full_path, input_study_path + '_dti-b0angle.nii.gz'))
    return None


def fit_noddi(input_study_path):
    """
    A simple function to fit the NODDI model using code provided from AMICO tutorials
    Please condult the following webpage: https://github.com/daducci/AMICO/wiki/NODDI
    """
    # set up paths and create directories, if needed
    input_study_full_path = os.path.join(int_data_dir, input_study_path, 'dwi')
    output_study_full_path = os.path.join(int_data_dir, input_study_path, 'dwi/noddi')
    os.makedirs(output_study_full_path, exist_ok=True)

    # load the b-values
    b_vals = np.loadtxt(os.path.join(input_study_full_path, input_study_path + '_dwi.bvals'))
    b_vecs = np.loadtxt(os.path.join(input_study_full_path, input_study_path + '_dwi.bvecs'))

    # save the transposed b-values and b-vectors to new text files according to FSL conventions
    np.savetxt(os.path.join(output_study_full_path, input_study_path + '_dwi.bvals.fsl'), b_vals.T, fmt='%.8f')
    np.savetxt(os.path.join(output_study_full_path, input_study_path + '_dwi.bvecs.fsl'), b_vecs.T, fmt='%.8f')

    # set up the amico object
    amico.core.setup()
    amico.util.fsl2scheme(bvalsFilename=os.path.join(output_study_full_path, input_study_path + '_dwi.bvals.fsl'),
                          bvecsFilename=os.path.join(output_study_full_path, input_study_path + '_dwi.bvecs.fsl'),
                          schemeFilename=os.path.join(output_study_full_path, input_study_path + '_noddi.scheme'))
    ae = amico.Evaluation(study_path=input_study_full_path,
                          subject=input_study_full_path,
                          output_path=output_study_full_path)
    # load data
    ae.load_data(dwi_filename=os.path.join(input_study_full_path, input_study_path + '_dwi-preproc.nii.gz'),
                 scheme_filename=os.path.join(output_study_full_path, input_study_path + '_noddi.scheme'),
                 mask_filename=os.path.join(input_study_full_path, input_study_path + '_dwi-binary-mask.nii.gz'),
                 b0_thr=100)

    # set model parameters
    ae.set_model("NODDI")
    ae.model.set(
        dPar=0.9e-3,  # mm^2/s
        dIso=2.0e-3,  # mm^2/s
        IC_VFs=np.linspace(0.01, 0.99, 72),
        IC_ODs=np.hstack((np.array([0.03, 0.06]), np.linspace(0.01, 0.99, 24))),
        isExvivo=True  # include dot compartment
    )
    ae.set_config('doComputeNRMSE', True)

    # generate and load response functions for each compartment
    ae.generate_kernels(regenerate=True)
    ae.load_kernels()

    # carry out the fitting procedure and save results
    ae.fit()
    ae.save_results()

    # load the NODDI outputs and compute AWF
    ndi = nib.load(os.path.join(output_study_full_path, 'fit_NDI.nii.gz'))
    fwf = nib.load(os.path.join(output_study_full_path, 'fit_FWF.nii.gz'))
    dot = nib.load(os.path.join(output_study_full_path, 'fit_dot.nii.gz'))
    awf = ndi.get_fdata() * (1.0 - fwf.get_fdata()) * (1.0 - dot.get_fdata())
    awf_image = nib.Nifti1Image(dataobj=awf.astype(np.float32), affine=ndi.affine)
    awf_image.header.set_xyzt_units(xyz='mm')
    nib.save(img=awf_image, filename=os.path.join(output_study_full_path, input_study_path + '_noddi-awf.nii.gz'))
    return None


if __name__ == '__main__':
    os.chdir(int_data_dir)
    study_dirs = sorted([name for name in os.listdir(os.getcwd()) if os.path.isdir(os.path.join(os.getcwd(), name))])
    for curr_study_path in study_dirs:
        print(curr_study_path)
        fit_dti(input_study_path=curr_study_path)
        fit_noddi(curr_study_path)
    os.chdir(project_dir)

import os
import numpy as np
import ants
import shutil
from skimage import exposure
import matplotlib.pyplot as plt

# create path variables
project_dir = '/export01/data/vgrouza/mbpko'
int_data_dir = os.path.join(project_dir, 'data/mri/interim')
proc_data_dir = os.path.join(project_dir, 'data/mri/processed')

def rigid_dwi2mgre(input_study_path: str):
    # set up paths and create directories, if needed
    input_study_dwi_path = os.path.join(int_data_dir, input_study_path, 'dwi')
    input_study_mgre_path = os.path.join(int_data_dir, input_study_path, 'mgre')
    output_study_full_path = os.path.join(input_study_dwi_path, 'tx_to_mgre')
    os.makedirs(output_study_full_path, exist_ok=True)

    # load images
    fixed_image = ants.get_average_of_timeseries(
        ants.image_read(os.path.join(input_study_mgre_path, input_study_path + '_mgre-mag-preproc.nii.gz')))
    moving_image = ants.image_read(os.path.join(input_study_dwi_path, input_study_path + '_dwi-b0-average.nii.gz'))

    # load the masks
    fixed_mask = ants.image_read(os.path.join(input_study_mgre_path, input_study_path + '_mgre-binary-mask.nii.gz'))
    moving_mask = ants.image_read(os.path.join(input_study_dwi_path, input_study_path + '_dwi-binary-mask.nii.gz'))

    # compute the transform
    tx_rigid = ants.registration(fixed=fixed_image,
                                 mask=fixed_mask,
                                 moving=moving_image,
                                 moving_mask=moving_mask,
                                 type_of_transform='Rigid',
                                 aff_metric='mattes',
                                 aff_iterations=(1000, 1000, 200),
                                 aff_shrink_factors=(4, 2, 1),
                                 aff_smoothing_sigmas=(4, 2, 1)
                                 )
    # save the transform
    shutil.copy2(src=tx_rigid['fwdtransforms'][0],
                 dst=os.path.join(output_study_full_path, 'tx_rigid.mat'))

    # save the warped image
    ants.image_write(image=tx_rigid['warpedmovout'],
                     filename=os.path.join(output_study_full_path, input_study_path + '_rigid_to_mgre.nii.gz'))


def syn_mese2mgre(input_study_path: str):
    # set up paths and create directories, if needed
    input_study_mese_path = os.path.join(int_data_dir, input_study_path, 'mese')
    input_study_mgre_path = os.path.join(int_data_dir, input_study_path, 'mgre')
    output_study_full_path = os.path.join(input_study_mese_path, 'tx_to_mgre')
    os.makedirs(output_study_full_path, exist_ok=True)

    # load images
    fixed_image = ants.get_average_of_timeseries(
        ants.image_read(os.path.join(input_study_mgre_path, input_study_path + '_mgre-mag-preproc.nii.gz')))
    moving_image = ants.get_average_of_timeseries(
        ants.image_read(os.path.join(input_study_mese_path, input_study_path + '_mese-preproc.nii.gz')))

    # load the masks
    fixed_mask = ants.image_read(os.path.join(input_study_mgre_path, input_study_path + '_mgre-binary-mask.nii.gz'))
    moving_mask = ants.image_read(os.path.join(input_study_mese_path, input_study_path + '_mese-binary-mask.nii.gz'))

    # compute the transform
    tx_syn = ants.registration(fixed=fixed_image,
                               moving=moving_image,
                               #fixed_mask=fixed_mask,
                               #moving_mask=moving_mask,
                               type_of_transform='SyNOnly',
                               verbose=True,
                               reg_iterations=(200, 200, 200),
                               smoothing_in_vox=True,
                               #initial_transform='Identity',
                               write_composite_transform=True,
                               syn_metric='mattes')
    # save the transform
    shutil.copy2(src=tx_syn['fwdtransforms'][0],
                 dst=os.path.join(output_study_full_path, 'tx_syn_warp.nii.gz'))
    shutil.copy2(src=tx_syn['fwdtransforms'][1],
                 dst=os.path.join(output_study_full_path, 'tx_rigid.mat'))
    shutil.copy2(src=tx_syn['invtransforms'][1],
                 dst=os.path.join(output_study_full_path, 'tx_syn_inv_warp.nii.gz'))

    # save the warped image
    ants.image_write(image=tx_syn['warpedmovout'],
                     filename=os.path.join(output_study_full_path, input_study_path + '_warp_to_mgre.nii.gz'))


def composite_mgre2template(input_study_path: str):
    # set up paths and create directories, if needed
    input_study_mgre_path = os.path.join(int_data_dir, input_study_path, 'mgre')
    template_path = os.path.join(project_dir, 'data/mri/priors')
    output_study_full_path = os.path.join(input_study_mgre_path, 'tx2template')
    os.makedirs(output_study_full_path, exist_ok=True)

    # load the images
    fixed_image = ants.image_read(os.path.join(template_path, 'dsurqe_template_masked.nii'))
    moving_image = ants.get_average_of_timeseries(
        ants.image_read(os.path.join(input_study_mgre_path, 'rpca', input_study_path + '_rpca-L1.nii.gz')))

    # load the masks
    fixed_mask = ants.get_mask(fixed_image)
    moving_mask = ants.image_read(os.path.join(input_study_mgre_path, input_study_path + '_mgre-binary-mask.nii.gz'))

    # enhance contrast of the mGRE image
    # rescale image [0, 1]

    moving_image_data = moving_image.numpy()
    moving_image_data = np.clip(moving_image_data, np.percentile(moving_image_data, 5), np.percentile(moving_image_data, 95))
    moving_image_data = (moving_image_data - np.min(moving_image_data)) / (np.max(moving_image_data) - np.min(moving_image_data))

    # determine kernel sizes in each dim relative to image shape
    kernel_size = (moving_image_data.shape[0] // 5,
                   moving_image_data.shape[1] // 5,
                   moving_image_data.shape[2] // 5)
    kernel_size = np.array(kernel_size)

    # set clip limit (higher limit --> more contrast; empirically determined to work well with this value)
    clip_limit = 0.1

    # perform contrast limited adaptive histogram equalization (CLAHE)
    moving_image_data = exposure.equalize_adapthist(moving_image_data,
                                                     kernel_size=kernel_size,
                                                     clip_limit=clip_limit) * moving_mask.numpy()

    # transform back into an ANTsImage object and perform N4 bias field correction
    moving_image = moving_image.new_image_like(moving_image_data)
    moving_image = ants.n4_bias_field_correction(moving_image, mask=moving_mask, spline_param=20)

    # carry out multi-step registration
    print('\t --> Rigid + Similarity + Affine Transforms...', end='')
    tx_rigid = ants.registration(fixed=fixed_image,
                                 moving=moving_image,
                                 #fixed_mask=fixed_mask,
                                 #moving_mask=moving_mask,
                                 type_of_transform='Rigid',
                                 verbose=False,
                                 aff_metric='mattes',
                                 aff_iterations=(1000, 1000, 200),
                                 aff_shrink_factors=(4, 2, 1),
                                 aff_smoothing_sigmas=(4, 2, 1))

    tx_simil = ants.registration(fixed=fixed_image,
                                 moving=tx_rigid['warpedmovout'],
                                 type_of_transform='Similarity',
                                 #fixed_mask=fixed_mask,
                                 #moving_mask=tx_rigid['warpedmovout'].get_mask(),
                                 verbose=False,
                                 aff_metric='mattes',
                                 aff_iterations=(1000, 1000, 200),
                                 aff_shrink_factors=(4, 2, 1),
                                 aff_smoothing_sigmas=(4, 2, 1))

    tx_affine = ants.registration(fixed=fixed_image,
                                  moving=tx_simil['warpedmovout'],
                                  type_of_transform='Affine',
                                  #fixed_mask=fixed_mask,
                                  #moving_mask=tx_simil['warpedmovout'].get_mask(),
                                  verbose=False,
                                  aff_metric='mattes',
                                  aff_iterations=(5000, 2500, 1000),
                                  aff_shrink_factors=(3, 2, 1),
                                  aff_smoothing_sigmas=(4, 2, 1), # 0.08, 0.04, 0
                                  smoothing_in_mm=False)
    print('[DONE]')
    print('\t --> SyN Transform...')
    tx_syn = ants.registration(fixed=fixed_image,
                               moving=tx_affine['warpedmovout'],
                               fixed_mask=fixed_mask,
                               type_of_transform='SyNOnly',
                               verbose=True,
                               reg_iterations=(200, 200, 200),
                               smoothing_in_vox=True,
                               initial_transform='Identity',
                               syn_metric='mattes')

    tx_syn2 = ants.registration(fixed=fixed_image,
                               moving=tx_affine['warpedmovout'],
                               fixed_mask=fixed_mask,
                               type_of_transform='SyNOnly',
                               verbose=True,
                               reg_iterations=(200, 200, 200),
                               smoothing_in_vox=True,
                               #initial_transform='Identity',
                               syn_metric='mattes')

    tx_syn3 = ants.registration(fixed=fixed_image,
                                moving=tx_affine['warpedmovout'],
                                fixed_mask=fixed_mask,
                                moving_mask=tx_affine['warpedmovout'].get_mask(),
                                type_of_transform='SyNOnly',
                                verbose=True,
                                reg_iterations=(200, 200, 200),
                                # smoothing_in_vox=True,
                                # initial_transform='Identity',
                                syn_metric='mattes')


    fixed_image.plot(overlay=tx_syn['warpedmovout'], overlay_cmap='hot', overlay_alpha=0.3, nslices=24, axis=0,
                     title=input_study_path + ' SyN to template')

    fixed_image.plot(overlay=tx_syn2['warpedmovout'], overlay_cmap='hot', overlay_alpha=0.3, nslices=24, axis=0,
                     title=input_study_path + ' SyN to template')

    # save the warped, contrast-enhanced image
    ants.image_write(image=tx_syn3['warpedmovout'] * fixed_mask,
                     filename=os.path.join(output_study_full_path, input_study_path + '_mgre2template3.nii.gz'))

    '''
    # make some plots    
    
    fig1 = plt.gcf()
    plt.close(fig1)
    fixed_image.plot(overlay=tx_affine['warpedmovout'], overlay_cmap='hot', overlay_alpha=0.6, nslices=12, axis=2, title=input_study_path + ' SyN to template')
    fig2 = plt.gcf()
    plt.close(fig2)
    fixed_image.plot(overlay=tx_affine['warpedmovout'], overlay_cmap='hot', overlay_alpha=0.6, nslices=24, axis=1, title=input_study_path + ' SyN to template')
    fig3 = plt.gcf()
    plt.close(fig3)

    # save the plots
    fig1.savefig(os.path.join(output_study_full_path, input_study_path + '_regfig1.png'))
    fig2.savefig(os.path.join(output_study_full_path, input_study_path + '_regfig2.png'))
    fig3.savefig(os.path.join(output_study_full_path, input_study_path + '_regfig3.png'))
    '''

    # save the warped, contrast-enhanced image
    ants.image_write(image=tx_syn['warpedmovout'] * fixed_mask,
                     filename=os.path.join(output_study_full_path, input_study_path + '_mgre2template.nii.gz'))

    # save
    print('Saving tx files...', end='')
    shutil.copy2(tx_rigid['fwdtransforms'][0], os.path.join(output_study_full_path, input_study_path + '_tx-rigid-mgre2template.mat'))
    shutil.copy2(tx_simil['fwdtransforms'][0], os.path.join(output_study_full_path, input_study_path + '_tx-simil-mgre2template.mat'))
    shutil.copy2(tx_affine['fwdtransforms'][0], os.path.join(output_study_full_path, input_study_path + '_tx-affine-mgre2template.mat'))
    shutil.copy2(tx_syn['fwdtransforms'][0], os.path.join(output_study_full_path, input_study_path + '_tx-syn-warp-mgre2template.nii.gz'))
    shutil.copy2(tx_syn['fwdtransforms'][1],
                 os.path.join(output_study_full_path, input_study_path + '_tx-syn-affine-mgre2template.mat'))
    shutil.copy2(tx_syn['invtransforms'][1],
                 os.path.join(output_study_full_path, input_study_path + '_tx-syn-invwarp-mgre2template.nii.gz'))

    # clear /tmp folder to prevent memory overflow
    os.chdir('/tmp')
    os.system('rm -f *.nii.gz')
    os.system('rm -f *.mat')
    os.chdir(int_data_dir)
    print('[DONE]')
    return None


def apply_inv_tx(input_study_path: str):
    # set up paths and create directories, if needed
    input_study_mgre_path = os.path.join(int_data_dir, input_study_path, 'mgre')
    template_path = os.path.join(project_dir, 'data/mri/priors')
    output_study_full_path = os.path.join(input_study_mgre_path, 'tx_to_template')
    os.makedirs(output_study_full_path, exist_ok=True)

    # load the images
    fixed_image = ants.image_read(os.path.join(template_path, 'dsurqe_template_masked.nii'))
    moving_image = ants.get_average_of_timeseries(
        ants.image_read(os.path.join(input_study_mgre_path, input_study_path + '_mgre-mag-preproc.nii.gz')))

    # load the masks
    fixed_mask = ants.get_mask(fixed_image)

    syn_warp = ants.image_read(os.path.join(output_study_full_path, 'tx_syn_warp.nii.gz'))

    # Apply the forward warp in reverse
    inverse_syn_image = ants.apply_transforms(
        fixed=moving_image,  # Swapping the fixed and moving image
        moving=fixed_image,  # Applying the reverse transform
        transformlist=[os.path.join(output_study_full_path, 'tx_syn_warp.nii.gz')],  # The forward SyN transform
        whichtoinvert=[True],  # Invert the SyN transform
        interpolator='bSpline'  # Optional: Set appropriate interpolator
    )


    # load the warp
    syn_tx_img = ants.transform_from_displacement_field(
        ants.image_read(os.path.join(output_study_full_path, 'tx_syn_warp.nii.gz')))

    # apply inverse warps
    template_to_mgre = ants.apply_transforms(fixed=moving_image,
                                             moving=fixed_image,
                                             interpolator='bSpline',
                                             transformlist=[os.path.join(output_study_full_path, 'tx_rigid.mat'),
                                                            os.path.join(output_study_full_path, 'tx_simil.mat'),
                                                            os.path.join(output_study_full_path, 'tx_affine.mat'),
                                                            os.path.join(output_study_full_path, 'tx_syn_warp.nii.gz')],
                                             whichtoinvert=[1, 1, 1, 1])



def apply_inv_tx_template_to_mese(input_study_path: str):
    # set up paths and create directories, if needed
    input_study_mgre_path = os.path.join(int_data_dir, input_study_path, 'mgre')
    input_study_mese_path = os.path.join(int_data_dir, input_study_path, 'mese')
    template_path = os.path.join(project_dir, 'data/mri/priors')
    output_study_full_path = os.path.join(input_study_mese_path, 'tx_to_template')
    output_study_full_path2 = os.path.join(input_study_mese_path, 'tx_to_mgre')
    os.makedirs(output_study_full_path, exist_ok=True)

    # load the images
    fixed_image = ants.image_read(os.path.join(template_path, 'abi2dsurqe_atlas.nii'))
    moving_image = ants.get_average_of_timeseries(
        ants.image_read(os.path.join(input_study_mgre_path, input_study_path + '_mgre-mag-preproc.nii.gz')))

    # apply inverse warps
    template_to_mgre = ants.apply_transforms(fixed=moving_image,
                                             moving=fixed_image,
                                             interpolator='genericLabel',
                                             transformlist=[os.path.join(output_study_full_path, 'tx_rigid.mat'),
                                                            os.path.join(output_study_full_path, 'tx_simil.mat'),
                                                            os.path.join(output_study_full_path, 'tx_affine.mat'),
                                                            os.path.join(output_study_full_path, 'tx_syn_inv_warp.h5')],
                                             whichtoinvert=[1, 1, 1, 0])

    # load the images for the second warp
    fixed_image = ants.get_average_of_timeseries(ants.image_read(os.path.join(input_study_mese_path, input_study_path + '_mese-preproc.nii.gz')))


    # apply inverse warps
    template_to_mese = ants.apply_transforms(fixed=fixed_image,
                                             moving=template_to_mgre,
                                             interpolator='genericLabel',
                                             transformlist=[os.path.join(output_study_full_path2, 'tx_rigid.mat'),
                                                            os.path.join(output_study_full_path2, 'tx_syn_inv_warp.nii.gz')],
                                             whichtoinvert=[1, 0])

    ants.image_write(image=template_to_mese, filename=os.path.join(output_study_full_path2, input_study_path + '_template-labels.nii.gz'))


def apply_fwd_tx(input_study_path: str):
    # set up paths and create directories, if needed
    input_study_mgre_path = os.path.join(int_data_dir, input_study_path, 'mgre')
    template_path = os.path.join(project_dir, 'data/mri/priors')
    output_study_full_path = os.path.join(input_study_mgre_path, 'tx_to_template')
    os.makedirs(output_study_full_path, exist_ok=True)

    # load the images
    fixed_image = ants.image_read(os.path.join(template_path, 'dsurqe_template_masked.nii'))
    moving_image = ants.get_average_of_timeseries(
        ants.image_read(os.path.join(input_study_mgre_path, input_study_path + '_mgre-mag-preproc.nii.gz')))

    # load the masks
    fixed_mask = ants.get_mask(fixed_image)

    # account for subjects which needed additional registration steps
    additional_reg_subj = ['sub-23_strain-ki', 'sub-21_strain-ki']
    if input_study_path in additional_reg_subj:
        syn_tx_img1 = ants.transform_from_displacement_field(
            ants.image_read(os.path.join(output_study_full_path, 'tx_syn1_warp.nii.gz')))
        syn_tx_img2 = ants.transform_from_displacement_field(
            ants.image_read(os.path.join(output_study_full_path, 'tx_syn2_warp.nii.gz')))
        warped_img = ants.apply_transforms(fixed=fixed_image,
                                           moving=moving_image,
                                           transformlist=[os.path.join(output_study_full_path, 'tx_syn1_affine.mat'),
                                                          os.path.join(output_study_full_path, 'tx_affine.mat'),
                                                          os.path.join(output_study_full_path, 'tx_simil.mat'),
                                                          os.path.join(output_study_full_path, 'tx_rigid.mat')])
        warped_img = ants.apply_ants_transform_to_image(syn_tx_img1, warped_img, fixed_image)
        warped_img = ants.apply_ants_transform_to_image(syn_tx_img2, warped_img, fixed_image) * fixed_mask

    else:
        # apply linear transforms
        warped_img = ants.apply_transforms(fixed=fixed_image,
                                           moving=moving_image,
                                           transformlist=[os.path.join(output_study_full_path, 'tx_affine.mat'),
                                                          os.path.join(output_study_full_path, 'tx_simil.mat'),
                                                          os.path.join(output_study_full_path, 'tx_rigid.mat')])
        # apply nonlinear transforms
        syn_tx_img = ants.transform_from_displacement_field(
            ants.image_read(os.path.join(output_study_full_path, 'tx_syn_warp.nii.gz')))

        warped_img = ants.apply_ants_transform_to_image(transform=syn_tx_img,
                                                        image=warped_img,
                                                        reference=fixed_image,
                                                        interpolation='bspline') * fixed_mask

    ants.image_write(image=warped_img,
                     filename=os.path.join(output_study_full_path, input_study_path + '_mgre-mag-warped.nii.gz'))
    return None



def apply_fwd_tx_mese(input_study_path: str):
    # set up paths and create directories, if needed
    input_study_mese_path = os.path.join(int_data_dir, input_study_path, 'mese')
    input_study_mgre_path = os.path.join(int_data_dir, input_study_path, 'mgre')
    template_path = os.path.join(project_dir, 'data/mri/priors')
    output_study_full_path = os.path.join(proc_data_dir, input_study_path, 'mese')
    os.makedirs(output_study_full_path, exist_ok=True)

    # first warp mese from mese space to mgre space
    mese_img = ants.image_read(os.path.join(input_study_mese_path, input_study_path + '_mese-preproc.nii.gz'))
    mese_img = ants.get_average_of_timeseries(mese_img)
    mgre_img = ants.image_read(os.path.join(input_study_mgre_path, input_study_path + '_mgre-mag-preproc.nii.gz'))
    mgre_img = ants.get_average_of_timeseries(mgre_img)

    alpha_img = ants.image_read(os.path.join(input_study_mese_path, 'decaes_lcurve2', input_study_path + '_decaes-alpha.nii.gz'))
    gmT2sp_img = ants.image_read(os.path.join(input_study_mese_path, 'decaes_lcurve2', input_study_path + '_decaes-gmT2(sp).nii.gz'))
    gmT2mp_img = ants.image_read(os.path.join(input_study_mese_path, 'decaes_lcurve2', input_study_path + '_decaes-gmT2(mp).nii.gz'))
    mwf_img = ants.image_read(os.path.join(input_study_mese_path, 'decaes_lcurve2', input_study_path + '_decaes-mwf.nii.gz'))
    iewf_img = ants.image_read(os.path.join(input_study_mese_path, 'decaes_lcurve2', input_study_path + '_decaes-iewf.nii.gz'))

    img_list = [mese_img, alpha_img, gmT2sp_img, gmT2mp_img, mwf_img, iewf_img]
    suffix_list = ['_mese-preproc-flat.nii.gz', '_decaes-alpha.nii.gz', '_decaes-gmT2(sp).nii.gz', '_decaes-gmT2(mp).nii.gz', '_decaes-mwf.nii.gz', '_decaes-iewf.nii.gz']

    template_img = ants.image_read(os.path.join(template_path, 'dsurqe_template_masked.nii'))
    template_mask = ants.image_read(os.path.join(template_path, 'dsurqe_template_brain_mask.nii.gz'))

    # cycle through images
    for i in range(0, len(img_list)):
        warped_img = ants.apply_transforms(fixed=mgre_img,
                                           moving=img_list[i],
                                           transformlist=[
                                               os.path.join(input_study_mese_path, 'tx_to_mgre', 'tx_syn_warp.nii.gz'),
                                               os.path.join(input_study_mese_path, 'tx_to_mgre', 'tx_rigid.mat')
                                               ],
                                           interpolator='bSpline')
        # then warp to template space
        warped_img = ants.apply_transforms(fixed=template_img,
                                           moving=warped_img,
                                           transformlist=[
                                               os.path.join(input_study_mgre_path, 'tx_to_template', 'tx_syn_warp.nii.gz'),
                                               os.path.join(input_study_mgre_path, 'tx_to_template', 'tx_affine.mat'),
                                               os.path.join(input_study_mgre_path, 'tx_to_template', 'tx_simil.mat'),
                                               os.path.join(input_study_mgre_path, 'tx_to_template', 'tx_rigid.mat')
                                               ],
                                           interpolator='bSpline')
        # then mask and save
        warped_img *= template_mask
        warped_img[warped_img < 0] = 0.0
        ants.image_write(image=warped_img, filename=os.path.join(output_study_full_path, input_study_path + suffix_list[i]))

    return None


def apply_fwd_tx_mgre(input_study_path: str):
    # set up paths and create directories, if needed
    input_study_mgre_path = os.path.join(int_data_dir, input_study_path, 'mgre')
    template_path = os.path.join(project_dir, 'data/mri/priors')
    output_study_full_path = os.path.join(proc_data_dir, input_study_path, 'mgre')
    os.makedirs(output_study_full_path, exist_ok=True)

    # first warp mese from mese space to mgre space
    mgre_img = ants.image_read(os.path.join(input_study_mgre_path, input_study_path + '_mgre-mag-preproc.nii.gz'))
    mgre_img = ants.get_average_of_timeseries(mgre_img)

    mwf_img = ants.image_read(os.path.join(input_study_mgre_path, 'rpca', input_study_path + '_rpca-mwf.nii.gz'))
    iewf_img = ants.image_read(os.path.join(input_study_mgre_path, 'rpca', input_study_path + '_rpca-iewf.nii.gz'))
    L1_t2star_img = ants.image_read(os.path.join(input_study_mgre_path, 'rpca', input_study_path + '_rpca-L1-t2star-map.nii.gz'))
    L2_t2star_img = ants.image_read(os.path.join(input_study_mgre_path, 'rpca', input_study_path + '_rpca-L2-t2star-map.nii.gz'))
    t2star_img = ants.image_read(os.path.join(input_study_mgre_path, 'rpca', input_study_path + '_mgre-mag-t2star-map.nii.gz'))


    img_list = [mgre_img, mwf_img, iewf_img, L1_t2star_img, L2_t2star_img, t2star_img]
    suffix_list = ['_mgre-mag-preproc-flat.nii.gz', '_rpca-mwf.nii.gz', '_rpca-iewf.nii.gz', '_rpca-L1-t2star-map.nii.gz', '_rpca-L2-t2star-map.nii.gz', '_mgre-mag-t2star-map.nii.gz']

    template_img = ants.image_read(os.path.join(template_path, 'dsurqe_template_masked.nii'))
    template_mask = ants.image_read(os.path.join(template_path, 'dsurqe_template_brain_mask.nii.gz'))

    # cycle through images
    for i in range(0, len(img_list)):
        # then warp to template space
        warped_img = ants.apply_transforms(fixed=template_img,
                                           moving=img_list[i],
                                           transformlist=[
                                               os.path.join(input_study_mgre_path, 'tx_to_template', 'tx_syn_warp.nii.gz'),
                                               os.path.join(input_study_mgre_path, 'tx_to_template', 'tx_affine.mat'),
                                               os.path.join(input_study_mgre_path, 'tx_to_template', 'tx_simil.mat'),
                                               os.path.join(input_study_mgre_path, 'tx_to_template', 'tx_rigid.mat')
                                               ],
                                           interpolator='bSpline')
        # then mask and save
        warped_img *= template_mask
        warped_img[warped_img < 0] = 0.0
        ants.image_write(image=warped_img, filename=os.path.join(output_study_full_path, input_study_path + suffix_list[i]))

    return None


def apply_fwd_tx_dwi(input_study_path: str):
    # set up paths and create directories, if needed
    input_study_dwi_path = os.path.join(int_data_dir, input_study_path, 'dwi')
    input_study_mgre_path = os.path.join(int_data_dir, input_study_path, 'mgre')
    template_path = os.path.join(project_dir, 'data/mri/priors')
    output_study_full_path = os.path.join(proc_data_dir, input_study_path, 'dwi')
    os.makedirs(output_study_full_path, exist_ok=True)

    # first warp mese from mese space to mgre space
    dwi_img = ants.image_read(os.path.join(input_study_dwi_path, input_study_path + '_dwi-b0-average.nii.gz'))
    mgre_img = ants.image_read(os.path.join(input_study_mgre_path, input_study_path + '_mgre-mag-preproc.nii.gz'))
    mgre_img = ants.get_average_of_timeseries(mgre_img)

    b0angle_img = ants.image_read(os.path.join(input_study_dwi_path, 'dti', input_study_path + '_dti-b0angle.nii.gz'))
    fa_img = ants.image_read(os.path.join(input_study_dwi_path, 'dti', input_study_path + '_dti-fa.nii.gz'))
    awf_img = ants.image_read(os.path.join(input_study_dwi_path, 'noddi', input_study_path + '_noddi-awf.nii.gz'))

    img_list = [dwi_img, b0angle_img, fa_img, awf_img]
    suffix_list = ['_dwi-b0-average.nii.gz', '_dti-b0angle.nii.gz', '_dti-fa.nii.gz', '_noddi-awf.nii.gz']

    template_img = ants.image_read(os.path.join(template_path, 'dsurqe_template_masked.nii'))
    template_mask = ants.image_read(os.path.join(template_path, 'dsurqe_template_brain_mask.nii.gz'))

    # cycle through images
    for i in range(0, len(img_list)):

        warped_img = ants.apply_transforms(fixed=mgre_img,
                                           moving=img_list[i],
                                           transformlist=[
                                               os.path.join(input_study_dwi_path, 'tx_to_mgre', 'tx_rigid.mat')
                                               ],
                                           imagetype=0,
                                           interpolator='bSpline'
                                           )
        # then warp to template space
        warped_img = ants.apply_transforms(fixed=template_img,
                                           moving=warped_img,
                                           transformlist=[
                                               os.path.join(input_study_mgre_path, 'tx_to_template', 'tx_syn_warp.nii.gz'),
                                               os.path.join(input_study_mgre_path, 'tx_to_template', 'tx_affine.mat'),
                                               os.path.join(input_study_mgre_path, 'tx_to_template', 'tx_simil.mat'),
                                               os.path.join(input_study_mgre_path, 'tx_to_template', 'tx_rigid.mat')
                                               ],
                                           imagetype=0,
                                           interpolator='bSpline'
                                           )
        # then mask and save
        warped_img *= template_mask
        ants.image_write(image=warped_img, filename=os.path.join(output_study_full_path, input_study_path + suffix_list[i]))

    return None


'''
From DWI:
_dti-fa.nii.gz
_dti-b0angle.nii.gz
_dwi-b0-average.nii.gz

From MESE:
_mese-preproc.nii.gz
_decaes-alpha.nii.gz
_decaes-gmT2(mp).nii.gz
_decaes-gmT2(sp).nii.gz
_decaes-iewf.nii.gz
_decaes-mwf.nii.gz

From MGRE
_mgre-mag-preproc.nii.gz
_rpca-L1-t2star-map.nii.gz
_rpca-L2-t2star-map.nii.gz
_rpca-mwf.nii.gz
_rpca-iewf.nii.gz


'''



if __name__ == '__main__':
    os.chdir(int_data_dir)
    study_dirs = sorted([name for name in os.listdir(os.getcwd()) if os.path.isdir(os.path.join(os.getcwd(), name))])
    for curr_study_path in study_dirs:
        print(curr_study_path)
        #rigid_dwi2mgre(input_study_path=curr_study_path)
        #syn_mese2mgre(input_study_path=curr_study_path)
        #composite_mgre2template(input_study_path=curr_study_path)
        #apply_fwd_tx(curr_study_path)
        #apply_fwd_tx_mese(curr_study_path)
        apply_fwd_tx_mgre(curr_study_path)
        #apply_fwd_tx_dwi(curr_study_path)
    os.chdir(project_dir)

import os
import numpy as np
import nibabel as nib
import h5py


# create path variables
project_dir = '/export01/data/vgrouza/mbpko'
int_data_dir = os.path.join(project_dir, 'data/mri/interim')


def nnls_mgre(input_study_path: str, reg: str):

    # set up paths and create directories, if needed; set the correct template for NNLS
    input_study_full_path = os.path.join(int_data_dir, input_study_path, 'mgre')
    if reg == 'chi2':
        output_study_full_path = os.path.join(int_data_dir, input_study_path, 'mgre/decaes_chi2')
        os.makedirs(output_study_full_path, exist_ok=True)
        settings_template_path = os.path.join(project_dir, 'config/settings_template_mgre_chi2.txt')

    elif reg == 'lcurve':
        output_study_full_path = os.path.join(int_data_dir, input_study_path, 'mgre/decaes_lcurve')
        os.makedirs(output_study_full_path, exist_ok=True)
        settings_template_path = os.path.join(project_dir, 'config/settings_template_mgre_lcurve.txt')


    else:
        raise NotImplemented

    # instantiate the paths and load the images
    mgre_img_path = os.path.join(input_study_full_path, input_study_path + '_mgre-mag-preproc.nii.gz')
    mask_img_path = os.path.join(input_study_full_path, input_study_path + '_mgre-binary-mask.nii.gz')

    # load and modify the template
    with open(settings_template_path, 'r') as file:
        lines = file.readlines()

    lines[0] = mgre_img_path + '\n'
    lines[2] = output_study_full_path + '\n'
    lines[4] = mask_img_path + '\n'

    with open(settings_template_path, 'w') as file:
        file.writelines(lines)

    # execute DECAES calculations command in python
    os.system(f"julia --project=@decaes --threads=auto -e 'using DECAES; main()' \
                @{settings_template_path}")

    # load the resultant data
    decaes_outputs = os.listdir(output_study_full_path)
    dist_file = h5py.File(
        os.path.join(output_study_full_path, [file for file in decaes_outputs if 't2dist.mat' in file][0]), 'r')
    maps_file = h5py.File(
        os.path.join(output_study_full_path, [file for file in decaes_outputs if 't2maps.mat' in file][0]), 'r')
    parts_file = h5py.File(
        os.path.join(output_study_full_path, [file for file in decaes_outputs if 't2parts.mat' in file][0]), 'r')

    # get the t2 times and NNLS distribution
    t2_times = np.array(maps_file['t2times'])
    t2_dist = np.transpose(np.array(dist_file['dist']))
    t2_dist[np.isnan(t2_dist) | np.isinf(t2_dist)] = 0.0

    # get the MWF, IEWF, gmT2(sp), and gmT2(mp)
    mwf = np.transpose(np.array(parts_file['sfr']))
    mwf[np.isnan(mwf) | np.isinf(mwf)] = 0.0

    iewf = np.transpose(np.array(parts_file['mfr']))
    iewf[np.isnan(mwf) | np.isinf(mwf)] = 0.0

    small_gm = np.transpose(np.array(parts_file['sgm']))
    small_gm[np.isnan(small_gm) | np.isinf(small_gm)] = 0.0

    med_gm = np.transpose(np.array(parts_file['mgm']))
    med_gm[np.isnan(med_gm) | np.isinf(med_gm)] = 0.0

    # get the residuals and snr - maybe we need the chi2 factor maps ('chi2factor') and reg param maps ('mu') later
    resnorm = np.transpose(np.array(maps_file['resnorm']))
    resnorm[np.isnan(resnorm) | np.isinf(resnorm)] = 0.0

    snr = np.transpose(np.array(maps_file['snr']))
    snr[np.isnan(snr) | np.isinf(snr)] = 0.0

    alpha = np.transpose(np.array(maps_file['alpha']))
    alpha[np.isnan(alpha) | np.isinf(alpha)] = 0.0

    # save NIfTIs
    mgre_img = nib.load(mgre_img_path)
    mask_img = nib.load(mask_img_path)
    np.savetxt(os.path.join(output_study_full_path, input_study_path + '_decaes-t2times.txt'), t2_times)
    nib.save(img=nib.Nifti1Image(dataobj=t2_dist, affine=mgre_img.affine, header=mgre_img.header),
             filename=os.path.join(output_study_full_path, input_study_path + '_decaes-t2dist.nii.gz'))
    nib.save(img=nib.Nifti1Image(dataobj=mwf, affine=mask_img.affine, header=mask_img.header),
             filename=os.path.join(output_study_full_path, input_study_path + '_decaes-mwf.nii.gz'))
    nib.save(img=nib.Nifti1Image(dataobj=iewf, affine=mask_img.affine, header=mask_img.header),
             filename=os.path.join(output_study_full_path, input_study_path + '_decaes-iewf.nii.gz'))
    nib.save(img=nib.Nifti1Image(dataobj=1e3*small_gm, affine=mask_img.affine, header=mask_img.header),
             filename=os.path.join(output_study_full_path, input_study_path + '_decaes-gmT2(sp).nii.gz'))
    nib.save(img=nib.Nifti1Image(dataobj=1e3*med_gm, affine=mask_img.affine, header=mask_img.header),
             filename=os.path.join(output_study_full_path, input_study_path + '_decaes-gmT2(mp).nii.gz'))
    nib.save(img=nib.Nifti1Image(dataobj=resnorm, affine=mask_img.affine, header=mask_img.header),
             filename=os.path.join(output_study_full_path, input_study_path + '_decaes-resnorm.nii.gz'))
    nib.save(img=nib.Nifti1Image(dataobj=snr, affine=mask_img.affine, header=mask_img.header),
             filename=os.path.join(output_study_full_path, input_study_path + '_decaes-snr.nii.gz'))
    nib.save(img=nib.Nifti1Image(dataobj=alpha, affine=mask_img.affine, header=mask_img.header),
             filename=os.path.join(output_study_full_path, input_study_path + '_decaes-alpha.nii.gz'))


def nnls_mese(input_study_path: str, reg: str):

    # set up paths and create directories, if needed; set the correct template for NNLS
    input_study_full_path = os.path.join(int_data_dir, input_study_path, 'mese')
    if reg == 'chi2':
        output_study_full_path = os.path.join(int_data_dir, input_study_path, 'mese/decaes_chi2')
        os.makedirs(output_study_full_path, exist_ok=True)
        settings_template_path = os.path.join(project_dir, 'config/settings_template_mese_chi2.txt')

    elif reg == 'lcurve':
        output_study_full_path = os.path.join(int_data_dir, input_study_path, 'mese/decaes_lcurve2')
        os.makedirs(output_study_full_path, exist_ok=True)
        settings_template_path = os.path.join(project_dir, 'config/settings_template_mese_lcurve.txt')
    elif reg == 'none':
        output_study_full_path = os.path.join(int_data_dir, input_study_path, 'mese/decaes_noreg')
        os.makedirs(output_study_full_path, exist_ok=True)
        settings_template_path = os.path.join(project_dir, 'config/settings_template_mese_noreg.txt')

    else:
        raise NotImplemented

    # instantiate the paths and load the images
    mese_img_path = os.path.join(input_study_full_path, input_study_path + '_mese-preproc.nii.gz')
    mask_img_path = os.path.join(input_study_full_path, input_study_path + '_mese-binary-mask.nii.gz')

    # load and modify the template
    with open(settings_template_path, 'r') as file:
        lines = file.readlines()

    lines[0] = mese_img_path + '\n'
    lines[2] = output_study_full_path + '\n'
    lines[4] = mask_img_path + '\n'

    with open(settings_template_path, 'w') as file:
        file.writelines(lines)

    # execute DECAES calculations command in python
    os.system(f"julia --project=@decaes --threads=auto -e 'using DECAES; main()' \
                @{settings_template_path}")

    # load the resultant data
    decaes_outputs = os.listdir(output_study_full_path)
    dist_file = h5py.File(
        os.path.join(output_study_full_path, [file for file in decaes_outputs if 't2dist.mat' in file][0]), 'r')
    maps_file = h5py.File(
        os.path.join(output_study_full_path, [file for file in decaes_outputs if 't2maps.mat' in file][0]), 'r')
    parts_file = h5py.File(
        os.path.join(output_study_full_path, [file for file in decaes_outputs if 't2parts.mat' in file][0]), 'r')

    # get the t2 times and NNLS distribution
    t2_times = np.array(maps_file['t2times'])
    t2_dist = np.transpose(np.array(dist_file['dist']))
    t2_dist[np.isnan(t2_dist) | np.isinf(t2_dist)] = 0.0

    # get the MWF, IEWF, gmT2(sp), and gmT2(mp)
    mwf = np.transpose(np.array(parts_file['sfr']))
    mwf[np.isnan(mwf) | np.isinf(mwf)] = 0.0

    iewf = np.transpose(np.array(parts_file['mfr']))
    iewf[np.isnan(mwf) | np.isinf(mwf)] = 0.0

    small_gm = np.transpose(np.array(parts_file['sgm']))
    small_gm[np.isnan(small_gm) | np.isinf(small_gm)] = 0.0

    med_gm = np.transpose(np.array(parts_file['mgm']))
    med_gm[np.isnan(med_gm) | np.isinf(med_gm)] = 0.0

    # get the residuals and snr - maybe we need the chi2 factor maps ('chi2factor') and reg param maps ('mu') later
    resnorm = np.transpose(np.array(maps_file['resnorm']))
    resnorm[np.isnan(resnorm) | np.isinf(resnorm)] = 0.0

    snr = np.transpose(np.array(maps_file['snr']))
    snr[np.isnan(snr) | np.isinf(snr)] = 0.0

    alpha = np.transpose(np.array(maps_file['alpha']))
    alpha[np.isnan(alpha) | np.isinf(alpha)] = 0.0

    # save NIfTIs
    mese_img = nib.load(mese_img_path)
    mask_img = nib.load(mask_img_path)
    np.savetxt(os.path.join(output_study_full_path, input_study_path + '_decaes-t2times.txt'), t2_times)

    nib.save(img=nib.Nifti1Image(dataobj=t2_dist, affine=mese_img.affine, header=mese_img.header),
             filename=os.path.join(output_study_full_path, input_study_path + '_decaes-t2dist.nii.gz'))
    nib.save(img=nib.Nifti1Image(dataobj=mwf, affine=mask_img.affine, header=mask_img.header),
             filename=os.path.join(output_study_full_path, input_study_path + '_decaes-mwf.nii.gz'))
    nib.save(img=nib.Nifti1Image(dataobj=iewf, affine=mask_img.affine, header=mask_img.header),
             filename=os.path.join(output_study_full_path, input_study_path + '_decaes-iewf.nii.gz'))
    nib.save(img=nib.Nifti1Image(dataobj=1e3 * small_gm, affine=mask_img.affine, header=mask_img.header),
             filename=os.path.join(output_study_full_path, input_study_path + '_decaes-gmT2(sp).nii.gz'))
    nib.save(img=nib.Nifti1Image(dataobj=1e3 * med_gm, affine=mask_img.affine, header=mask_img.header),
             filename=os.path.join(output_study_full_path, input_study_path + '_decaes-gmT2(mp).nii.gz'))
    nib.save(img=nib.Nifti1Image(dataobj=resnorm, affine=mask_img.affine, header=mask_img.header),
             filename=os.path.join(output_study_full_path, input_study_path + '_decaes-resnorm.nii.gz'))
    nib.save(img=nib.Nifti1Image(dataobj=snr, affine=mask_img.affine, header=mask_img.header),
             filename=os.path.join(output_study_full_path, input_study_path + '_decaes-snr.nii.gz'))
    nib.save(img=nib.Nifti1Image(dataobj=alpha, affine=mask_img.affine, header=mask_img.header),
             filename=os.path.join(output_study_full_path, input_study_path + '_decaes-alpha.nii.gz'))




if __name__ == '__main__':
    os.chdir(int_data_dir)
    study_dirs = sorted([name for name in os.listdir(os.getcwd()) if os.path.isdir(os.path.join(os.getcwd(), name))])
    for curr_study_path in study_dirs:
        print(curr_study_path)
        #nnls_mgre(input_study_path=curr_study_path, reg='chi2')
        nnls_mese(input_study_path=curr_study_path, reg='lcurve')
    os.chdir(project_dir)
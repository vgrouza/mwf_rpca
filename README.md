# Myelin water fraction imaging with robust principal component analysis
Implementation of the BSS-rPCA by Song et al (2020) for MWI using mGRE MRI data.
All methods are described in the following publication: https://doi.org/10.1016/j.neuroimage.2024.120850

All code included to reproduce the analysis carried out for rPCA of pre-clinical mGRE data. Specifically:
-mgre_recon_preproc.py leverages a custom reconstrution method implemented in pv_custom_recon.py, and carries out correction for bipolar gradient misalignment using methods implemented in bipolar_corrections.py
-dwi_recon_preproc.py similarly utilizes a custom reconstruction method to produce and correct diffusion weighted (dwGRASE) data.
-dwi_dti_noddi.py demonstrates how to fit the diffusion tensor model as well as NODDI to pre-clinical MRI data.
-rpca_api.py demonstrates how to run the rPCA analysis on mGRE data, implemented in mwf_rpca.py
-decaes_api.py demonstrates how to run DECAES on MESE data, in a Python-friednly way.

Reproducible Code for "Spatial Bayesian Neural Networks"
================================
*Andrew Zammit-Mangion, Michael D. Kaminski, Ba-Hien Tran, Maurizio Filippone, Noel Cressie*

Spatial Bayesian neural networks (SBNNs) are Bayesian neural networks that are adapted for modelling spatial processes. They incorporate a spatial embedding layer and, possibly, spatially varying weights and biases. This page provides reproducible code for the paper "Spatial Bayesian Neural Networks". Specifically, this code will calibrate BNNs and SBNNs to three "target" processes: a stationary Gaussian process, a non-stationary Gaussian process, and a stationary lognormal process. Results, saved in the `figures/` directory, show that the SBNN variants can reproduce properties of these target processes better than classical BNNs of similar complexity.

<img align="right" src="https://github.com/andrewzm/bnn-spatial/assets/150125/2321d1a6-6ddd-4439-b620-6fb0681cdaf7" alt="drawing" width="100%"/>

<img align="right" src="https://github.com/andrewzm/bnn-spatial/assets/150125/0729ff81-4b2e-43cc-898d-478bdf39d6be" alt="drawing" width="100%"/>


## Repo Structure

The repository contains the following structure:
```
├── bnn_spatial/            # Python package
├── src/                    # Source scripts folder
    ├── configs/            # Configuration files
    ├── figures/            # Output figures
    ├── BNN_degeneracy.py   # Script to gen. data for Fig. 1
    ├── FitModels.py        # Script to gen. data for sim studies
    ├── Plot_Results.R      # Plot all figures
    ├── utils.R             # R utility functions
├── LICENSE                 # MIT License
├── Makefile                # Run all sim. experiments and plot results
├── Readme.md               # This file

```

## Instructions

We suggest installing the following conda environment

```
conda create -n bnn-spatial pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```

Activate the conda environment, then install the required python packages

```
pip3 install matplotlib scipy seaborn pandas pyyaml
```

(Note: On Windows, after the installation of the python packages, we had to uninstall and re-install `pillow` for the code to work. Simply do `pip3 uninstall pillow` and then `pip install pillow`.)

Clone the repository, activate the conda environment and then call `make`. This will generate the data for Figure 1 and also run all the simulation experiments according to the configurations given in the `configs/` directory. Note that about 6 GB of GPU memory is required for the BNN and SBNN-SI variants. Around 76 GB is required for the SBNN-SV variants, possibly due to some coding inefficiencies. SBNN-SVs can therefore only be calibrated on high-end GPUs (in our case we used an NVIDIA A100). It may take a few days to reproduce the results for all the experiments. If you would just like to run one example, we suggest the following:

```
python src/FitModels.py --config src/configs/Section4_1_SBNN-SI.yaml
```

After the simulations are complete, `make` will call `src/Plot_Results.R` to generate all manuscript figures. Please make sure you have the packages listed at the top of this R script installed.

**Note: The code may not work with the latest versions of PyTorch (> 1.9.0)**

Exact versions used: Python 3.7.13, PyTorch 1.9.0+cu111, NumPy 1.21.5, SciPy 1.7.3, Matplotlib 3.5.3, Seaborn 0.12.1, pandas 1.3.5


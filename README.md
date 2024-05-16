# dp_tutorial

This folder contains the files and resources for a minimal re-implementation of [DeepPhase](https://github.com/sebastianstarke/AI4Animation/tree/master/AI4Animation/SIGGRAPH_2022/PyTorch). The original implementation contains only the barebone model, while this repository contains the full pipeline powered by Pytorch Lightning and Wandb.

Author: Dexter Tsin

Email: [dexter.tsin@princeton.edu](mailto:dexter.tsin@princeton.edu)

Last updated: 5/15/2024

## Overview

The `dp_tutorial` folder is organized as follows:

- `data/`: Dataset folder. The data used in this tutorail can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1CqdjAgz6hCRHMY6L6MFcNBRrEfuIwLee?usp=sharing). The dataset is originally collected by Berman et al., 2014 and retracked with SLEAP. The dataset contains 2D pose data and their corresponding behavior labels (Wing grooming (Left/ Right), Locomotion (slowest, medium, fastest), Anterior grooming, and idle).
- `expts_wandb/`: Contains the trained models and model checkpoints.
- `base/`: Contains all models and dataset classes.
- `notebooks/`: Contains Jupyter notebooks with preliminary analysis and rephasing experiments.
- `train_scripts/`: Contains utility scripts used in the tutorial.

## Getting Started

To get started with the tutorial, follow these steps:

1. Clone this repository to your local machine.
2. Install the required dependencies listed in the `environment.yml` file through this command `conda env create -f environment.yml -p /scratch/gpfs/{your netID}/.conda/envs/deepphase` in Della.
3. Train the model with the scripts in the `train_scripts/` folder using `train_model.sh` script.
4. Running the Jupyter notebooks in the `notebooks/` folder to analyze results.

## License

This project is licensed under the [MIT License](LICENSE.md).

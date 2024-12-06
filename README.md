# OS-induction-paper
This repository contains the source code utilized to solve both the Orr-Sommerfeld equation and the coupled Orr-Sommerfeld and induction equations relevant to the research paper:

- A. Piterskaya, M. Mortensen; "A study of the Orr-Sommerfeld and induction equations by Galerkin and Petrov-Galerkin spectral methods utilizing Chebyshev polynomials". Journal of Computational and Applied Mathematics, Volume 459, 2025, 116374. https://doi.org/10.1016/j.cam.2024.116374

The model described in the paper has been implemented within the spectral Galerkin framework Shenfun (https://github.com/spectralDNS/shenfun), version 4.1.1.

To facilitate the conda installation process, kindly refer to the 'environment.yml' file, which contains a comprehensive list of dependencies required to establish a fully operational shenfun environment.

# Codespace

The code in this repository can be tested using a codespace. Press the green Code button and choose to "create a codespace on main". A virtual machine will then be created with all the required software in environment.yml installed in a coda environment. To activate this do

    source activate ./venv

in the terminal of the codespace after the installation is finished. You may then run the program using

    python OrrSommerfeld_eigs.py

Please also refer to the Jupyter Notebook file 'analysis.ipynb' for examples on calculating eigenvalues and condition numbers for both the Orr-Sommerfeld equation and the coupled Orr-Sommerfeld and induction equations.

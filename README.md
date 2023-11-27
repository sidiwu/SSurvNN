# SSurvNN

This repo consists of the data sets and R code for the simulation studies and real application described in the manuscript "Simplified Survival Neural Networks Based on a Loss Function-Free
Transformation on Time-to-Event Outcomes". You should find three .R files and one folder named "datasets" in this repo. 

The three .R files are:

- **functions.R**: R script containing the backbone for the other scripts. It includes the core functions used for transformation (step I) and feature extraction (step II). This file needs to be sourced.

- **real_applications.R**: R script producing the results for 3 data sets (METABRIC, GBSG & FLCHAIN) given in the real applications section (Found in Section 4).

- **simulation_studies.R**: R script producing the results for various simulation settings under 3 different scenarios presented in the simulation studies section (Found in Section 3). For each scenario, we considered 4 simulated datasets with sample size of 500 or 5000 and censoring rate of 0.3 or 0.6.

The "datasets" folder contains two sub-folders:

- **real_datasets** folder: A folder containing two real datasets, metabric.csv & gbsg.csv, used for real applications.

- **sim_datasets** folder: A folder containing 12 .RData files, with each file being a simulated dataset. Files starting with "sim_prop" are for Scenario 1: proportional & linear; Files starting with "sim_nonlinear" are for Scenario 2: proportional & nonlinear; Files starting with "sim_nonprop" are for Scenario 3: nonproportional & nonlinear. "N500" and "N5000" indicate that the simulated datasets are in size of 500 and 5000, respectively, while "censor0.3" and "censor0.6" indicate that the datasets have approximate 30% and 60% censoring individuals, separately.

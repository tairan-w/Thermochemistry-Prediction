# DeepTherm
This is the repository for the paper "DeepTherm: A Unified Deep Learning Approach to ‎Thermochemistry Prediction for Gas-phase Molecules‎".

DeepTherm is a deep learning framework designed to predict thermochemical properties, such as enthalpy of formation, entropy, and heat capacity for diverse molecular species. The project integrates directed message-passing networks and global attention mechanisms to capture both local and long-range dependencies in molecular graphs.

The datasets are provided along with the paper.

The code was built based on [DMPNN](https://github.com/chemprop/chemprop). Thanks a lot for their code sharing!

## Dependencies

+ cuda >= 8.0
+ cuDNN
+ RDKit
+ torch >= 1.2.0

Tips: Using code `conda install -c rdkit rdkit` can help you install package RDKit quickly.

## Directory Structure
├── data/                   # Contains data processing scripts
├── features/               # Contains feature extraction scripts
├── models/                 # Contains model definitions
├── train/                  # Contains training, evaluation, and cross-validation scripts

## Environment


Anaconda was used to create the virtual environment for this project. Feel free to use one of the following commands to set up the required environment:

Conda commands:

```
conda create -n DeepTherm python=3.6
conda activate DeepTherm  
conda install pandas keras scikit-learn xlrd
``` 


## Overview of files

It must be noted that the files for SVR and ANN are very similar and could be combined in to single files. They were kept separate for this repository in order to reduce ambiguity.


### data
- **dataset\_complete.csv**: Complete dataset used for the paper
- **dataset\_processed.csv**: Dataset filtered to the features used for the models
- **octene\_isomers.csv**: The Octene isomer dataset - filtering happens within the `inference.py` script.
- **nonyne\_isomers.csv**: The Nonyne isomer dataset - filtering happens within the `inference.py` script.
- **Training dataset.csv**: The training dataset for the final model.

### models
- **Transfer\_learing**: Trained graph neural network model with cross-validation of 5 fold and ensemble learning of 5 models.
- **final\_ANN\_model.pkl**: Trained ANN model with the best found combination of hyperparameters. 
  `{'batch_size': 128, 'epochs': 5000, 'l1': 80, 'l2': 80, 'loss': 'mean_absolute_error', 'r1': 0.1, 'r2': 0.2}`
- **final\_SVR\_model.pkl**: Trained SVR model with the best found combination of hyperparameters. 
  `{'C': 6000, 'epsilon': 0.15}`


### scripts
- **run\_base\_models.py**: Running the base model with message-passing neural networks.
- **run\_transfer\_models.py**: Transfer model parameters to a new model. Any number of model layers can be frozen. If the models are branched, any number of branches can be transferred.
- **error\_estimation\_ann.py**: 10 fold grid search over each of 10 folds of the entire dataset in order to estimate the prediction abilities of an ANN.
- **error\_estimation\_svr.py**: 10 fold grid search over each of 10 folds of the entire dataset in order to estimate the prediction abilities of a SVR.
- **final\_model\_ann.py**: The script used to generate the final ANN model, `final_ANN_model.pkl`, found through a 10 fold grid search over the entire dataset.
- **final\_model\_svr.py**: The script used to generate the final SVR model, `final_SVR_model.pkl`, found through a 10 fold grid search over the entire dataset.
- **inference.py**: The script in order to infer the enthalpy of the Nonane isomers. Requires a command line argument specifying whether to use the ANN or the SVR.
- **process\_dataset.py**: The script used to create `dataset_processed.csv` from `dataset_complete.csv`.
- **sensitivity\_analysis.py**: The script used in order to run a sensitivity analysis over the final SVR model.


### results
- **grid\_search_ann.csv**: The complete results from the grid search for the ANN
- **grid\_search\_svr.csv**: The complete results from the grid search for the SVR



### Authorship  

Tairan Wang was responsible for the code and data.

### Acknowledgement 

This code was developed at the King Abdullah University of Science and Technology (KAUST).



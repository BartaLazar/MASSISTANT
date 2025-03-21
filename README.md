# MASSISTANT

![img.png](img.png)


## What is MASSISTANT?

MASSISTANT is a tool for predicting de novo SELFIES representation of a molecule from its EI mass spectra. Developed by Lazar Barta at Envalior Materials as part of a master internship at Maastricht University.

**Disclaimer**: The models were trained on the NIST EI mass spectra dataset and is not provided. You can find an example of the required format in the `Dataset` folder.

## Approach

## Train your own model
1) Import your dataset in the `Dataset` folder.
2) Use `Code/dataset_cleaning.py` to clean your dataset. It will create the `.csv` files which will be necessary to train your models. Before running it, update the path to your dataset in the script.
3) Encode the SELFIES representation of your dataset using `Code/Full_systems/Selfies_Mol/Featurization/selfies_featurization_one_hot.ipynb`. A one-hot encoding of the SELFIES representation is necessary to train the models. This notebook will also generate the train and test data for the model.
4) To train your models, use `Code/Full_systems/Selfies_Mol/Multitask_Classifier/SELFIES_multitask_tuner.ipynb`. This notebook will train the model and save it in the `Models` folder.

## Predict SELFIES representation from mass spectra
Will be available soon.

## Files in this folder
`.env` contains environment variables for the project. You can modify the variables to your needs, but it is not necessary. Only adjust them when you are familiar with the project structure and the functioning of the code.
`.rootfloder` is a marker file to indicate the root folder of the project. It is used by the scripts to navigate to the root folder of the project. **Do not delete this file.**
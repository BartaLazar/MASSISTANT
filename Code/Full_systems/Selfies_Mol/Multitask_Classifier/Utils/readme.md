# Code/Full_systems/Selfies_Mol/Multitask_Classifier/Utils folder

This folder contains utility functions for the multitask classifier model.

## Files:
- `SELFIES_DC_multitask.py` contains all the methods necessary to tune, train and evaluate the multitask classifier models. Its methods are used by the `SELFIES_multitask_tuner.ipynb` notebook.

- `random_combinations.py` is used to generate random combinations of hyperparameters for random sampling for the tuning of the models. You can modify the bounds and steps of the hyperparameters in the script.

- `global_summary_creator.ipynb` is used to create a global summary of the models. It creates the `global_summary.csv` files in the `Models` folder and creates a global summary of the models. **Run this notebook after training the models.**

- `progress_callback.py` is a callback function used to monitor the training of the models. It is used in the `SELFIES_DC_multitask.py` script.
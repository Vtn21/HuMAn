"""train_mpihdm05.py

Train the HuMAn neural network (architecture defined in "human.py").
Uses the MPI-HDM05 dataset, for learning subject-specific motions.

Procedure types (control using the "PROCEDURE" global variable):
- Train: train the model from scratch.
- Transfer: fine-tune the universal model created using "train_universal.py".

Author: Victor T. N.
"""

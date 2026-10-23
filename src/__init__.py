
"""
Indoor Temperature Sensor Reconstruction
========================================

A numerical methods project for reconstructing missing indoor
temperature measurements using interpolation and regression.

This package provides reusable Python functions for sensor data
preprocessing, numerical reconstruction, regression modelling,
and evaluation of reconstruction accuracy.

Main Modules
------------
data_processing
    Extract, clean and organise environmental sensor measurements.

interpolation
    Create artificial missing gaps and reconstruct temperatures
    using linear and cubic spline interpolation.

ml_model
    Train and apply linear regression models using environmental
    predictor variables.

evaluation
    Calculate reconstruction errors and evaluate numerical
    reconstruction methods.

validation
    Generate missing-temperature windows and compare seven
    reconstruction methods, including endpoint-constrained regression.

Notes
-----
The project focuses on retrospective temperature reconstruction.
Some methods require known measurements on both sides of a
missing-data interval.

The complete workflow and experimental results are available
in notebooks/01_data_exploration.ipynb.
"""

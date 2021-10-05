"""
Tests for functions in crps_net.py.

Authors: Steven Brey
"""

import numpy as np
import pytest
import sys
from ensemble.crps_net import crps_sample_score

# This code is designed to work whether or not tensorflow is installed.
# However, this changes what tests can be run when package builds.
try:
    import tensorflow as tf
    TF_IMPORT_ERROR = None
except ImportError as e:
    TF_IMPORT_ERROR = e

# ------------------------------------------------------------------------------
# ------ baseline functions to compare to crps-sample score to ------
# ------------------------------------------------------------------------------
def _error(actual: np.ndarray, predicted: np.ndarray):
    """ Simple error """
    return actual - predicted

def mae(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Absolute Error """
    return np.mean(np.abs(_error(actual, predicted)))


# ------------------------------------------------------------------------------
# ------Test that crps-sample-score same as mean absolute error for n=1 -------
# ------------------------------------------------------------------------------
MIN_VALUE = -100
MAX_VALUE = 100

PAIRS = []
for i in range(100) : 
    
    # Generate random single value forecasts 
    DETERMINISTIC = np.random.uniform(low=MIN_VALUE, high=MAX_VALUE, size=(1, 1))
    # Generate random verifications
    VERIFICATION = np.random.uniform(low=MIN_VALUE, high=MAX_VALUE, size=(1, 1))
    
    PAIRS.append((DETERMINISTIC, VERIFICATION))


DECIMALS = 8

@pytest.mark.parametrize("deterministic, verification", PAIRS)
def test_same_as_mae(deterministic, verification) :
    """Assert that a single member crps-sample-score is 
    the exact same as mae"""

    mae_score  = np.round(mae(verification, deterministic), DECIMALS)

    if TF_IMPORT_ERROR is None : 
        crps_score = np.round(crps_sample_score(verification, deterministic).numpy(), DECIMALS)
    else : 
        crps_score = np.round(crps_sample_score(verification, deterministic), DECIMALS)

    assert(mae_score == crps_score),\
    (f"crps sample score {crps_score} does not equal mae score {mae_score}")
    
    
# ------------------------------------------------------------------------------
# ------------ Assert that the error is zero for perfect ensembles  ------------
# ------------------------------------------------------------------------------
PAIRS = []
for i in range(100) : 
    
    # Generate random verifications
    VERIFICATION = np.random.uniform(low=MIN_VALUE, high=MAX_VALUE, size=1)
    
    # Generate perfect ensemble by repeating the verification i+1 times
    FORECAST = np.repeat(VERIFICATION, i+1).reshape(1,-1)
    
    PAIRS.append((VERIFICATION, FORECAST))

    
@pytest.mark.parametrize("verification, forecast", PAIRS)    
def test_perfect_forecast(verification, forecast) : 
    """Assert that a perfect forecast with an ensemble of any
    size has a crps-sample-score of exactly 0"""
    
    if TF_IMPORT_ERROR is None : 
        crps_score = np.round(crps_sample_score(verification, forecast).numpy(), DECIMALS)
    else :
        crps_score = np.round(crps_sample_score(verification, forecast), DECIMALS)
    
    assert(crps_score == 0.0),\
    (f"crps_score should be 0, got {crps_score}")
    

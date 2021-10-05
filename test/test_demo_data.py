"""
Tests for functions in demo_data.py.

Authors: Steven Brey
"""

import numpy as np
from ensemble.demo_data import create_point_gamma_data


# ---------------------------------------------------------
# ------ Make sure the function works, returns data 
# ----------------------------------------------------------

# TODO: Write better tests for demo data ...

def test_create_point_gamma() : 

    x, y = create_point_gamma_data()

    assert(x is not None),\
    ("x should be numeric! not None.")

    assert(y is not None),\
    ("y should be numeric! not None.")